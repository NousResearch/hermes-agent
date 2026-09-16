"""The post-turn review fork must not overlap a live turn, nor replay an oversized transcript.

Observed pattern this pins down: a 65-call turn finishes, the automatic background review forks
with the WHOLE ~205K-token snapshot, and a queued gateway follow-up starts ~100ms later — two
>200K contexts decoding at once for the same session. The contracts asserted here:

* a live turn already registered on the session wins; the automatic review never forks;
* a queued gateway follow-up wins the same way (it becomes the next live turn);
* an oversized snapshot keeps a bounded, user-anchored recent suffix instead of replaying verbatim;
* a normal bounded snapshot still replays verbatim, so ordinary learning + warm-cache parity stay;
* the fork stays detached from the canonical session either way.
"""

from __future__ import annotations

import datetime as _dt
import functools
import hashlib
import itertools
import threading
import types
from typing import cast

import pytest

import run_agent as run_agent_module
from agent import background_review as background_review_module
from agent import review_admission
from agent.model_metadata import estimate_messages_tokens_rough
from agent.turn_facade import TurnFacadeMixin
from run_agent import AIAgent


class ImmediateThread:
    """Run the review worker inline so the assertions see a finished review."""

    def __init__(self, *, target, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


def _bare_agent(session_id: str = "overlap-session") -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "whatsapp"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.session_id = session_id
    agent._parent_session_id = ""
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._cached_system_prompt = "test-cached-system-prompt"
    agent.session_start = _dt.datetime(2026, 9, 10, 16, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.background_review_callback = None
    agent.status_callback = None
    agent._safe_print = lambda *_args, **_kwargs: None
    agent._background_review_agent = None
    agent._background_review_run = None
    agent._background_review_lock = threading.Lock()
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    return agent


@pytest.fixture
def review_forks(monkeypatch):
    """Capture every fork the review path builds, plus the history it replays."""
    forks: list[dict] = []

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []
            self.record = {"init": kwargs, "history": None, "attrs": {}}
            forks.append(self.record)

        def __setattr__(self, name, value):
            if name not in ("record", "_session_messages"):
                self.record["attrs"][name] = value
            object.__setattr__(self, name, value)

        def run_conversation(self, **kwargs):
            self.record["history"] = kwargs.get("conversation_history")

        def release_clients(self):
            pass

    monkeypatch.setattr(run_agent_module, "AIAgent", FakeReviewAgent)
    monkeypatch.setattr(
        run_agent_module, "threading", types.SimpleNamespace(Thread=ImmediateThread)
    )
    return forks


@pytest.fixture(autouse=True)
def clear_review_admission_state():
    """One failed assertion must not leak a process-global session admission into later tests."""
    with review_admission._lock:
        review_admission._live_turns.clear()
        review_admission._review_runs.clear()
        review_admission._turn_keys.clear()
        review_admission._tokens = itertools.count(1)
    yield
    with review_admission._lock:
        review_admission._live_turns.clear()
        review_admission._review_runs.clear()
        review_admission._turn_keys.clear()
        review_admission._tokens = itertools.count(1)


def _snapshot(pairs: int, filler_chars: int) -> list[dict]:
    """``pairs`` user/assistant exchanges, each assistant turn carrying ``filler_chars`` of text."""
    messages: list[dict] = []
    for i in range(pairs):
        messages.append({"role": "user", "content": f"question {i}"})
        messages.append({
            "role": "assistant",
            "content": f"answer {i} " + ("x" * filler_chars),
        })
    return messages


def _config(**task_overrides) -> dict:
    return {"auxiliary": {"background_review": {"enabled": True, **task_overrides}}}


def _patch_config(monkeypatch, cfg: dict) -> None:
    import hermes_cli.config as config_module

    monkeypatch.setattr(config_module, "load_config_readonly", lambda *a, **k: cfg)
    monkeypatch.setattr(config_module, "load_config", lambda *a, **k: cfg)


def _assert_plain_user_anchor(history: list[dict]) -> None:
    assert history, "replayed history must not be empty"
    first = history[0]
    assert first.get("role") == "user"
    content = first.get("content")
    if isinstance(content, list):
        assert not any(
            isinstance(block, dict)
            and (block.get("type") == "tool_result" or block.get("tool_use_id"))
            for block in content
        )


# ---------------------------------------------------------------------------
# 1 + 2 — the foreground always wins
# ---------------------------------------------------------------------------


def test_live_turn_on_the_same_session_blocks_the_automatic_review(
    review_forks, monkeypatch, caplog
):
    """A turn already registered on this session means the review must not fork at all."""
    _patch_config(monkeypatch, _config())
    agent = _bare_agent()
    token = review_admission.note_turn_started(agent.session_id)
    try:
        with caplog.at_level("INFO"):
            AIAgent._spawn_background_review(
                agent,
                messages_snapshot=[{"role": "user", "content": "hello"}],
                review_memory=True,
            )
    finally:
        review_admission.note_turn_finished(agent.session_id, token)

    assert review_forks == [], "review forked while a live turn held the session"
    assert review_admission.REASON_LIVE_TURN in caplog.text
    assert (
        review_admission.owner_tag(
            review_admission.current_profile_key(), agent.session_id
        )
        in caplog.text
    )


def test_second_agent_cancels_canonical_review_for_the_same_owner(monkeypatch):
    targets = []

    class CapturedThread:
        def __init__(self, *, target, daemon=None, name=None):
            targets.append(target)

        def start(self):
            pass

    _patch_config(monkeypatch, _config(defer="never"))
    monkeypatch.setattr(
        run_agent_module,
        "threading",
        types.SimpleNamespace(Thread=CapturedThread),
    )
    first = _bare_agent("shared-session")
    second = _bare_agent("shared-session")
    AIAgent._spawn_background_review(
        first,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )
    first_run = first._background_review_run
    assert first_run is not None

    cancelled = background_review_module.cancel_background_review_for_live_turn(
        second, wait=False
    )

    assert cancelled is first_run
    assert first_run.cancel_requested.is_set()
    targets[0]()


def test_rotated_parent_keeps_review_ownership_scoped_to_each_session():
    agent = _bare_agent("session-before-rotation")
    profile_key = review_admission.current_profile_key()
    first = background_review_module.prepare_background_review_run(
        agent,
        session_id="session-before-rotation",
        profile_key=profile_key,
    )
    assert first is not None

    agent.session_id = "session-after-rotation"
    second = background_review_module.prepare_background_review_run(
        agent,
        session_id="session-after-rotation",
        profile_key=profile_key,
    )

    try:
        assert second is not None
        assert (
            review_admission.current_review_run("session-before-rotation", profile_key)
            is first
        )
        assert (
            review_admission.current_review_run("session-after-rotation", profile_key)
            is second
        )
        cancelled = background_review_module.cancel_background_review_for_live_turn(
            agent,
            wait=False,
            session_id="session-after-rotation",
            profile_key=profile_key,
        )
        assert cancelled is second
        assert second.cancel_requested.is_set()
        assert first.cancel_requested.is_set() is False
    finally:
        background_review_module.finish_background_review_run(agent, second)
        background_review_module.finish_background_review_run(agent, first)


def test_rotated_parent_cancel_targets_the_registry_owner_not_the_slot():
    """After rotation the slot points at the NEWEST prepared run; cancelling the OLD session
    must resolve through the canonical registry and fence only that session's run."""
    agent = _bare_agent("session-before-rotation")
    profile_key = review_admission.current_profile_key()
    first = background_review_module.prepare_background_review_run(
        agent, session_id="session-before-rotation", profile_key=profile_key
    )
    agent.session_id = "session-after-rotation"
    second = background_review_module.prepare_background_review_run(
        agent, session_id="session-after-rotation", profile_key=profile_key
    )
    assert first is not None and second is not None
    assert agent._background_review_run is second

    try:
        cancelled = background_review_module.cancel_background_review_for_live_turn(
            agent,
            wait=False,
            session_id="session-before-rotation",
            profile_key=profile_key,
        )
        assert cancelled is first
        assert first.cancel_requested.is_set()
        assert second.cancel_requested.is_set() is False
        # An unadmitted run is revoked on the spot; its successor keeps its ownership.
        assert first.request_done.is_set()
        assert (
            review_admission.current_review_run("session-before-rotation", profile_key)
            is None
        )
        assert (
            review_admission.current_review_run("session-after-rotation", profile_key)
            is second
        )
    finally:
        background_review_module.finish_background_review_run(agent, second)
        background_review_module.finish_background_review_run(agent, first)


def test_cancelling_prepared_unstarted_review_acknowledges_without_worker():
    agent = _bare_agent("prepared-without-worker")
    run = background_review_module.prepare_background_review_run(agent)
    assert run is not None
    cancellation_returned = threading.Event()

    def cancel_and_wait():
        background_review_module.cancel_background_review_for_live_turn(agent)
        cancellation_returned.set()

    foreground = threading.Thread(target=cancel_and_wait, daemon=True)
    foreground.start()
    try:
        assert cancellation_returned.wait(2.0), (
            "an unstarted prepared run has no worker that can publish its acknowledgement"
        )
    finally:
        background_review_module.finish_background_review_run(agent, run)
        foreground.join(timeout=2.0)

    assert foreground.is_alive() is False
    assert run.request_done.is_set()
    assert review_admission.current_review_run(agent.session_id) is None


def test_live_turn_does_not_enter_conversation_loop_until_review_acknowledges(
    monkeypatch,
):
    """The foreground fences and interrupts an admitted fork, but must not start its own
    provider work until that fork publishes its request exit — with no timeout escape."""
    import agent.conversation_loop as conversation_loop_module

    agent = _bare_agent("facade-waits-for-review")
    # A full (lease-less) facade pass reads these on the way to the loop.
    agent._session_db = None
    agent._persist_disabled = False
    agent._reset_activity_labels_after_turn = lambda: None
    agent._conversation_root_id = lambda: agent.session_id
    agent.log_prefix = ""
    agent._vprint = lambda *_a, **_k: None
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._pending_redirect = None
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False

    run = background_review_module.prepare_background_review_run(agent)
    assert run is not None
    assert (
        run.begin_request(object()) is True
    )  # an admitted fork with a request on the wire

    entered_wait = threading.Event()
    release = threading.Event()
    loop_entered = threading.Event()
    observed_timeouts: list = []

    class ControlledCompletion:
        def __init__(self):
            self.set_calls = 0

        def wait(self, timeout=None):
            observed_timeouts.append(timeout)
            entered_wait.set()
            assert release.wait(timeout=10.0)
            return True

        def set(self):
            self.set_calls += 1

        def is_set(self):
            return self.set_calls > 0

    run.request_done = ControlledCompletion()
    monkeypatch.setattr(
        background_review_module, "_interrupt_background_review", lambda _fork: None
    )

    def fake_run(_agent, _message, _system, history, *_args, **_kwargs):
        loop_entered.set()
        return {"final_response": "ok", "messages": history or [], "failed": False}

    monkeypatch.setattr(conversation_loop_module, "run_conversation", fake_run)

    outcome: dict = {}

    def foreground():
        try:
            outcome["result"] = TurnFacadeMixin.run_conversation(agent, "hi")
        except BaseException as exc:  # noqa: BLE001 — surfaced by the assertions below
            outcome["error"] = exc

    turn = threading.Thread(target=foreground, daemon=True)
    turn.start()
    try:
        assert entered_wait.wait(2.0)
        assert run.cancel_requested.is_set()
        # The waiter is parked on the fork's exit: the loop cannot have been entered.
        assert loop_entered.is_set() is False
    finally:
        release.set()
        turn.join(timeout=10.0)
        background_review_module.finish_background_review_run(agent, run)

    assert not turn.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert observed_timeouts == [None]
    assert loop_entered.is_set()
    assert outcome["result"]["final_response"] == "ok"
    assert review_admission.other_live_turn(agent.session_id, None) is False


def test_review_uses_the_durable_turn_lease_shared_by_other_processes(tmp_path):
    import os

    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    review_db = SessionDB(path)
    foreground_db = SessionDB(path)
    review_db.create_session("shared-session", source="test")
    parent = types.SimpleNamespace(_session_db=review_db)
    review_agent = types.SimpleNamespace()
    run = background_review_module._BackgroundReviewRun()

    lease, reason = background_review_module._try_acquire_durable_review_lease(
        parent, review_agent, "shared-session", run
    )

    assert reason is None
    assert lease is not None
    foreground_holder = f"pid={os.getpid()}:turn=foreground"
    assert not foreground_db.try_acquire_session_turn_lease(
        "shared-session", foreground_holder, ttl_seconds=5
    )
    lease.release()
    assert foreground_db.try_acquire_session_turn_lease(
        "shared-session", foreground_holder, ttl_seconds=5
    )
    foreground_db.release_session_turn_lease("shared-session", foreground_holder)


def test_queued_gateway_followup_blocks_the_automatic_review(
    review_forks, monkeypatch, caplog
):
    """A follow-up already queued for this session becomes the next live turn — it wins."""
    _patch_config(monkeypatch, _config())
    agent = _bare_agent()
    agent.followup_pending_callback = lambda: True

    with caplog.at_level("INFO"):
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
        )

    assert review_forks == [], (
        "review forked while a follow-up was queued for the session"
    )
    assert review_admission.REASON_QUEUED_FOLLOWUP in caplog.text
    assert (
        review_admission.owner_tag(
            review_admission.current_profile_key(), agent.session_id
        )
        in caplog.text
    )


def test_raising_followup_probe_fails_safe_and_blocks_automatic_review(
    review_forks, monkeypatch, caplog
):
    """Unknown foreground state cannot authorize lower-priority provider work."""
    _patch_config(monkeypatch, _config())
    agent = _bare_agent()

    def raise_from_probe():
        raise RuntimeError("probe failed")

    agent.followup_pending_callback = raise_from_probe
    with caplog.at_level("INFO"):
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
        )

    assert review_forks == []
    assert "admission_probe_failed" in caplog.text
    assert (
        review_admission.owner_tag(
            review_admission.current_profile_key(), agent.session_id
        )
        in caplog.text
    )


def test_raising_request_admission_gate_fails_safe():
    def raise_from_gate():
        raise RuntimeError("gate failed")

    run = background_review_module._BackgroundReviewRun(admission_gate=raise_from_gate)

    assert run.begin_request(object()) is False
    assert run.refused_reason == "admission_probe_failed"
    assert run.cancel_requested.is_set()


def test_live_turn_starting_after_early_gate_fences_the_prepared_review(
    review_forks,
    monkeypatch,
    caplog,
):
    """Close the check-before-prepare race: once a run is installed, a new turn must win."""
    _patch_config(monkeypatch, _config())
    agent = _bare_agent()
    original_prepare = background_review_module.prepare_background_review_run
    race_token = None

    def prepare_then_start_live_turn(parent, **kwargs):
        nonlocal race_token
        run = original_prepare(parent, **kwargs)
        race_token = review_admission.note_turn_started(parent.session_id)
        return run

    monkeypatch.setattr(
        background_review_module,
        "prepare_background_review_run",
        prepare_then_start_live_turn,
    )
    try:
        with caplog.at_level("INFO"):
            AIAgent._spawn_background_review(
                agent,
                messages_snapshot=[{"role": "user", "content": "hello"}],
                review_memory=True,
            )
    finally:
        if race_token is not None:
            review_admission.note_turn_finished(
                getattr(agent, "session_id", ""), race_token
            )

    assert review_forks == [], (
        "review crossed a live turn that started after its early gate"
    )
    assert review_admission.REASON_LIVE_TURN in caplog.text


def test_foreground_registration_and_review_publication_are_linearized_without_sleep():
    """Force a live turn into the old sample-to-publication gap."""

    class TrackingRLock:
        def __init__(self):
            self._lock = threading.RLock()
            self._owner = None
            self._depth = 0

        def __enter__(self):
            self._lock.acquire()
            owner = threading.get_ident()
            if self._owner == owner:
                self._depth += 1
            else:
                self._owner = owner
                self._depth = 1
            return self

        def __exit__(self, *_args):
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
            self._lock.release()

        def held_by_current_thread(self):
            return self._owner == threading.get_ident()

    session_id = "forced-registry-race"
    live_registered = threading.Event()
    live_finished = threading.Event()
    published_while_live = []
    cancelled_forks = []
    live_tokens = []
    live_threads = []
    review_agent = object()
    original_lock = review_admission._lock
    admission_lock = TrackingRLock()
    review_admission._lock = admission_lock

    class ObservedRun(background_review_module._BackgroundReviewRun):
        def __setattr__(self, name, value):
            if (
                name == "_review_agent"
                and value is review_agent
                and getattr(self, "_observe_publication", False)
            ):
                published_while_live.append(live_registered.is_set())
            super().__setattr__(name, value)

    def register_and_cancel_live_turn():
        with admission_lock:
            token = review_admission.note_turn_started(session_id, "/profile")
            live_tokens.append(token)
            live_registered.set()
            cancelled_forks.append(run.cancel())
        live_finished.set()

    def foreground_gate():
        assert review_admission.other_live_turn(session_id, None, "/profile") is False
        live_thread = threading.Thread(target=register_and_cancel_live_turn)
        live_threads.append(live_thread)
        live_thread.start()
        # The fixed path owns the registry lock and must not wait for the contender. The old path
        # did not, so force the live token into its stale sample-to-publication window.
        if not admission_lock.held_by_current_thread():
            assert live_registered.wait(timeout=1.0)
        return None

    try:
        run = ObservedRun(
            admission_gate=foreground_gate,
            foreground_admission_lock=review_admission.admission_lock(),
        )
        run._observe_publication = True

        assert run.begin_request(review_agent) is True
        assert len(live_threads) == 1
        assert live_finished.wait(timeout=1.0)
        live_threads[0].join(timeout=1.0)

        assert published_while_live == [False]
        assert cancelled_forks == [review_agent]
        assert run.cancel_requested.is_set()
    finally:
        for token in live_tokens:
            review_admission.note_turn_finished(session_id, token, "/profile")
        review_admission._lock = original_lock


def test_queue_side_cancel_is_never_blocked_by_the_foreground_probe():
    """The run lock must not be held across the host foreground probe.

    The probe is the gateway's, and it takes that session's admission lock; the gateway's queue
    fence takes the SAME admission lock and then cancels, which takes the run lock. Holding the run
    lock across the probe inverts the two orders, so the fence — running on the gateway event loop —
    would wedge behind a review that is itself waiting on the loop's lock.
    """
    probe_entered = threading.Event()
    release_probe = threading.Event()
    cancel_returned = threading.Event()
    admitted: list = []

    def parked_gate():
        probe_entered.set()
        release_probe.wait(timeout=10.0)
        return None

    run = background_review_module._BackgroundReviewRun(admission_gate=parked_gate)
    requester = threading.Thread(
        target=lambda: admitted.append(run.begin_request(object()))
    )
    canceller = threading.Thread(
        target=lambda: (run.cancel(), cancel_returned.set()),
    )
    try:
        requester.start()
        assert probe_entered.wait(timeout=10.0)
        canceller.start()
        fence_blocked = not cancel_returned.wait(timeout=10.0)
    finally:
        release_probe.set()
        requester.join(timeout=10.0)
        canceller.join(timeout=10.0)

    assert not fence_blocked, (
        "queue-side cancel blocked behind the review's foreground probe"
    )
    assert admitted == [False]
    assert run.cancel_requested.is_set()


def test_followup_arriving_after_post_prepare_check_blocks_provider_request(
    review_forks, monkeypatch
):
    """Force the late queue interleaving without sleep: admission re-checks after fork build."""
    _patch_config(monkeypatch, _config())
    agent = _bare_agent()
    queued = {"value": False}
    agent.followup_pending_callback = lambda: queued["value"]
    original_build = background_review_module.build_cache_parity_fork

    def build_then_queue(*args, **kwargs):
        fork = original_build(*args, **kwargs)
        queued["value"] = True
        return fork

    monkeypatch.setattr(
        background_review_module, "build_cache_parity_fork", build_then_queue
    )

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert len(review_forks) == 1, (
        "the forced interleaving must reach fork construction"
    )
    assert review_forks[0]["history"] is None, (
        "provider-capable run_conversation was admitted"
    )
    assert agent._background_review_run is None


def test_admission_gate_refusal_logs_one_body_free_line(
    review_forks, monkeypatch, caplog
):
    """The last-word gate is the one that actually closes the race, so it must leave a trace.

    Every other skip/defer decision logs ``reason slug + hashed owner tag``; a refusal at
    ``begin_request`` used to log nothing at all, so the cheapest signal that self-improvement is
    being starved was invisible. One INFO record, no message bodies, no raw session id, and the
    same profile+session hash every other review line carries.
    """
    _patch_config(monkeypatch, _config())
    session_id = "raw-session-id-must-not-be-logged"
    agent = _bare_agent(session_id)
    queued = {"value": False}
    agent.followup_pending_callback = lambda: queued["value"]
    original_build = background_review_module.build_cache_parity_fork

    def build_then_queue(*args, **kwargs):
        fork = original_build(*args, **kwargs)
        queued["value"] = True
        return fork

    monkeypatch.setattr(
        background_review_module, "build_cache_parity_fork", build_then_queue
    )

    with caplog.at_level("INFO"):
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[{"role": "user", "content": "private message body"}],
            review_memory=True,
        )

    assert review_forks[0]["history"] is None, (
        "provider-capable run_conversation was admitted"
    )
    refusals = [
        record
        for record in caplog.records
        if review_admission.REASON_QUEUED_FOLLOWUP in record.getMessage()
    ]
    assert len(refusals) == 1, "gate refusal must log exactly once"
    assert refusals[0].levelname == "INFO"
    refusal_text = refusals[0].getMessage()
    assert (
        review_admission.owner_tag(review_admission.current_profile_key(), session_id)
        in refusal_text
    )
    assert session_id not in refusal_text
    assert "private message body" not in refusal_text


@pytest.mark.parametrize(
    "hook_name,expected_line",
    [
        ("prepare_background_review_run", "skipped after prepare"),
        ("build_cache_parity_fork", "refused at admission"),
    ],
)
def test_late_gate_skip_line_tags_the_session_captured_at_spawn(
    review_forks, monkeypatch, caplog, hook_name, expected_line
):
    """A parent may rotate sessions while its review is still being admitted; each late gate's
    skip line must name the frozen review owner, not whatever the parent points at by then.

    ``hook_name`` is the last production step before the gate: the post-prepare re-check runs
    right after ``prepare_background_review_run``; ``begin_request`` right after
    ``build_cache_parity_fork``. Both hooks rotate the parent and queue a follow-up, so the gate
    refuses and logs — which session tag it carries is the contract.
    """
    _patch_config(monkeypatch, _config())
    agent = _bare_agent("session-one")
    profile_key = review_admission.current_profile_key()
    queued = {"value": False}
    agent.followup_pending_callback = lambda: queued["value"]
    original = getattr(background_review_module, hook_name)

    def hook_then_rotate(*args, **kwargs):
        result = original(*args, **kwargs)
        agent.session_id = "session-two"
        queued["value"] = True
        return result

    monkeypatch.setattr(background_review_module, hook_name, hook_then_rotate)

    with caplog.at_level("INFO"):
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
        )

    assert all(fork["history"] is None for fork in review_forks), (
        "provider-capable run_conversation was admitted"
    )
    assert agent._background_review_run is None
    refusals = [
        record
        for record in caplog.records
        if review_admission.REASON_QUEUED_FOLLOWUP in record.getMessage()
    ]
    assert len(refusals) == 1, "gate refusal must log exactly once"
    refusal_text = refusals[0].getMessage()
    assert expected_line in refusal_text
    assert review_admission.owner_tag(profile_key, "session-one") in refusal_text
    assert review_admission.owner_tag(profile_key, "session-two") not in refusal_text


def test_early_gate_skip_line_tags_the_frozen_review_session(
    review_forks, monkeypatch, caplog
):
    """A gateway candidate freezes its owner at finalize and spawns after delivery; when the
    parent rotated in between, the skip line must name the frozen owner the gate checked."""
    _patch_config(monkeypatch, _config())
    agent = _bare_agent("session-two")
    profile_key = review_admission.current_profile_key()
    token = review_admission.note_turn_started("session-one", profile_key)
    try:
        with caplog.at_level("INFO"):
            AIAgent._spawn_background_review(
                agent,
                messages_snapshot=[{"role": "user", "content": "hello"}],
                review_memory=True,
                _review_profile_key=profile_key,
                _review_session_id="session-one",
            )
    finally:
        review_admission.note_turn_finished("session-one", token, profile_key)

    assert review_forks == [], "review forked while a live turn held the frozen session"
    skips = [
        record
        for record in caplog.records
        if review_admission.REASON_LIVE_TURN in record.getMessage()
    ]
    assert len(skips) == 1, "early gate skip must log exactly once"
    skip_text = skips[0].getMessage()
    assert review_admission.owner_tag(profile_key, "session-one") in skip_text
    assert review_admission.owner_tag(profile_key, "session-two") not in skip_text


def test_cancelled_review_does_not_log_a_gate_refusal(
    review_forks, monkeypatch, caplog
):
    """A live turn cancelling the run is already logged by the canceller — no second line."""
    _patch_config(monkeypatch, _config())
    agent = _bare_agent()
    original_build = background_review_module.build_cache_parity_fork

    def build_then_cancel(*args, **kwargs):
        fork = original_build(*args, **kwargs)
        agent._background_review_run.cancel()
        return fork

    monkeypatch.setattr(
        background_review_module, "build_cache_parity_fork", build_then_cancel
    )

    with caplog.at_level("INFO"):
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
        )

    assert review_forks[0]["history"] is None, "cancelled run reached the provider"
    assert "refused at admission" not in caplog.text


@pytest.mark.parametrize(
    "canceller,expected_reason",
    [
        (
            background_review_module.cancel_background_review_for_pending_followup,
            "queued_followup_cancelled",
        ),
        (
            functools.partial(
                background_review_module.cancel_background_review_for_live_turn,
                wait=False,
            ),
            "live_turn_cancelled",
        ),
    ],
    ids=["queue_side_fence", "live_turn"],
)
def test_cancellation_logs_owner_and_reason_once(
    monkeypatch, caplog, canceller, expected_reason
):
    """A fence is a skip decision like any other: whoever cancels the review leaves one INFO
    line with the hashed owner and a stable slug, and a repeated fence on the same run adds
    nothing. Without it a review that vanished mid-flight was invisible in the logs."""
    agent = _bare_agent("raw-session-id-must-not-be-logged")
    profile_key = review_admission.current_profile_key()
    monkeypatch.setattr(
        background_review_module, "_interrupt_background_review", lambda _fork: None
    )
    run = background_review_module.prepare_background_review_run(
        agent, session_id=agent.session_id, profile_key=profile_key
    )
    assert run is not None and run.begin_request(object())

    def cancellations():
        return [
            record
            for record in caplog.records
            if expected_reason in record.getMessage()
        ]

    try:
        with caplog.at_level("INFO"):
            canceller(agent, session_id=agent.session_id, profile_key=profile_key)
            assert len(cancellations()) == 1, "cancellation must log exactly once"
            canceller(agent, session_id=agent.session_id, profile_key=profile_key)
            assert len(cancellations()) == 1, "a repeated fence must not log again"
    finally:
        background_review_module.finish_background_review_run(agent, run)

    assert run.cancel_requested.is_set()
    line = cancellations()[0]
    assert line.levelname == "INFO"
    text = line.getMessage()
    assert review_admission.owner_tag(profile_key, agent.session_id) in text
    assert agent.session_id not in text


def test_explicit_refine_remains_exempt_from_late_followup_gate(
    review_forks, monkeypatch
):
    _patch_config(monkeypatch, _config())
    agent = _bare_agent()
    agent.followup_pending_callback = lambda: True

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
        explicit=True,
    )

    assert len(review_forks) == 1
    assert review_forks[0]["history"] is not None


def test_explicit_refine_has_no_automatic_aggregate_input_budget(
    review_forks, monkeypatch
):
    _patch_config(monkeypatch, _config(max_input_tokens=1))
    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
        explicit=True,
    )

    assert review_forks[0]["attrs"].get("_review_input_token_budget") is None


def test_queue_side_fence_does_not_cancel_explicit_refine(monkeypatch):
    agent = _bare_agent()
    monkeypatch.setattr(
        background_review_module, "_interrupt_background_review", lambda _agent: None
    )
    run = background_review_module.prepare_background_review_run(
        agent, followup_cancellable=False
    )
    assert run is not None
    assert run.begin_request(object()) is True

    background_review_module.cancel_background_review_for_pending_followup(agent)

    assert run.cancel_requested.is_set() is False
    background_review_module.finish_background_review_run(agent, run)


def test_terminal_review_cannot_be_relabeled_preempted_before_slot_cleanup():
    run = background_review_module._BackgroundReviewRun()
    assert run.begin_request(object()) is True

    assert run.mark_request_finished() is True
    assert run.cancel() is None

    assert run.cancel_requested.is_set() is False


def test_cancel_after_provider_phase_returns_does_not_requeue_completed_review(
    review_forks, monkeypatch
):
    """A cancel landing after the provider phase returned, while the lease release, usage
    attribution and fork unregister are still pending, must not relabel a COMPLETED review as
    preempted: its memory/skill writes already happened, so a requeue would run it all again."""
    _patch_config(monkeypatch, _config())
    agent = _bare_agent()
    runs, enqueued, interrupts = [], [], []
    original_prepare = background_review_module.prepare_background_review_run

    def _capture_prepare(*args, **kwargs):
        runs.append(original_prepare(*args, **kwargs))
        return runs[-1]

    monkeypatch.setattr(
        background_review_module, "prepare_background_review_run", _capture_prepare
    )
    monkeypatch.setattr(
        agent, "_requeue_deferred_review", lambda kwargs: enqueued.append(kwargs)
    )
    monkeypatch.setattr(
        background_review_module,
        "_interrupt_background_review",
        lambda review_agent: interrupts.append(review_agent),
    )
    # Land the live-turn cancel inside the post-return window of the fork's finally.
    monkeypatch.setattr(
        background_review_module,
        "_record_review_usage_to_parent",
        lambda *_args, **_kwargs: (
            background_review_module.cancel_background_review_for_live_turn(
                agent, wait=False
            )
        ),
    )

    AIAgent._spawn_background_review_now(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
        task_cfg={"defer": "auto"},
        _idle_queue_origin=True,
        _review_session_id=agent.session_id,
    )

    assert review_forks and review_forks[0]["history"] is not None
    assert runs[0].request_done.is_set()
    assert runs[0].cancel_requested.is_set() is False
    assert interrupts == []
    assert enqueued == []


def test_foreground_wait_has_no_timeout_escape_into_provider_work():
    entered = threading.Event()
    release = threading.Event()
    observed_timeouts = []

    class ControlledCompletion:
        def wait(self, timeout=None):
            observed_timeouts.append(timeout)
            entered.set()
            release.wait(timeout=10.0)
            return True

    run = background_review_module._BackgroundReviewRun()
    run.request_done = ControlledCompletion()
    waiter = threading.Thread(
        target=background_review_module.wait_for_background_review_cancellation,
        args=(run,),
    )
    waiter.start()
    assert entered.wait(timeout=10.0)
    release.set()
    waiter.join(timeout=10.0)

    assert not waiter.is_alive()
    assert observed_timeouts == [None]


def test_deferred_dispatch_does_not_exclude_a_different_current_turn(
    review_forks,
    monkeypatch,
    caplog,
):
    """Dispatch excludes only its spawning turn, not a newer foreground turn."""
    from agent import review_idle_queue

    _patch_config(monkeypatch, _config())
    agent = _bare_agent()
    queued: dict = {}
    monkeypatch.setattr(run_agent_module, "_review_should_defer", lambda *_args: True)
    monkeypatch.setattr(
        review_idle_queue.QUEUE,
        "enqueue",
        lambda _agent, _key, kwargs: queued.update(kwargs),
    )

    spawning_token = review_admission.note_turn_started(agent.session_id)
    agent._active_turn_token = spawning_token
    try:
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
        )
    finally:
        review_admission.note_turn_finished(agent.session_id, spawning_token)

    current_token = review_admission.note_turn_started(agent.session_id)
    agent._active_turn_token = current_token
    try:
        with caplog.at_level("INFO"):
            AIAgent._spawn_background_review_now(agent, **queued)
    finally:
        review_admission.note_turn_finished(agent.session_id, current_token)

    assert review_forks == [], "deferred review crossed a newer foreground turn"
    assert review_admission.REASON_LIVE_TURN in caplog.text


def test_review_thread_keeps_the_session_identity_captured_at_spawn(
    review_forks, monkeypatch
):
    targets = []

    class CapturedThread:
        def __init__(self, *, target, daemon=None, name=None):
            targets.append(target)

        def start(self):
            pass

    _patch_config(monkeypatch, _config(defer="never"))
    monkeypatch.setattr(
        run_agent_module,
        "threading",
        types.SimpleNamespace(Thread=CapturedThread),
    )
    agent = _bare_agent("session-one")
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "session one"}],
        review_memory=True,
    )

    agent.session_id = "session-two"
    targets[0]()

    assert review_forks[0]["init"]["parent_session_id"] == "session-one"
    assert review_forks[0]["attrs"]["session_id"] == "session-one"


def test_turn_registration_is_released_when_review_cancellation_raises(monkeypatch):
    """An exceptional cancellation path must not block reviews on the session forever."""
    session_id = "cancel-error-session"
    agent = types.SimpleNamespace(session_id=session_id, platform="whatsapp")

    def fail_cancel(_agent, **_kwargs):
        raise RuntimeError("cancel failed")

    monkeypatch.setattr(
        background_review_module,
        "cancel_background_review_for_live_turn",
        fail_cancel,
    )

    with pytest.raises(RuntimeError, match="cancel failed"):
        TurnFacadeMixin.run_conversation(cast(TurnFacadeMixin, agent), "hello")

    assert review_admission.other_live_turn(session_id, None) is False


def test_active_turn_token_assignment_exception_leaves_no_live_token():
    """Registration is balanced even when the host rejects the token attribute."""

    class RejectActiveTurnToken:
        session_id = "assignment-error-session"
        platform = "whatsapp"

        def __setattr__(self, name, value):
            if name == "_active_turn_token":
                raise RuntimeError("assignment failed")
            object.__setattr__(self, name, value)

    agent = RejectActiveTurnToken()
    with pytest.raises(RuntimeError, match="assignment failed"):
        TurnFacadeMixin.run_conversation(cast(TurnFacadeMixin, agent), "hello")

    assert review_admission.other_live_turn(agent.session_id, None) is False


def test_review_admitted_when_the_session_is_free(review_forks, monkeypatch):
    """The exclusion is scoped to the session: another session's turn must not starve learning."""
    _patch_config(monkeypatch, _config())
    agent = _bare_agent()
    token = review_admission.note_turn_started("some-other-session")
    try:
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
        )
    finally:
        review_admission.note_turn_finished("some-other-session", token)

    assert len(review_forks) == 1
    assert review_forks[0]["history"] is not None


def test_live_turn_registry_isolated_by_canonical_profile_and_session():
    session_id = "shared-session-id"
    token = review_admission.note_turn_started(session_id, "/profiles/alpha")
    try:
        assert (
            review_admission.other_live_turn(session_id, None, "/profiles/alpha")
            is True
        )
        assert (
            review_admission.other_live_turn(session_id, None, "/profiles/beta")
            is False
        )
    finally:
        review_admission.note_turn_finished(session_id, token, "/profiles/beta")

    # Cleanup uses the token's registration key, not the caller's current/wrong profile, and
    # remains idempotent.
    review_admission.note_turn_finished(session_id, token, "/profiles/alpha")
    assert (
        review_admission.other_live_turn(session_id, None, "/profiles/alpha") is False
    )


def test_review_check_keeps_spawning_profile_across_deferred_dispatch(
    review_forks, monkeypatch
):
    _patch_config(monkeypatch, _config())
    agent = _bare_agent("shared-session-id")
    alpha_token = review_admission.note_turn_started(
        agent.session_id, "/profiles/alpha"
    )
    try:
        # Simulate the idle dispatcher running outside the originating profile context. The fixed
        # profile key must keep beta independent from alpha even though the session IDs match.
        AIAgent._spawn_background_review_now(
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
            task_cfg={},
            _review_profile_key="/profiles/beta",
            _idle_queue_origin=True,
        )
    finally:
        review_admission.note_turn_finished(
            agent.session_id, alpha_token, "/profiles/alpha"
        )

    assert len(review_forks) == 1
    assert review_forks[0]["history"] is not None


# ---------------------------------------------------------------------------
# 3 + 4 — replay is bounded for oversized sessions, verbatim for normal ones
# ---------------------------------------------------------------------------


def test_replay_token_budget_defaults_when_missing():
    assert (
        review_admission.replay_token_budget({})
        == review_admission.MAX_REPLAY_TOKENS_DEFAULT
    )


@pytest.mark.parametrize(
    "raw",
    [0, -1, review_admission.MAX_REPLAY_TOKENS_DEFAULT + 1],
)
def test_operator_cannot_disable_or_raise_automatic_replay_bound(raw):
    assert (
        review_admission.replay_token_budget({"max_replay_tokens": raw})
        == review_admission.MAX_REPLAY_TOKENS_DEFAULT
    )


@pytest.mark.parametrize("raw", [None, True, False, "not-a-number"])
def test_invalid_replay_token_budget_warns_and_uses_default(raw, caplog):
    with caplog.at_level("WARNING", logger=review_admission.__name__):
        budget = review_admission.replay_token_budget({"max_replay_tokens": raw})

    assert budget == review_admission.MAX_REPLAY_TOKENS_DEFAULT
    assert "Invalid auxiliary.background_review.max_replay_tokens" in caplog.text


@pytest.mark.parametrize(
    "carrier_content",
    [
        [{"type": "tool_result", "content": "result"}],
        [{"type": "text", "tool_use_id": "toolu_sensitive", "text": "result"}],
    ],
    ids=["tool-result-type", "tool-use-id"],
)
def test_anthropic_tool_result_user_message_cannot_anchor_replay(carrier_content):
    carrier = {"role": "user", "content": carrier_content}
    latest_turn = [
        {"role": "user", "content": "latest question"},
        {"role": "assistant", "content": "latest answer"},
    ]
    snapshot = [
        {"role": "user", "content": "old question " + ("x" * 8_000)},
        {"role": "assistant", "content": "old tool request"},
        carrier,
        {"role": "assistant", "content": "tool result acknowledged"},
        *latest_turn,
    ]
    budget = estimate_messages_tokens_rough(snapshot[2:])

    replay, reason = review_admission.bounded_replay_history(snapshot, budget)

    assert replay == latest_turn
    assert reason == review_admission.REASON_OVERSIZED


def test_openai_tool_call_and_result_remain_paired_in_bounded_replay():
    complete_turn = [
        {"role": "user", "content": "run the tool"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        {"role": "assistant", "content": "done"},
    ]
    snapshot = [
        {"role": "user", "content": "old question " + ("x" * 8_000)},
        {"role": "assistant", "content": "old answer"},
        *complete_turn,
    ]
    budget = estimate_messages_tokens_rough(complete_turn)

    replay, reason = review_admission.bounded_replay_history(snapshot, budget)

    assert replay == complete_turn
    assert replay[1]["tool_calls"][0]["id"] == replay[2]["tool_call_id"]
    assert reason == review_admission.REASON_OVERSIZED


def test_bounded_replay_estimates_each_message_once(monkeypatch):
    from agent import model_metadata

    snapshot = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer"},
    ]
    estimated = []

    def estimate_one(messages):
        assert len(messages) == 1
        estimated.append(messages[0])
        return 10

    monkeypatch.setattr(model_metadata, "estimate_messages_tokens_rough", estimate_one)

    replay, reason = review_admission.bounded_replay_history(snapshot, budget=25)

    assert estimated == snapshot
    assert replay == snapshot[2:]
    assert reason == review_admission.REASON_OVERSIZED


def test_session_tag_is_deterministic_and_does_not_disclose_session_id():
    session_id = "customer@example.com:private-conversation-12345"
    expected = hashlib.sha256(session_id.encode()).hexdigest()[:8]

    assert review_admission.session_tag(session_id) == expected
    assert review_admission.session_tag(session_id) == expected
    assert session_id not in expected


def test_owner_tag_distinguishes_profiles_without_disclosing_identity():
    session_id = "customer@example.com:private-conversation-12345"

    alpha = review_admission.owner_tag("/profiles/alpha", session_id)
    beta = review_admission.owner_tag("/profiles/beta", session_id)

    assert alpha != beta
    assert alpha == review_admission.owner_tag("/profiles/alpha", session_id)
    assert session_id not in alpha
    assert "/profiles/alpha" not in alpha


def test_oversized_snapshot_is_replayed_bounded(review_forks, monkeypatch, caplog):
    """An oversized transcript must never be replayed verbatim into the fork."""
    _patch_config(monkeypatch, _config(max_replay_tokens=2_000))
    snapshot = _snapshot(pairs=60, filler_chars=800)
    assert estimate_messages_tokens_rough(snapshot) > 2_000

    agent = _bare_agent()
    with caplog.at_level("INFO"):
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=snapshot,
            review_memory=True,
        )

    assert len(review_forks) == 1
    history = review_forks[0]["history"]
    assert history is not None
    assert len(history) < len(snapshot), (
        "full transcript replayed into an oversized review"
    )
    assert estimate_messages_tokens_rough(history) <= 2_000
    _assert_plain_user_anchor(history)
    assert review_admission.REASON_OVERSIZED in caplog.text


def test_zero_max_replay_tokens_still_bounds_automatic_replay(
    review_forks, monkeypatch, caplog
):
    """``max_replay_tokens: 0`` cannot switch the automatic replay bound off: on the composed
    spawn path the ceiling still applies and the oversized snapshot is replayed bounded."""
    monkeypatch.setattr(review_admission, "MAX_REPLAY_TOKENS_DEFAULT", 2_000)
    _patch_config(monkeypatch, _config(max_replay_tokens=0))
    snapshot = _snapshot(pairs=60, filler_chars=800)
    assert estimate_messages_tokens_rough(snapshot) > 2_000

    agent = _bare_agent()
    with caplog.at_level("INFO"):
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=snapshot,
            review_memory=True,
        )

    assert len(review_forks) == 1
    history = review_forks[0]["history"]
    assert history is not None
    assert len(history) < len(snapshot), (
        "max_replay_tokens=0 replayed the full transcript into an automatic review"
    )
    assert estimate_messages_tokens_rough(history) <= 2_000
    _assert_plain_user_anchor(history)
    assert review_admission.REASON_OVERSIZED in caplog.text


def test_bounded_snapshot_still_replays_the_full_transcript(review_forks, monkeypatch):
    """Normal sessions keep the verbatim (warm-cache) replay — learning is not degraded."""
    _patch_config(monkeypatch, _config(max_replay_tokens=2_000))
    snapshot = _snapshot(pairs=3, filler_chars=40)
    assert estimate_messages_tokens_rough(snapshot) < 2_000

    agent = _bare_agent()
    AIAgent._spawn_background_review(
        agent, messages_snapshot=snapshot, review_memory=True
    )

    assert len(review_forks) == 1
    assert review_forks[0]["history"] == snapshot


def test_single_turn_larger_than_replay_budget_skips_automatic_review(
    review_forks,
    monkeypatch,
    caplog,
):
    """Do not pay for a review when no complete user-led suffix can fit the replay budget."""
    _patch_config(monkeypatch, _config(max_replay_tokens=100))
    snapshot = [{"role": "user", "content": "x" * 10_000}]

    agent = _bare_agent()
    with caplog.at_level("INFO"):
        AIAgent._spawn_background_review(
            agent,
            messages_snapshot=snapshot,
            review_memory=True,
        )

    assert review_forks == []
    assert review_admission.REASON_OVERSIZED in caplog.text


# ---------------------------------------------------------------------------
# 5 — persistence isolation survives both paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pairs,filler", [(3, 40), (60, 800)])
def test_review_fork_stays_detached_from_the_canonical_session(
    review_forks, monkeypatch, pairs, filler
):
    """Bounded or oversized, the fork shares the session id but never writes that session."""
    _patch_config(monkeypatch, _config(max_replay_tokens=2_000))
    agent = _bare_agent()

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=_snapshot(pairs=pairs, filler_chars=filler),
        review_memory=True,
    )

    assert len(review_forks) == 1
    attrs = review_forks[0]["attrs"]
    assert attrs["session_id"] == agent.session_id  # prefix-cache parity
    assert attrs["_persist_disabled"] is True
    assert attrs["_session_db"] is None
    assert attrs["_end_session_on_close"] is False

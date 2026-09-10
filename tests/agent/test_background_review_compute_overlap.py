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
        review_admission._turn_keys.clear()
        review_admission._tokens = itertools.count(1)
    yield
    with review_admission._lock:
        review_admission._live_turns.clear()
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


def test_raising_followup_probe_fails_open_and_allows_automatic_review(
    review_forks, monkeypatch
):
    """An unhealthy host probe must not starve automatic learning forever."""
    _patch_config(monkeypatch, _config())
    agent = _bare_agent()

    def raise_from_probe():
        raise RuntimeError("probe failed")

    agent.followup_pending_callback = raise_from_probe
    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    assert len(review_forks) == 1


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

    Every other skip/defer decision logs ``reason slug + hashed session tag``; a refusal at
    ``begin_request`` used to log nothing at all, so the cheapest signal that self-improvement is
    being starved was invisible. One INFO record, no message bodies, no raw session id.
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
    assert review_admission.session_tag(session_id) in refusal_text
    assert session_id not in refusal_text
    assert "private message body" not in refusal_text


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


@pytest.mark.parametrize("raw", [0, -1])
def test_nonpositive_replay_token_budget_is_unlimited(raw):
    assert review_admission.replay_token_budget({"max_replay_tokens": raw}) is None


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

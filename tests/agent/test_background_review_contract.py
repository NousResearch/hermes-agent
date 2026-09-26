"""P2.1 background-review host-contract tests.

These pin the *observable contract* the hook surface promises, independent of
what any plugin does with it:

* ``HOOK_CONTRACT_VERSION`` is importable and equals 2.
* The three ``background_review_*`` hooks are registered in ``VALID_HOOKS``.
* ``background_review_started`` can append a ``prompt_suffix``.
* ``background_review_finished`` fires **exactly once** per fork — on the
  success path, the exception path, and both #84423 cancellation windows —
  carrying the fork's provenance.

They drive the real ``_run_review_in_thread`` synchronously via the same
``ImmediateThread`` + ``FakeReviewAgent`` harness as ``test_background_review``.
"""

from __future__ import annotations

import hermes_cli.plugins as plugins_module
import run_agent as run_agent_module
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.model = "fake-model"
    agent.platform = "telegram"
    agent.provider = "openai"
    agent.base_url = ""
    agent.api_key = ""
    agent.api_mode = ""
    agent.session_id = "test-session"
    agent._current_turn_id = "test-session:task:abcd1234"
    agent._parent_session_id = ""
    agent._credential_pool = None
    agent._memory_store = object()
    agent._memory_enabled = True
    agent._user_profile_enabled = False
    agent._cached_system_prompt = "test-cached-system-prompt"
    import datetime as _dt

    agent.session_start = _dt.datetime(2026, 1, 1, 12, 0, 0)
    agent._MEMORY_REVIEW_PROMPT = "review memory"
    agent._SKILL_REVIEW_PROMPT = "review skills"
    agent._COMBINED_REVIEW_PROMPT = "review both"
    agent.background_review_callback = None
    agent.status_callback = None
    agent._safe_print = lambda *_a, **_k: None
    agent.memory_notifications = "on"
    return agent


class ImmediateThread:
    def __init__(self, *, target, daemon=None, name=None):
        self._target = target

    def start(self):
        self._target()


class _HookRecorder:
    """Stand-in for ``hermes_cli.plugins.invoke_hook`` that records every
    firing and can inject returns for ``background_review_started``."""

    def __init__(self, started_returns=None):
        self.calls = []  # list[(hook_name, kwargs)]
        self._started_returns = started_returns or []

    def __call__(self, hook_name, **kwargs):
        self.calls.append((hook_name, kwargs))
        if hook_name == "background_review_started":
            return list(self._started_returns)
        return []

    def names(self):
        return [name for name, _ in self.calls]

    def of(self, hook_name):
        return [kw for name, kw in self.calls if name == hook_name]


def _install(monkeypatch, recorder, review_agent_cls):
    monkeypatch.setattr(run_agent_module, "AIAgent", review_agent_cls)
    monkeypatch.setattr(run_agent_module.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(plugins_module, "invoke_hook", recorder)


def test_hook_contract_version_is_two():
    from agent.background_review import HOOK_CONTRACT_VERSION

    assert HOOK_CONTRACT_VERSION == 2


def test_background_review_hooks_registered():
    from hermes_cli.plugins import VALID_HOOKS

    for name in (
        "background_review_started",
        "background_review_message",
        "background_review_finished",
    ):
        assert name in VALID_HOOKS


def test_background_review_finished_fires_once_on_success(monkeypatch):
    recorder = _HookRecorder()

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            self._session_messages = [
                {"role": "assistant", "content": "did a thing"}
            ]

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    _install(monkeypatch, recorder, FakeReviewAgent)

    AIAgent._spawn_background_review(
        _bare_agent(),
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    finished = recorder.of("background_review_finished")
    assert len(finished) == 1
    assert finished[0]["status"] == "finished"
    assert finished[0]["error"] is None
    # Provenance: the context carried into the daemon tags this fork.
    assert finished[0]["context"].execution_kind == "background_review"
    assert finished[0]["context"].session_id == "test-session"


def test_background_review_finished_fires_once_on_failure(monkeypatch):
    recorder = _HookRecorder()

    class ExplodingReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            raise RuntimeError("boom")

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    _install(monkeypatch, recorder, ExplodingReviewAgent)

    agent = _bare_agent()
    agent._emit_auxiliary_failure = lambda *_a, **_k: None

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    finished = recorder.of("background_review_finished")
    assert len(finished) == 1
    assert finished[0]["status"] == "failed"
    assert "boom" in (finished[0]["error"] or "")


def test_background_review_finished_not_re_emitted_when_post_success_output_raises(monkeypatch):
    """A crash in the post-'finished' summary/output code must NOT emit a second
    terminal event. background_review_finished is structurally exactly-once, so
    the plugin's counters stay consistent (fork_started == finished + failed).
    """
    import agent.background_review as bg

    recorder = _HookRecorder()

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            self._session_messages = [
                {"role": "assistant", "content": "did a thing"}
            ]

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    _install(monkeypatch, recorder, FakeReviewAgent)
    # Non-empty actions => the post-'finished' output block runs...
    monkeypatch.setattr(bg, "summarize_background_review_actions",
                        lambda *_a, **_k: ["created a memory"])

    agent = _bare_agent()
    # ...and _safe_print (unguarded) raises there, landing in the outer except.
    def _boom_print(*_a, **_k):
        raise RuntimeError("print exploded after finished")
    agent._safe_print = _boom_print
    agent._emit_auxiliary_failure = lambda *_a, **_k: None

    AIAgent._spawn_background_review(
        agent,
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    finished = recorder.of("background_review_finished")
    assert len(finished) == 1, "terminal event must fire exactly once"
    assert finished[0]["status"] == "finished"
    assert finished[0]["error"] is None


def test_background_review_started_prompt_suffix_appended(monkeypatch):
    recorder = _HookRecorder(started_returns=[{"prompt_suffix": "EXTRA-INSTRUCTION"}])

    seen = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            seen["user_message"] = kwargs.get("user_message", "")

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    _install(monkeypatch, recorder, FakeReviewAgent)

    AIAgent._spawn_background_review(
        _bare_agent(),
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    # The suffix returned by background_review_started is concatenated onto the
    # review prompt handed to the fork.
    assert "EXTRA-INSTRUCTION" in seen["user_message"]
    assert recorder.names().count("background_review_started") == 1


# ── #84423 cancellation windows ──────────────────────────────────────────────
#
# main gained a per-review cancel token (``_BackgroundReviewRun``) so a new live
# turn can fence out or interrupt an in-flight review. That created two exits
# the v1 contract did not describe:
#
#   A. startup fence — ``_run_review_in_thread`` returns before the body, so a
#      v1 host emitted background_review_started with NO terminal event at all.
#   B. cancelled-but-nominally-successful — ``begin_request()`` refuses
#      admission (transcript empty), or a live turn interrupts mid-flight
#      (``interrupt()`` is cooperative, so ``run_conversation`` returns
#      normally and the transcript is truncated). Both land on the success
#      path, which a v1 host reported as ``status="finished"``.
#
# Both break the counter invariant a consumer relies on:
#     started == finished + failed + cancelled


class _PassiveReviewAgent:
    """Fork stand-in that records nothing and never raises."""

    def __init__(self, **kwargs):
        self._session_messages = []

    def run_conversation(self, **kwargs):
        self._session_messages = [{"role": "assistant", "content": "did a thing"}]

    def shutdown_memory_provider(self):
        pass

    def close(self):
        pass


def _capture_run(monkeypatch):
    """Hand the test the real run token the spawn path builds."""
    import agent.background_review as bg

    box = {}
    real = bg.prepare_background_review_run

    def _capturing(agent_obj):
        run = real(agent_obj)
        box["run"] = run
        return run

    monkeypatch.setattr(bg, "prepare_background_review_run", _capturing)
    return box


def test_startup_fence_still_emits_a_terminal_event(monkeypatch):
    """Window A: cancelled before the daemon body runs.

    background_review_started has already fired on the foreground thread, so
    returning at the fence without a terminal event strands the fork's counter
    open forever.
    """
    import agent.background_review as bg

    recorder = _HookRecorder()

    def _already_cancelled(agent_obj):
        run = bg._BackgroundReviewRun()
        run.cancel()
        return run

    monkeypatch.setattr(bg, "prepare_background_review_run", _already_cancelled)
    _install(monkeypatch, recorder, _PassiveReviewAgent)

    AIAgent._spawn_background_review(
        _bare_agent(),
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    started = recorder.of("background_review_started")
    finished = recorder.of("background_review_finished")
    assert len(started) == 1
    assert len(finished) == 1, "the fence must not strand the started event"
    assert finished[0]["status"] == "cancelled"
    assert finished[0]["error"] is None
    assert finished[0]["messages"] == []


def test_refused_admission_reports_cancelled_not_finished(monkeypatch):
    """Window B, first half: a live turn cancels after the fork is constructed
    but before its first provider call, so ``begin_request()`` refuses and
    ``run_conversation`` never runs. The transcript is empty — reporting that as
    'finished' would let a consumer commit an empty extraction as the fork's
    whole output."""
    box = _capture_run(monkeypatch)
    recorder = _HookRecorder()

    class CancellingOnConstruct:
        def __init__(self, **kwargs):
            self._session_messages = []
            box["run"].cancel()

        def run_conversation(self, **kwargs):
            raise AssertionError("must not be admitted after cancel")

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    _install(monkeypatch, recorder, CancellingOnConstruct)

    AIAgent._spawn_background_review(
        _bare_agent(),
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    finished = recorder.of("background_review_finished")
    assert len(finished) == 1
    assert finished[0]["status"] == "cancelled"
    assert finished[0]["messages"] == []


def test_midflight_interrupt_reports_cancelled_not_finished(monkeypatch):
    """Window B, second half: the fork was admitted and produced part of a
    transcript, then a live turn interrupted it. ``interrupt()`` is cooperative
    — ``run_conversation`` returns normally — so control flow alone cannot tell
    this from success. Only the run token can."""
    box = _capture_run(monkeypatch)
    recorder = _HookRecorder()

    class InterruptedMidFlight:
        def __init__(self, **kwargs):
            self._session_messages = []

        def run_conversation(self, **kwargs):
            # Partial work, then a live turn supersedes this review.
            self._session_messages = [
                {"role": "assistant", "content": "half a thought"}
            ]
            box["run"].cancel()

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    _install(monkeypatch, recorder, InterruptedMidFlight)

    AIAgent._spawn_background_review(
        _bare_agent(),
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    finished = recorder.of("background_review_finished")
    assert len(finished) == 1
    assert finished[0]["status"] == "cancelled", (
        "a truncated transcript must not be reported as a finished review"
    )
    # The partial transcript is still handed over — a consumer may want it,
    # it just must not mistake it for the whole review.
    assert finished[0]["messages"] == [
        {"role": "assistant", "content": "half a thought"}
    ]


def test_uncancelled_success_still_reports_finished(monkeypatch):
    """Control: the cancelled status must not leak onto the ordinary path."""
    _capture_run(monkeypatch)
    recorder = _HookRecorder()
    _install(monkeypatch, recorder, _PassiveReviewAgent)

    AIAgent._spawn_background_review(
        _bare_agent(),
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    finished = recorder.of("background_review_finished")
    assert len(finished) == 1
    assert finished[0]["status"] == "finished"


def test_tool_call_capability_skip_still_emits_a_terminal_event(monkeypatch):
    """Window C: the worker declines to build a fork because the parent's
    provider cannot emit Hermes tool calls and the review is not routed.

    background_review_started has already fired on the foreground thread, so
    this early return must close the fork out too — as ``failed`` with the
    reason, since no review ran (it was not superseded by a live turn, so
    ``cancelled`` would misdescribe it)."""
    import agent.background_review as bg

    recorder = _HookRecorder()
    monkeypatch.setattr(bg, "_parent_can_emit_tool_calls", lambda _agent: False)
    monkeypatch.setattr(bg, "_resolve_review_runtime", lambda *_a, **_k: {"routed": False})
    _install(monkeypatch, recorder, _PassiveReviewAgent)

    AIAgent._spawn_background_review(
        _bare_agent(),
        messages_snapshot=[{"role": "user", "content": "hello"}],
        review_memory=True,
    )

    started = recorder.of("background_review_started")
    finished = recorder.of("background_review_finished")
    assert len(started) == 1
    assert len(finished) == 1, "the capability skip must not strand the started event"
    assert finished[0]["status"] == "failed"
    assert "cannot emit Hermes tool calls" in (finished[0]["error"] or "")
    assert finished[0]["messages"] == []
    assert finished[0]["context"] is started[0]["context"]

"""P2.1 background-review host-contract tests.

These pin the *observable contract* the hook surface promises, independent of
what any plugin does with it:

* ``HOOK_CONTRACT_VERSION`` is importable and equals 1.
* The three ``background_review_*`` hooks are registered in ``VALID_HOOKS``.
* ``background_review_started`` can append a ``prompt_suffix``.
* ``background_review_finished`` fires **exactly once** per fork — on both the
  success path and the exception path — carrying the fork's provenance.

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


def test_hook_contract_version_is_one():
    from agent.background_review import HOOK_CONTRACT_VERSION

    assert HOOK_CONTRACT_VERSION == 1


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

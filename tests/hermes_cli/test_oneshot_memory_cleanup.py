"""Tests for oneshot memory-provider cleanup (SIGABRT fix, #61875).

Verifies that:
1. The explicit cleanup path forwards the agent's real transcript to
   shutdown_memory_provider (mirroring cli.py:_run_cleanup) instead of
   letting run_agent.py's on_session_end see an empty list.
2. Cleanup runs exactly once even when both the explicit ``finally`` path
   and the atexit fallback fire.
3. Missing / non-list ``_session_messages`` falls back to the no-messages
   call instead of raising.
"""

from unittest.mock import patch

from hermes_cli.oneshot import (
    _make_oneshot_memory_shutdown,
    _run_agent,
    _shutdown_memory_provider_with_transcript,
)


class AgentStub:
    """Minimal stand-in for AIAgent recording memory-shutdown calls."""

    _NO_MESSAGES = object()  # sentinel: shutdown called without a transcript

    def __init__(self, messages=None):
        if messages is not None:
            self._session_messages = messages
        self.shutdown_calls = []
        self.close_calls = 0

    def run_conversation(self, prompt):
        return {"final_response": "ok"}

    def shutdown_memory_provider(self, messages=_NO_MESSAGES):
        self.shutdown_calls.append(messages)

    def close(self):
        self.close_calls += 1


def _drive_run_agent(agent):
    """Run _run_agent with all heavy dependencies stubbed out.

    Returns ``(response, result, atexit_hook)`` where ``atexit_hook`` is the
    fallback _run_agent registered (captured instead of installed, so tests
    don't leak hooks into the interpreter).
    """
    hooks = []
    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={}),
        patch("hermes_cli.tools_config._get_platform_tools", return_value=set()),
        patch("run_agent.AIAgent", return_value=agent),
        patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=None),
        patch("hermes_cli.oneshot.get_fallback_chain", return_value=None),
        patch("atexit.register", side_effect=hooks.append),
    ):
        response, result = _run_agent("hi")
    assert len(hooks) == 1, "oneshot must register exactly one atexit fallback"
    return response, result, hooks[0]


class TestOneshotForwardsTranscript:
    def test_explicit_cleanup_forwards_real_transcript(self, monkeypatch):
        monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
        transcript = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        agent = AgentStub(messages=transcript)

        response, _, _ = _drive_run_agent(agent)

        assert response == "ok"
        assert agent.shutdown_calls == [transcript], (
            "explicit cleanup must forward agent._session_messages, "
            "not call shutdown_memory_provider() bare"
        )
        assert agent.close_calls == 1

    def test_cleanup_runs_once_when_atexit_also_fires(self, monkeypatch):
        monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
        agent = AgentStub(messages=[{"role": "user", "content": "hi"}])

        _, _, atexit_hook = _drive_run_agent(agent)
        assert len(agent.shutdown_calls) == 1  # explicit finally path fired

        atexit_hook()  # simulated interpreter-exit fallback
        atexit_hook()  # atexit must also tolerate repeat invocation

        assert len(agent.shutdown_calls) == 1, (
            "atexit fallback must be a no-op after the explicit cleanup ran"
        )

    def test_atexit_fallback_forwards_transcript_when_it_fires_first(self):
        """When only the atexit path runs (finally skipped, e.g. hard-exit
        wedge), the fallback itself must forward the real transcript."""
        transcript = [{"role": "user", "content": "hi"}]
        agent = AgentStub(messages=transcript)
        hook = _make_oneshot_memory_shutdown(lambda: agent)

        hook()

        assert agent.shutdown_calls == [transcript]

    def test_missing_session_messages_falls_back_to_bare_call(self):
        agent = AgentStub(messages=None)  # attribute absent entirely
        _shutdown_memory_provider_with_transcript(agent)
        assert agent.shutdown_calls == [AgentStub._NO_MESSAGES]

    def test_non_list_session_messages_falls_back_to_bare_call(self):
        agent = AgentStub(messages=None)
        agent._session_messages = "not-a-list"
        _shutdown_memory_provider_with_transcript(agent)
        assert agent.shutdown_calls == [AgentStub._NO_MESSAGES]

    def test_none_agent_is_noop(self):
        _shutdown_memory_provider_with_transcript(None)  # must not raise

    def test_shutdown_exceptions_are_swallowed(self):
        class ExplodingAgent(AgentStub):
            def shutdown_memory_provider(self, messages=AgentStub._NO_MESSAGES):
                raise RuntimeError("boom")

        _shutdown_memory_provider_with_transcript(ExplodingAgent(messages=[]))

    def test_make_oneshot_memory_shutdown_is_idempotent(self):
        agent = AgentStub(messages=[])
        hook = _make_oneshot_memory_shutdown(lambda: agent)
        hook()
        hook()
        assert len(agent.shutdown_calls) == 1

from contextlib import nullcontext

from agent.conversation_compression import (
    _queue_context_engine_compression_notification,
)
from cli import HermesCLI


class DummyAgent:
    def __init__(self):
        self.compression_enabled = True
        self._cached_system_prompt = "FULL CACHED SYSTEM PROMPT SHOULD NOT BE NESTED"
        self.session_id = "new-session"
        self.calls = []
        self.flush_calls = []
        self.flush_error = None
        self.host_events = []
        self.boundary_calls = []
        self.context_compressor = type("ContextEngineStub", (), {})()
        self.context_compressor.on_session_start = self._record_boundary

    def _record_boundary(self, session_id, **kwargs):
        self.host_events.append("notify")
        self.boundary_calls.append((session_id, kwargs))

    def _flush_messages_to_session_db(self, messages, _session_id=None):
        self.host_events.append("persist")
        self.flush_calls.append((list(messages), _session_id))
        if self.flush_error is not None:
            raise self.flush_error

    def _compress_context(
        self,
        messages,
        system_message,
        *,
        approx_tokens=None,
        focus_topic=None,
        force=False,
        defer_context_engine_notification=False,
    ):
        self.calls.append(
            {
                "messages": messages,
                "system_message": system_message,
                "approx_tokens": approx_tokens,
                "focus_topic": focus_topic,
                "force": force,
                "defer_context_engine_notification": (
                    defer_context_engine_notification
                ),
            }
        )
        if defer_context_engine_notification:
            _queue_context_engine_compression_notification(
                self,
                new_session_id=self.session_id,
                old_session_id="old-session",
            )
        return ([{"role": "user", "content": "[CONTEXT SUMMARY]: compacted"}], "new system prompt")


def test_manual_compress_does_not_pass_cached_system_prompt(monkeypatch):
    """Manual /compress should rebuild the next prompt without nesting the old one."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.conversation_history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]
    cli.agent = DummyAgent()
    cli.session_id = "old-session"
    cli._pending_title = "old title"
    cli._busy_command = lambda _message: nullcontext()

    monkeypatch.setattr(
        "agent.manual_compression_feedback.summarize_manual_compression",
        lambda *args, **kwargs: {
            "noop": False,
            "headline": "compressed",
            "token_line": "tokens reduced",
            "note": "",
        },
    )

    cli._manual_compress("/compress database schema")

    assert len(cli.agent.calls) == 1
    call = cli.agent.calls[0]
    assert call["system_message"] is None
    assert call["system_message"] != cli.agent._cached_system_prompt
    assert call["focus_topic"] == "database schema"
    assert cli.session_id == "new-session"
    assert cli._pending_title is None
    assert len(cli.agent.flush_calls) == 1
    assert cli.agent.host_events == ["persist", "notify"]
    assert len(cli.agent.boundary_calls) == 1


def test_manual_compress_flush_failure_discards_notification(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.conversation_history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]
    cli.agent = DummyAgent()
    cli.agent.flush_error = RuntimeError("synthetic child flush failure")
    cli.session_id = "old-session"
    cli._pending_title = "old title"
    cli._busy_command = lambda _message: nullcontext()

    cli._manual_compress("/compress")

    assert len(cli.agent.flush_calls) == 1
    assert cli.agent.host_events == ["persist"]
    assert cli.agent.boundary_calls == []

from contextlib import nullcontext

from cli import HermesCLI


class DummyAgent:
    def __init__(self):
        self.compression_enabled = True
        self._cached_system_prompt = "FULL CACHED SYSTEM PROMPT SHOULD NOT BE NESTED"
        self.session_id = "new-session"
        self.calls = []

    def _compress_context(self, messages, system_message, *, approx_tokens=None, focus_topic=None):
        self.calls.append(
            {
                "messages": messages,
                "system_message": system_message,
                "approx_tokens": approx_tokens,
                "focus_topic": focus_topic,
            }
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

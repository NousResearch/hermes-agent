from contextlib import nullcontext

from cli import HermesCLI


class DummyCompressor:
    def __init__(self, tokens=151_000, context_length=236_000):
        self.last_prompt_tokens = tokens
        self.context_length = context_length


class DummyRolloverAgent:
    def __init__(self, tokens=151_000):
        self.compression_enabled = True
        self.context_compressor = DummyCompressor(tokens=tokens)
        self._cached_system_prompt = "SYSTEM PROMPT SHOULD BE ESTIMATED BUT NOT NESTED"
        self.tools = [{"type": "function", "function": {"name": "terminal"}}]
        self.session_id = "old-session"
        self.calls = []
        self.flushed = []

    def _compress_context(self, messages, system_message, *, approx_tokens=None, focus_topic=None, force=False):
        self.calls.append(
            {
                "messages": list(messages),
                "system_message": system_message,
                "approx_tokens": approx_tokens,
                "focus_topic": focus_topic,
                "force": force,
            }
        )
        self.session_id = "child-session"
        return ([{"role": "user", "content": "[CONTEXT SUMMARY]: keep working from the handoff"}], "new prompt")

    def _flush_messages_to_session_db(self, messages, offset):
        self.flushed.append((list(messages), offset))


def make_cli(agent):
    cli = HermesCLI.__new__(HermesCLI)
    cli.agent = agent
    cli.session_id = "old-session"
    cli.conversation_history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "fourth"},
    ]
    cli._pending_title = "old title"
    cli._busy_command = lambda _message: nullcontext()
    cli._transfer_session_yolo = lambda old, new: None
    return cli


def test_cli_self_rollover_compresses_to_child_session_at_default_150k(monkeypatch):
    """Default CLI sessions should automatically roll over once prompt pressure reaches 150k."""
    agent = DummyRolloverAgent(tokens=151_000)
    cli = make_cli(agent)

    monkeypatch.setattr(
        "agent.model_metadata.estimate_request_tokens_rough",
        lambda *args, **kwargs: 151_000,
    )

    rolled = cli._maybe_self_rollover_after_turn()

    assert rolled is True
    assert len(agent.calls) == 1
    call = agent.calls[0]
    assert call["system_message"] is None
    assert call["approx_tokens"] == 151_000
    assert call["force"] is True
    assert "fresh context transfer" in call["focus_topic"]
    assert cli.session_id == "child-session"
    assert cli._pending_title is None
    assert cli.conversation_history == [{"role": "user", "content": "[CONTEXT SUMMARY]: keep working from the handoff"}]
    assert agent.flushed == [(cli.conversation_history, None)]


def test_cli_self_rollover_stays_idle_below_threshold(monkeypatch):
    """Below 150k, the CLI must not churn sessions or compress early."""
    agent = DummyRolloverAgent(tokens=149_999)
    cli = make_cli(agent)

    monkeypatch.setattr(
        "agent.model_metadata.estimate_request_tokens_rough",
        lambda *args, **kwargs: 149_999,
    )

    rolled = cli._maybe_self_rollover_after_turn()

    assert rolled is False
    assert agent.calls == []
    assert cli.session_id == "old-session"
    assert cli._pending_title == "old title"


def test_cli_self_rollover_can_be_disabled_by_config(monkeypatch):
    """Users can opt out if they prefer the existing in-session compression behavior only."""
    import cli as cli_module

    agent = DummyRolloverAgent(tokens=200_000)
    cli = make_cli(agent)
    monkeypatch.setitem(cli_module.CLI_CONFIG, "cli_self_rollover", {"enabled": False, "threshold_tokens": 150_000})

    rolled = cli._maybe_self_rollover_after_turn()

    assert rolled is False
    assert agent.calls == []

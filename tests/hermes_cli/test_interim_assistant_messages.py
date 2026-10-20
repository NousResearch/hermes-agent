"""Classic-CLI coverage for mid-turn assistant commentary."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

import cli as cli_mod
from hermes_cli.cli_single_query import _configure_quiet_agent


def _bare_cli():
    """Build only the display state used by the callback."""
    cli = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    cli.interim_assistant_messages = True
    cli._stream_started = False
    cli._stream_box_opened = False
    cli._stream_box_live = False
    cli._stream_buf = ""
    cli._stream_table_buf = []
    cli._in_stream_table = False
    cli._reasoning_box_opened = False
    cli._held_status_lines = []
    cli.show_reasoning = False
    cli.final_response_markdown = "strip"
    cli.conversation_history = [{"role": "user", "content": "hello"}]
    cli.agent = SimpleNamespace(
        _session_messages=[{"role": "assistant", "content": "persisted by the agent"}]
    )
    setattr(cli, "_invalidate", lambda *_args, **_kwargs: None)
    return cli


def test_interim_callback_renders_multiline_scrollback_without_persisting(monkeypatch):
    cli = _bare_cli()
    rendered = []
    conversation_before = copy.deepcopy(cli.conversation_history)
    session_before = copy.deepcopy(getattr(cli.agent, "_session_messages"))
    monkeypatch.setattr(cli_mod, "_cprint", rendered.append)
    monkeypatch.setattr("hermes_cli.cli_stream_mixin._terminal_columns", lambda: 80)

    cli._on_interim_assistant("  我先检查配置。\nThen I will run the tests.  ")

    visible = "\n".join(rendered)
    assert "╭" in rendered[0] and "◆" in rendered[0]
    assert "我先检查配置。" in visible
    assert "Then I will run the tests." in visible
    assert "╰" in rendered[-1]
    assert cli.conversation_history == conversation_before
    assert getattr(cli.agent, "_session_messages") == session_before
    assert cli._stream_started is False
    assert cli._stream_box_opened is False


@pytest.mark.parametrize("text", ["", "   ", None, 123])
def test_interim_callback_ignores_non_visible_text(monkeypatch, text):
    cli = _bare_cli()
    rendered = []
    monkeypatch.setattr(cli_mod, "_cprint", rendered.append)

    cli._on_interim_assistant(text)

    assert rendered == []


def test_interim_callback_respects_config_gate(monkeypatch):
    cli = _bare_cli()
    cli.interim_assistant_messages = False
    rendered = []
    monkeypatch.setattr(cli_mod, "_cprint", rendered.append)

    cli._on_interim_assistant("hidden")

    assert rendered == []


def test_already_streamed_interim_remains_single_writer(monkeypatch):
    cli = _bare_cli()
    cli._stream_started = True
    cli._stream_box_opened = True
    events = []
    cli._flush_stream = lambda: events.append("flush")
    cli._reset_stream_state = lambda: events.append("reset")
    monkeypatch.setattr(cli_mod, "_cprint", lambda text: events.append(("print", text)))

    cli._on_interim_assistant("the stream already showed this", already_streamed=True)

    assert events == []
    assert cli._stream_started is True
    assert cli._stream_box_opened is True


def test_unstreamed_interim_closes_and_resets_live_stream_chrome(monkeypatch):
    cli = _bare_cli()
    cli._stream_started = True
    cli._stream_box_opened = True
    events = []
    cli._flush_stream = lambda: events.append("flush")
    cli._reset_stream_state = lambda: events.append("reset")
    setattr(cli, "_invalidate", lambda *_args, **_kwargs: events.append("invalidate"))
    monkeypatch.setattr(cli_mod, "_cprint", lambda text: events.append(("print", text)))

    cli._on_interim_assistant("new commentary")

    assert events[:2] == ["flush", "reset"]
    assert any(
        isinstance(event, tuple) and event[0] == "print" and "new commentary" in event[1]
        for event in events
    )
    assert events[-1] == "invalidate"


def test_unstreamed_interim_closes_reasoning_chrome_first(monkeypatch):
    cli = _bare_cli()
    cli._reasoning_box_opened = True
    events = []
    cli._close_reasoning_box = lambda: events.append("close-reasoning")
    monkeypatch.setattr(cli_mod, "_cprint", lambda text: events.append(("print", text)))

    cli._on_interim_assistant("visible after reasoning")

    assert events[0] == "close-reasoning"
    assert any(isinstance(event, tuple) and "visible after reasoning" in event[1] for event in events)


def _init_with_fake_agent(monkeypatch, *, enabled: bool, streaming: bool = False):
    """Exercise the real constructor call while replacing its external edges."""
    import run_agent
    from hermes_cli import mcp_startup

    cli = cli_mod.HermesCLI(compact=True)
    cli.interim_assistant_messages = enabled
    cli.streaming_enabled = streaming
    setattr(cli, "_session_db", object())
    cli._resumed = False
    cli.conversation_history = []
    cli.finalize_preloaded_skills = lambda: None
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.__dict__.update(kwargs)

    monkeypatch.setattr(cli_mod, "_prepare_deferred_agent_startup", lambda: None)
    monkeypatch.setattr(mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **_kwargs: None)
    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    monkeypatch.setattr(cli_mod, "_active_agent_ref", None)

    runtime = {
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "provider": "custom",
        "requested_provider": "custom",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }
    assert cli._init_agent(runtime_override=runtime) is True
    return cli, captured


def test_init_agent_wires_callback_even_when_token_streaming_is_off(monkeypatch):
    cli, captured = _init_with_fake_agent(monkeypatch, enabled=True, streaming=False)

    assert captured["stream_delta_callback"] is None
    assert captured["interim_assistant_callback"] == cli._on_interim_assistant


def test_init_agent_omits_callback_when_interim_messages_are_disabled(monkeypatch):
    _cli, captured = _init_with_fake_agent(monkeypatch, enabled=False)

    assert captured["interim_assistant_callback"] is None


def test_display_config_initializes_interim_message_gate(monkeypatch):
    config = copy.deepcopy(cli_mod.CLI_CONFIG)
    config["display"]["interim_assistant_messages"] = False
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", config)

    cli = cli_mod.HermesCLI(compact=True)

    assert cli.interim_assistant_messages is False


def test_quiet_single_query_clears_interim_callback_with_all_presenters():
    sentinel = object()
    agent = SimpleNamespace(
        quiet_mode=False,
        suppress_status_output=False,
        stream_delta_callback=sentinel,
        interim_assistant_callback=sentinel,
        tool_gen_callback=sentinel,
        reasoning_callback=sentinel,
        tool_progress_callback=sentinel,
        tool_start_callback=sentinel,
        tool_complete_callback=sentinel,
        tool_progress_mode="all",
    )

    _configure_quiet_agent(agent)

    assert agent.quiet_mode is True
    assert agent.suppress_status_output is True
    assert agent.stream_delta_callback is None
    assert agent.interim_assistant_callback is None
    assert agent.tool_gen_callback is None
    assert agent.reasoning_callback is None
    assert agent.tool_progress_callback is None
    assert agent.tool_start_callback is None
    assert agent.tool_complete_callback is None
    assert agent.tool_progress_mode == "off"


def test_oneshot_clears_interim_callback_before_turn_and_on_error(monkeypatch):
    import hermes_cli.oneshot as oneshot
    import run_agent

    seen = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.interim_assistant_callback = object()
            self._session_messages = []
            self.session_id = "oneshot-test"

        def run_conversation(self, *_args, **_kwargs):
            seen["callback_during_turn"] = self.interim_assistant_callback
            raise RuntimeError("turn failed")

        def shutdown_memory_provider(self, *_args):
            seen["memory_closed"] = True

        def close(self):
            seen["agent_closed"] = True

    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "test-model", "provider": "custom"}},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_with_fallback",
        lambda *_args, **_kwargs: (
            {
                "api_key": "test-key",
                "base_url": "https://example.invalid/v1",
                "provider": "custom",
                "requested_provider": "custom",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
            None,
        ),
    )
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr("tools.process_registry.process_registry.wait_for_pending_completions", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)

    with pytest.raises(RuntimeError, match="turn failed"):
        oneshot._run_agent("hello", model="test-model", provider="custom")

    assert seen["callback_during_turn"] is None
    assert seen["memory_closed"] is True
    assert seen["agent_closed"] is True

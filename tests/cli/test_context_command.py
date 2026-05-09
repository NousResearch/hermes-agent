"""Tests for the CLI /context command."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI
from hermes_cli.commands import resolve_command


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "openai/gpt-5.5"
    cli_obj.provider = "openai"
    cli_obj.base_url = ""
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.session_start = datetime(2026, 5, 7, 13, 0)
    cli_obj.conversation_history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    return cli_obj


def _attach_agent(cli_obj):
    cli_obj.agent = SimpleNamespace(
        model=cli_obj.model,
        provider=cli_obj.provider,
        base_url=cli_obj.base_url,
        tools=[
            {"type": "function", "function": {"name": "search", "description": "web search"}},
            {"type": "function", "function": {"name": "read_file", "description": "read files"}},
        ],
        _cached_system_prompt="system prompt text",
        system_prompt="fallback system prompt",
        enabled_toolsets=["web", "file"],
        context_compressor=SimpleNamespace(
            last_prompt_tokens=1234,
            context_length=200000,
            compression_count=2,
            threshold_percent=0.5,
        ),
    )
    return cli_obj


def test_context_command_is_available_in_cli_registry():
    cmd = resolve_command("context")
    assert cmd is not None
    assert cmd.gateway_only is False
    assert cmd.cli_only is True
    assert cmd.category == "Info"


def test_process_command_context_dispatches_to_reporter():
    cli_obj = _make_cli()

    with patch.object(cli_obj, "_show_context_report", create=True) as mock_context:
        assert cli_obj.process_command("/context") is True

    mock_context.assert_called_once_with()


def test_show_context_report_breaks_down_request_buckets(capsys):
    cli_obj = _attach_agent(_make_cli())

    with patch("agent.model_metadata.estimate_messages_tokens_rough", return_value=50), \
         patch("agent.model_metadata.estimate_request_tokens_rough", return_value=200):
        cli_obj._show_context_report()

    output = capsys.readouterr().out
    assert "Context Composition" in output
    assert "Provider:" in output
    assert "system_prompt:" in output
    assert "conversation_messages:" in output
    assert "tool_schemas:" in output
    assert "Estimated request:" in output
    assert "Context window:" in output
    assert "Cache:" in output
    assert "Loaded tools:" in output
    assert "2" in output
    assert "web, file" in output


def test_show_context_report_handles_missing_agent(capsys):
    cli_obj = _make_cli()

    cli_obj._show_context_report()

    output = capsys.readouterr().out
    assert "No active agent" in output

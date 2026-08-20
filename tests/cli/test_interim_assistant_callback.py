"""Interactive CLI projection for mid-turn assistant narration."""

from __future__ import annotations

import inspect
import re

import pytest


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture
def cli_stub(monkeypatch):
    import cli as cli_mod
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.final_response_markdown = "raw"
    cli.show_timestamps = False
    cli._interactive_turn = True
    cli._spinner_text = ""
    cli._invalidate = lambda *args, **kwargs: None
    cli._reset_stream_state()

    emitted: list[str] = []
    monkeypatch.setattr(cli_mod, "_cprint", emitted.append)
    monkeypatch.setattr(cli_mod, "_terminal_width_for_streaming", lambda: 74)
    return cli, emitted


def test_interactive_cli_renders_unstreamed_interim_message(cli_stub):
    cli, emitted = cli_stub

    cli._on_interim_assistant("Current action: search BRC-6804.")

    plain = _strip_ansi("\n".join(emitted))
    assert plain.count("Current action: search BRC-6804.") == 1
    assert "Hermes" in plain
    assert cli._stream_box_opened is False
    assert cli._stream_buf == ""


def test_interactive_cli_closes_already_streamed_segment_without_duplicate(cli_stub):
    cli, emitted = cli_stub
    cli._stream_delta("Current action: open the first result.")

    cli._on_interim_assistant(
        "Current action: open the first result.",
        already_streamed=True,
    )

    plain = _strip_ansi("\n".join(emitted))
    assert plain.count("Current action: open the first result.") == 1
    assert cli._stream_box_opened is False
    assert cli._stream_buf == ""


def test_noninteractive_cli_keeps_interim_message_off_stdout(cli_stub):
    cli, emitted = cli_stub
    cli._interactive_turn = False

    cli._on_interim_assistant("This must not precede the final answer.")

    assert emitted == []


def test_interactive_agent_setup_wires_interim_callback():
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    source = inspect.getsource(CLIAgentSetupMixin._init_agent)
    assert "interim_assistant_callback=(" in source
    assert "self._on_interim_assistant" in source
    assert 'getattr(self, "interim_assistant_messages_enabled", True)' in source

"""display.response_box: boxed (default) vs plain (no TUI borders).

Full-width box-drawing rules wrap badly in multiplexers (herdr, tmux splits)
whose pane width differs from the client's. Plain keeps the skinned response
color and a short label, without the borders.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


@pytest.fixture
def cli_stub(monkeypatch):
    from cli import HermesCLI
    import cli as climod

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.final_response_markdown = "raw"
    cli.show_timestamps = False
    cli.response_box = "boxed"
    cli._reset_stream_state()
    cli._scrollback_box_width = lambda width=None: 40

    emitted = []
    monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(s))
    monkeypatch.setattr(climod, "_terminal_width_for_streaming", lambda: 40)
    return cli, emitted


def test_default_config_is_boxed():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["display"]["response_box"] == "boxed"


def test_boxed_stream_draws_full_width_borders(cli_stub):
    cli, emitted = cli_stub
    cli.response_box = "boxed"
    cli._stream_delta("Hello from the box.\n")
    cli._flush_stream()
    plain = [_strip_ansi(e).strip() for e in emitted]
    assert any(line.startswith("╭─") and line.endswith("╮") for line in plain)
    assert any(line.startswith("╰") and line.endswith("╯") for line in plain)
    assert any("Hello from the box." in line for line in plain)


def test_plain_stream_skips_box_borders(cli_stub):
    cli, emitted = cli_stub
    cli.response_box = "plain"
    cli._stream_delta("Hello without a box.\n")
    cli._flush_stream()
    plain = [_strip_ansi(e) for e in emitted]
    joined = "\n".join(plain)
    assert "╭" not in joined
    assert "╰" not in joined
    assert "╮" not in joined
    assert "╯" not in joined
    assert any("Hello without a box." in line for line in plain)
    # Short label is still printed so timestamps/skin branding survive.
    assert any("Hermes" in line for line in plain)


def test_plain_close_is_noop_when_box_never_opened(cli_stub):
    cli, emitted = cli_stub
    cli.response_box = "plain"
    cli._print_response_box_close(leading_newline=True)
    assert emitted == []


def test_response_box_plain_helper_defaults_boxed(cli_stub):
    cli, _ = cli_stub
    del cli.response_box
    assert cli._response_box_plain() is False
    cli.response_box = "PLAIN"
    assert cli._response_box_plain() is True
    cli.response_box = "boxed"
    assert cli._response_box_plain() is False


def test_plain_assistant_response_skips_panel(cli_stub, monkeypatch):
    cli, _ = cli_stub
    cli.response_box = "plain"
    import cli as climod

    calls = []

    class _FakeConsole:
        def print(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(climod, "ChatConsole", _FakeConsole)
    cli._print_assistant_response(
        "body text", label="⚕ Hermes", resp_color="#CD7F32", resp_text="#FFF8DC"
    )
    assert len(calls) == 2
    assert "⚕ Hermes" in str(calls[0][0][0])
    assert calls[1][0][0] == "body text"
    assert "Panel" not in type(calls[0][0][0]).__name__


def test_boxed_assistant_response_uses_panel(cli_stub, monkeypatch):
    cli, _ = cli_stub
    cli.response_box = "boxed"
    import cli as climod
    from rich.panel import Panel

    calls = []

    class _FakeConsole:
        def print(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(climod, "ChatConsole", _FakeConsole)
    cli._print_assistant_response(
        "body text", label="⚕ Hermes", resp_color="#CD7F32", resp_text="#FFF8DC"
    )
    assert len(calls) == 1
    assert isinstance(calls[0][0][0], Panel)


def test_tts_open_respects_plain(cli_stub):
    cli, emitted = cli_stub
    cli.response_box = "plain"
    cli._print_response_box_open(" ⚕ Hermes ", width=80)
    plain = [_strip_ansi(e) for e in emitted]
    assert len(plain) == 1
    assert "╭" not in plain[0]
    assert "Hermes" in plain[0]

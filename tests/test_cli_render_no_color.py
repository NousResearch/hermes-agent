"""T2.4: NO_COLOR and TERM=dumb must strip color from TUI output.

RED premise: ``ChatConsole`` hard-codes ``Console(force_terminal=True, color_system="truecolor")``,
so it emits ANSI even when the user/terminal asked for plain text (NO_COLOR spec, TERM=dumb).
"""
from hermes_cli.cli_render import ChatConsole


def test_no_color_env_strips_color_system(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    con = ChatConsole()
    assert con._inner.color_system is None


def test_no_color_env_emits_no_ansi(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    con = ChatConsole()
    con.print("[red]danger[/]")
    assert "\033[" not in con._buffer.getvalue()


def test_term_dumb_strips_color_system(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    con = ChatConsole()
    assert con._inner.color_system is None


def test_default_keeps_truecolor(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    con = ChatConsole()
    assert con._inner.color_system == "truecolor"

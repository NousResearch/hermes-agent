"""Environment the dashboard chat PTY gives ``hermes --tui`` (``_resolve_chat_argv``)."""

from pathlib import Path

from hermes_cli import web_server_chat


def test_chat_pty_enables_tui_bidi_for_xterm(monkeypatch, tmp_path: Path):
    """xterm.js has no bidi, so the dashboard PTY asks the TUI to reorder RTL text itself
    (#157750); a server-side HERMES_TUI_BIDI still wins."""
    import hermes_cli.main_tui_launch as tui_launch

    monkeypatch.setattr(tui_launch, "_make_tui_argv", lambda *_a, **_k: (["node", "fake-tui.js"], tmp_path))
    monkeypatch.setattr(tui_launch, "_apply_tui_python_env", lambda _env: None)

    monkeypatch.delenv("HERMES_TUI_BIDI", raising=False)
    _argv, _cwd, env = web_server_chat._resolve_chat_argv()
    assert env["HERMES_TUI_BIDI"] == "1"

    monkeypatch.setenv("HERMES_TUI_BIDI", "0")
    _argv, _cwd, env = web_server_chat._resolve_chat_argv()
    assert env["HERMES_TUI_BIDI"] == "0"

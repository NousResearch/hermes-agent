"""Top-level parser integration test for ``hermes terminal-setup``."""

from __future__ import annotations


def test_main_dispatches_terminal_setup(monkeypatch):
    import hermes_cli.main as main_mod

    called: dict[str, str] = {}

    monkeypatch.setattr(main_mod, "configure_windows_stdio", lambda: None, raising=False)
    monkeypatch.setattr(main_mod, "_cleanup_quarantined_exes", lambda: None)
    monkeypatch.setattr(main_mod, "_recover_from_interrupted_install", lambda: None)
    monkeypatch.setattr(main_mod, "_try_termux_fast_tui_launch", lambda: False)
    monkeypatch.setattr(main_mod, "_try_termux_fast_cli_launch", lambda: False)
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda args: None)
    monkeypatch.setattr(main_mod, "cmd_terminal_setup", lambda args: called.setdefault("command", args.command))
    monkeypatch.setattr("hermes_cli.config.get_container_exec_info", lambda: None)
    monkeypatch.setattr("sys.argv", ["hermes", "terminal-setup"])

    main_mod.main()

    assert called == {"command": "terminal-setup"}

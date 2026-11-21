"""T2.8: config knobs the TUI claims must have readers (grep showed 0 readers upstream).

Wired here: display.tui_agents_nudge (one-time hint on first delegation rows),
display.tui_auto_resume_recent (bare `hermes --tui` continues the MRU session).
display.show_cost is wired in T2.5 (status bar segment).
"""
import inspect
import queue
import types

import hermes_cli.cli_status_bar_mixin as sbm
import hermes_cli.cli_subagent_monitor as sm
import hermes_cli.main as main_mod


def test_display_knobs_have_readers():
    assert "show_cost" in inspect.getsource(sbm), "display.show_cost has no reader"
    assert "tui_agents_nudge" in inspect.getsource(sm), "display.tui_agents_nudge has no reader"
    assert "tui_auto_resume_recent" in inspect.getsource(main_mod), "display.tui_auto_resume_recent has no reader"


def test_auto_resume_recent_sets_mru_session(monkeypatch):
    import hermes_cli.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "load_config", lambda: {"display": {"tui_auto_resume_recent": True}})
    monkeypatch.setattr(main_mod, "_latest_session_id", lambda use_tui: "sess-9")
    monkeypatch.delenv("HERMES_TUI_RESUME", raising=False)
    args = types.SimpleNamespace(resume=None)
    main_mod._apply_tui_auto_resume(args, True)
    assert args.resume == "sess-9"


def test_auto_resume_explicit_resume_and_env_win(monkeypatch):
    import hermes_cli.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "load_config", lambda: {"display": {"tui_auto_resume_recent": True}})
    monkeypatch.setattr(main_mod, "_latest_session_id", lambda use_tui: "sess-9")
    args = types.SimpleNamespace(resume="mine")
    main_mod._apply_tui_auto_resume(args, True)
    assert args.resume == "mine"
    monkeypatch.setenv("HERMES_TUI_RESUME", "env-sess")
    args = types.SimpleNamespace(resume=None)
    main_mod._apply_tui_auto_resume(args, True)
    assert args.resume is None


def test_auto_resume_off_by_default(monkeypatch):
    import hermes_cli.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "load_config", lambda: {"display": {}})
    monkeypatch.delenv("HERMES_TUI_RESUME", raising=False)
    args = types.SimpleNamespace(resume=None)
    main_mod._apply_tui_auto_resume(args, True)
    assert args.resume is None


def test_agents_nudge_prints_once_when_enabled(monkeypatch):
    import cli
    prints = []
    monkeypatch.setattr(cli, "_cprint", lambda text="": prints.append(text))
    monkeypatch.setattr(cli, "CLI_CONFIG", {"display": {"tui_agents_nudge": True}})
    fake_cli = types.SimpleNamespace(agent=None, session_id="sid", _pending_input=queue.Queue())
    dock = sm.SubagentMonitor(fake_cli)
    dock._agents_nudge(True)
    dock._agents_nudge(True)
    assert len(prints) == 1, "nudge must fire exactly once"


def test_agents_nudge_silent_when_disabled_or_no_rows(monkeypatch):
    import cli
    prints = []
    monkeypatch.setattr(cli, "_cprint", lambda text="": prints.append(text))
    monkeypatch.setattr(cli, "CLI_CONFIG", {"display": {"tui_agents_nudge": False}})
    dock = sm.SubagentMonitor(types.SimpleNamespace(agent=None, session_id="sid", _pending_input=queue.Queue()))
    dock._agents_nudge(True)
    monkeypatch.setattr(cli, "CLI_CONFIG", {"display": {"tui_agents_nudge": True}})
    dock._agents_nudge(False)
    assert prints == []

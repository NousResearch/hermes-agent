"""Tests for `--resume latest` and `--in DIR` launch sugar.

`hermes --tui --resume latest --in ./dir` (and the classic-CLI equivalents)
resolve "latest" through the same workspace-scoped MRU lookup as `-c`, with
`--in` re-homing the process before any session resolution happens.
"""

from __future__ import annotations

from argparse import Namespace

import pytest


def _args(**overrides):
    base = {
        "cli": False,
        "continue_last": None,
        "in_dir": None,
        "model": None,
        "no_restore_cwd": False,
        "provider": None,
        "query": None,
        "resume": None,
        "safe_mode": False,
        "toolsets": None,
        "tui": True,
        "tui_dev": False,
        "worktree": False,
    }
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture
def main_mod(monkeypatch):
    import hermes_cli.main as mod

    monkeypatch.setattr(mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(mod, "_sync_bundled_skills_for_startup", lambda: False)
    monkeypatch.setattr(mod, "_pin_kanban_board_env", lambda: None)
    return mod


@pytest.fixture
def launched(main_mod, monkeypatch):
    """Capture the _launch_tui call instead of exec'ing Node."""
    captured = {}

    def fake_launch(resume_session_id=None, **kwargs):
        captured["resume"] = resume_session_id
        captured.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)
    return captured


# ---------------------------------------------------------------------------
# argparse surface
# ---------------------------------------------------------------------------


def test_top_level_parser_accepts_in_and_resume_latest():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat = build_top_level_parser()
    args = parser.parse_args(["--tui", "--resume", "latest", "--in", "./dir"])
    assert args.tui is True
    assert args.resume == "latest"
    assert args.in_dir == "./dir"


def test_chat_subparser_accepts_in_flag():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat = build_top_level_parser()
    args = parser.parse_args(["chat", "--in", "/tmp", "--resume", "latest"])
    assert args.in_dir == "/tmp"
    assert args.resume == "latest"


def test_top_level_in_value_not_mistaken_for_subcommand(monkeypatch):
    # `hermes --in chat` — "chat" is the flag's value, not the subcommand.
    import sys

    import hermes_cli.main as mod

    monkeypatch.setattr(sys, "argv", ["hermes", "--in", "chat", "--resume", "latest"])
    assert mod._first_positional_argv() is None


# ---------------------------------------------------------------------------
# --resume latest resolution
# ---------------------------------------------------------------------------


def test_resume_latest_resolves_to_mru_session(main_mod, launched, monkeypatch):
    monkeypatch.setattr(
        main_mod, "_resolve_last_session", lambda source="cli": "20260807_120000_abc123"
    )
    # Keyword must NOT fall through to title resolution.
    monkeypatch.setattr(
        main_mod,
        "_resolve_session_by_name_or_id",
        lambda val: val if val != "latest" else pytest.fail("'latest' hit title resolution"),
    )

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(_args(resume="latest"))
    assert exc.value.code == 0
    assert launched["resume"] == "20260807_120000_abc123"


def test_resume_latest_tui_falls_back_to_cli_source(main_mod, launched, monkeypatch):
    calls = []

    def fake_resolve(source="cli"):
        calls.append(source)
        return "cli_session_1" if source == "cli" else None

    monkeypatch.setattr(main_mod, "_resolve_last_session", fake_resolve)
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda v: v)

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(_args(resume="latest"))
    assert exc.value.code == 0
    assert calls == ["tui", "cli"]
    assert launched["resume"] == "cli_session_1"


def test_resume_latest_is_case_insensitive(main_mod, launched, monkeypatch):
    monkeypatch.setattr(main_mod, "_resolve_last_session", lambda source="cli": "sess_1")
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda v: v)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(resume="Latest"))
    assert launched["resume"] == "sess_1"


def test_resume_latest_no_sessions_exits_with_error(main_mod, monkeypatch, capsys):
    monkeypatch.setattr(main_mod, "_resolve_last_session", lambda source="cli": None)

    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(_args(resume="latest"))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "No previous TUI session found" in out


def test_resume_real_id_untouched_by_latest_keyword(main_mod, launched, monkeypatch):
    monkeypatch.setattr(
        main_mod,
        "_resolve_last_session",
        lambda source="cli": pytest.fail("MRU lookup must not run for explicit IDs"),
    )
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda v: v)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(resume="20260807_120000_abc123"))
    assert launched["resume"] == "20260807_120000_abc123"


# ---------------------------------------------------------------------------
# --in DIR
# ---------------------------------------------------------------------------


def test_in_dir_chdirs_before_session_resolution(main_mod, launched, monkeypatch, tmp_path):
    import os

    target = tmp_path / "projdir"
    target.mkdir()
    start = os.getcwd()
    seen_cwd = {}

    def fake_resolve(source="cli"):
        seen_cwd["at_resolve"] = os.getcwd()
        return "sess_scoped"

    monkeypatch.setattr(main_mod, "_resolve_last_session", fake_resolve)
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda v: v)

    try:
        with pytest.raises(SystemExit):
            main_mod.cmd_chat(_args(resume="latest", in_dir=str(target)))
    finally:
        os.chdir(start)

    assert seen_cwd["at_resolve"] == str(target.resolve())
    assert launched["resume"] == "sess_scoped"


def test_in_dir_sets_no_restore_cwd(main_mod, launched, monkeypatch, tmp_path):
    import os

    target = tmp_path / "pin-here"
    target.mkdir()
    start = os.getcwd()

    args = _args(resume=None, in_dir=str(target))
    try:
        with pytest.raises(SystemExit):
            main_mod.cmd_chat(args)
    finally:
        os.chdir(start)

    assert args.no_restore_cwd is True


def test_in_dir_missing_directory_exits(main_mod, monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main_mod.cmd_chat(_args(in_dir=str(tmp_path / "nope")))
    assert exc.value.code == 1
    assert "--in directory not found" in capsys.readouterr().out


def test_in_dir_expands_user_home(main_mod, launched, monkeypatch, tmp_path):
    import os

    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    start = os.getcwd()

    try:
        with pytest.raises(SystemExit):
            main_mod.cmd_chat(_args(in_dir="~/proj"))
        assert os.getcwd() == str((home / "proj").resolve())
    finally:
        os.chdir(start)


def test_in_dir_overrides_inherited_terminal_cwd(
    main_mod, launched, monkeypatch, tmp_path
):
    """A relay's explicit --in must beat its worker parent's TERMINAL_CWD."""
    import os

    from agent.runtime_cwd import resolve_agent_cwd

    relay_home = tmp_path / "relay-home"
    worker_workspace = tmp_path / "kanban" / "boards" / "other" / "workspaces" / "t_old"
    relay_home.mkdir()
    worker_workspace.mkdir(parents=True)
    monkeypatch.setenv("TERMINAL_CWD", str(worker_workspace))
    start = os.getcwd()

    try:
        with pytest.raises(SystemExit):
            main_mod.cmd_chat(_args(in_dir=str(relay_home)))
        assert os.getcwd() == str(relay_home)
        assert resolve_agent_cwd() == relay_home
    finally:
        os.chdir(start)


def test_dispatcher_workspace_overrides_resumed_session_cwd(
    main_mod, launched, monkeypatch, tmp_path
):
    """A dispatcher-owned run stays in its current task workspace on resume."""
    import os

    from agent.runtime_cwd import resolve_agent_cwd
    from hermes_state import SessionDB

    home = tmp_path / "home"
    workspace = tmp_path / "current-board" / "workspaces" / "t_current"
    saved = tmp_path / "other-board" / "workspaces" / "t_stale"
    home.mkdir()
    workspace.mkdir(parents=True)
    saved.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_current")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    db = SessionDB()
    db.create_session("resume-me", source="cli", cwd=str(saved))
    db.close()
    start = os.getcwd()

    try:
        monkeypatch.chdir(workspace)
        with pytest.raises(SystemExit):
            main_mod.cmd_chat(_args(resume="resume-me"))
        assert os.getcwd() == str(workspace)
        assert resolve_agent_cwd() == workspace
    finally:
        os.chdir(start)


def test_resume_restores_existing_saved_cwd(main_mod, launched, monkeypatch, tmp_path):
    import os

    from agent.runtime_cwd import resolve_agent_cwd
    from hermes_state import SessionDB

    home = tmp_path / "home"
    launch = tmp_path / "launch"
    saved = tmp_path / "saved"
    home.mkdir()
    launch.mkdir()
    saved.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    db = SessionDB()
    db.create_session("resume-me", source="cli", cwd=str(saved))
    db.close()
    start = os.getcwd()

    try:
        monkeypatch.chdir(launch)
        with pytest.raises(SystemExit):
            main_mod.cmd_chat(_args(resume="resume-me"))
        assert os.getcwd() == str(saved)
        assert resolve_agent_cwd() == saved
    finally:
        os.chdir(start)


def test_resume_saved_cwd_overrides_inherited_terminal_cwd(
    main_mod, launched, monkeypatch, tmp_path
):
    """Process cwd and the runtime resolver must agree after a normal resume."""
    import os

    from agent.runtime_cwd import resolve_agent_cwd
    from hermes_state import SessionDB

    home = tmp_path / "home"
    launch = tmp_path / "launch"
    inherited = tmp_path / "inherited"
    saved = tmp_path / "saved"
    for directory in (home, launch, inherited, saved):
        directory.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(inherited))
    db = SessionDB()
    db.create_session("resume-me", source="cli", cwd=str(saved))
    db.close()
    start = os.getcwd()

    try:
        monkeypatch.chdir(launch)
        with pytest.raises(SystemExit):
            main_mod.cmd_chat(_args(resume="resume-me"))
        assert os.getcwd() == str(saved)
        assert resolve_agent_cwd() == saved
    finally:
        os.chdir(start)


def test_resume_with_missing_saved_cwd_stays_put_without_creating_it(
    main_mod, launched, monkeypatch, tmp_path
):
    import os

    from agent.runtime_cwd import resolve_agent_cwd
    from hermes_state import SessionDB

    home = tmp_path / "home"
    launch = tmp_path / "launch"
    missing = tmp_path / "kanban" / "boards" / "bogus" / "workspaces" / "t_old"
    home.mkdir()
    launch.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    db = SessionDB()
    db.create_session("resume-me", source="cli", cwd=str(missing))
    db.close()
    start = os.getcwd()

    try:
        monkeypatch.chdir(launch)
        with pytest.raises(SystemExit):
            main_mod.cmd_chat(_args(resume="resume-me"))
        assert os.getcwd() == str(launch)
        assert resolve_agent_cwd() == launch
        assert not missing.exists()
    finally:
        os.chdir(start)

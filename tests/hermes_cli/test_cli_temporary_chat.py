"""``hermes --temporary``: parser flag, resume conflict, blocked commands, and deletion at /new and exit."""

from unittest.mock import MagicMock

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _make_cli(db=None, *, temporary=True):
    from cli import HermesCLI
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.console = MagicMock()
    cli.agent = None
    cli.conversation_history = []
    cli.session_id = "temp-session"
    cli._delete_session_on_exit = temporary
    cli._session_db = db
    cli.temporary = temporary
    cli._console_print = MagicMock()
    return cli


@pytest.mark.parametrize("argv", [["--temporary"], ["chat", "--temporary"]])
def test_parser_accepts_temporary(argv):
    from hermes_cli._parser import build_top_level_parser
    parser, *_ = build_top_level_parser()
    assert parser.parse_args(argv).temporary is True


def test_temporary_cannot_resume():
    from hermes_cli.cli_temporary import init_temporary_mode
    with pytest.raises(ValueError, match="--temporary"):
        init_temporary_mode(MagicMock(), True, "some-session")


@pytest.mark.parametrize("command", ["/resume", "/resume abc", "/branch"])
def test_history_adopting_commands_are_refused(command):
    cli = _make_cli()
    cli._handle_resume_command = MagicMock()
    cli._handle_branch_command = MagicMock()
    assert cli.process_command(command) is True
    cli._handle_resume_command.assert_not_called()
    cli._handle_branch_command.assert_not_called()


def test_new_session_boundary_deletes_the_temporary_chat(db):
    db.create_session("temp-session", source="cli")
    db.append_messages_batch("temp-session", [{"role": "user", "content": "secret"}])
    db.mark_session_temporary("temp-session")
    db.create_session("saved", source="cli")
    db.append_messages_batch("saved", [{"role": "user", "content": "keep"}])
    cli = _make_cli(db)
    cli.conversation_history = [{"role": "user", "content": "secret"}]  # non-empty: a normal CLI would keep it

    assert cli._discard_session_if_empty("temp-session") is True
    assert db.get_session("temp-session") is None
    assert cli._discard_session_if_empty("saved") is False  # never a saved chat
    assert db.get_session("saved") is not None


def test_normal_cli_keeps_a_chat_with_history(db):
    db.create_session("s", source="cli")
    db.append_messages_batch("s", [{"role": "user", "content": "hi"}])
    cli = _make_cli(db, temporary=False)
    cli.conversation_history = [{"role": "user", "content": "hi"}]
    assert cli._discard_session_if_empty("s") is False
    assert db.get_session("s") is not None


def _run_cmd_chat(monkeypatch, argv):
    import sys
    import types

    import hermes_cli.main as main_mod
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=main_mod.cmd_chat)
    args = parser.parse_args(argv)
    calls = {}
    fake_cli = types.ModuleType("cli")
    fake_cli.main = lambda **kwargs: calls.setdefault("cli", kwargs)
    fake_banner = types.ModuleType("hermes_cli.banner")
    fake_banner.prefetch_update_check = lambda: None
    fake_skills_sync = types.ModuleType("tools.skills_sync")
    fake_skills_sync.sync_skills = lambda quiet=True: None
    monkeypatch.setitem(sys.modules, "cli", fake_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.banner", fake_banner)
    monkeypatch.setitem(sys.modules, "tools.skills_sync", fake_skills_sync)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_launch_tui", lambda *a, **kw: calls.setdefault("tui", kw))
    main_mod.cmd_chat(args)
    return calls


def test_classic_cli_receives_temporary(monkeypatch):
    assert _run_cmd_chat(monkeypatch, ["chat", "--temporary"])["cli"]["temporary"] is True


def test_cmd_chat_forwards_temporary_to_the_tui(monkeypatch):
    assert _run_cmd_chat(monkeypatch, ["chat", "--tui", "--temporary"])["tui"]["temporary"] is True
    assert _run_cmd_chat(monkeypatch, ["chat", "--tui"])["tui"]["temporary"] is False


def _launch_tui_capture(monkeypatch, **kwargs):
    import sys
    from pathlib import Path

    from hermes_cli import main_tui_launch

    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    captured = {"summaries": 0}
    monkeypatch.setattr(main_tui_launch, "_make_tui_argv", lambda tui_dir, tui_dev: (["node", "dist/entry.js"], Path(".")))
    monkeypatch.setattr(main_tui_launch.subprocess, "call", lambda argv, cwd=None, env=None: captured.update(env=env) or 0)
    monkeypatch.setattr(main_tui_launch, "_print_tui_exit_summary",
                        lambda *a, **kw: captured.update(summaries=captured["summaries"] + 1))
    with pytest.raises(SystemExit):
        main_tui_launch._launch_tui(**kwargs)
    return captured


def test_tui_launch_runs_the_ink_app_in_temporary_mode(monkeypatch):
    captured = _launch_tui_capture(monkeypatch, temporary=True)
    assert captured["env"]["HERMES_TUI_TEMPORARY"] == "1"
    assert captured["summaries"] == 0  # no "resume this session" hint for a deleted chat


def test_normal_tui_launch_never_inherits_temporary_mode(monkeypatch):
    monkeypatch.setenv("HERMES_TUI_TEMPORARY", "1")  # e.g. exported by an outer temporary TUI
    captured = _launch_tui_capture(monkeypatch)
    assert "HERMES_TUI_TEMPORARY" not in captured["env"]
    assert captured["summaries"] == 1


def test_tui_temporary_cannot_resume():
    from hermes_cli.main_tui_launch import _launch_tui
    with pytest.raises(SystemExit, match="--temporary"):
        _launch_tui(resume_session_id="abc", temporary=True)


def test_exit_summary_never_offers_to_resume_a_temporary_chat(capsys):
    cli = _make_cli()
    cli.conversation_history = [{"role": "user", "content": "secret"}, {"role": "assistant", "content": "ok"}]
    cli._print_exit_summary(clear_screen=False)
    out = capsys.readouterr().out
    assert "Temporary chat deleted" in out
    assert "--resume" not in out


def test_temporary_cli_recalls_but_never_writes_prompt_history(tmp_path):
    from types import SimpleNamespace

    from hermes_cli.cli_temporary import prompt_history
    path = tmp_path / ".hermes_history"
    saved = prompt_history(SimpleNamespace(_history_file=path, temporary=False))
    saved.append_string("an older saved prompt")
    temp = prompt_history(SimpleNamespace(_history_file=path, temporary=True))
    assert list(temp.load_history_strings()) == ["an older saved prompt"]
    temp.append_string("my secret prompt")
    assert "my secret prompt" in temp.get_strings()  # up-arrow works inside the chat
    assert "secret" not in path.read_text()

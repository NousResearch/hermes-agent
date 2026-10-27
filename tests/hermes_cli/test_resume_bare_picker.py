"""Bare ``hermes --resume`` picks a session from this workspace instead of demanding an ID.

Covers the argparse surface (``nargs="?"``), the workspace scoping of the picker, and the
refusals that keep a non-interactive launch from blocking on a prompt nobody can answer.
"""

import contextlib
import sys
from argparse import Namespace

import pytest

from hermes_cli import main_resume_args
from hermes_cli._parser import build_top_level_parser


@pytest.fixture
def db(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")

    @contextlib.contextmanager
    def _db():
        yield db

    monkeypatch.setattr("hermes_cli.main._session_db", _db)
    yield db
    db.close()


def _session(db, sid, preview, *, git_repo_root):
    db.create_session(sid, "cli", cwd=git_repo_root, git_repo_root=git_repo_root)
    db.append_message(sid, "user", preview)


def _args(**kw):
    base = dict(resume=True, query=None, oneshot=None, oneshot_exit=False, source=None)
    base.update(kw)
    return Namespace(**base)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    key = str(tmp_path / "repo")
    monkeypatch.setattr("hermes_cli.main._resolve_workspace_key", lambda: key)
    return key


@pytest.fixture
def interactive(monkeypatch):
    """A launch that may prompt (the TTY check itself is tested separately)."""
    monkeypatch.setattr(main_resume_args, "_can_prompt", lambda args: True)


@pytest.fixture
def picker(monkeypatch):
    seen = {}

    def _fake(sessions, session_db=None):
        seen["ids"] = [s["id"] for s in sessions]
        seen["db_open"] = session_db is not None
        return seen.get("choice")

    monkeypatch.setattr("hermes_cli.sessions_cmd_browse._session_browse_picker", _fake)
    return seen


def test_parser_accepts_bare_resume():
    parser, _subparsers, chat = build_top_level_parser()

    assert parser.parse_args(["--resume"]).resume is True
    assert parser.parse_args(["-r"]).resume is True
    assert parser.parse_args(["--resume", "abc123"]).resume == "abc123"
    assert chat.parse_args(["--resume"]).resume is True


def test_blank_resume_value_counts_as_bare():
    assert main_resume_args.is_bare_resume(_args(resume="  "))
    assert main_resume_args.is_bare_resume(_args(resume=True))
    assert not main_resume_args.is_bare_resume(_args(resume="latest"))
    assert not main_resume_args.is_bare_resume(Namespace())


def test_can_prompt_refuses_one_shots(monkeypatch):
    monkeypatch.setattr(sys, "stdin", Namespace(isatty=lambda: True))
    monkeypatch.setattr(sys, "stdout", Namespace(isatty=lambda: True))

    assert main_resume_args._can_prompt(_args())
    assert not main_resume_args._can_prompt(_args(query="hello"))
    assert not main_resume_args._can_prompt(_args(oneshot="hello"))
    assert not main_resume_args._can_prompt(_args(oneshot_exit=True))

    monkeypatch.setattr(sys, "stdin", Namespace(isatty=lambda: False))
    assert not main_resume_args._can_prompt(_args())


def test_non_interactive_launch_refuses_without_prompting(picker, capsys):
    with pytest.raises(SystemExit) as exc:
        main_resume_args.resolve_bare_resume(_args(), interactive=False)

    assert exc.value.code == 2
    assert "--resume needs a session ID" in capsys.readouterr().err
    assert "ids" not in picker  # the picker never ran


def test_id_resume_is_untouched(monkeypatch):
    def _boom():  # a resolved ID must not open the session store at all
        raise AssertionError("no lookup expected")

    monkeypatch.setattr("hermes_cli.main._session_db", _boom)
    args = _args(resume="20250305_091523_a1b2c3d4")

    main_resume_args.resolve_bare_resume(args)
    assert args.resume == "20250305_091523_a1b2c3d4"


def test_picker_shows_only_this_workspace(db, workspace, interactive, picker):
    _session(db, "here", "here", git_repo_root=workspace)
    _session(db, "there", "there", git_repo_root="/elsewhere/other")
    picker["choice"] = "here"
    args = _args()

    main_resume_args.resolve_bare_resume(args)

    assert picker["ids"] == ["here"]
    assert picker["db_open"] is True  # status tags / delete stay available while picking
    assert args.resume == "here"


def test_empty_workspace_exits_with_a_pointer(db, workspace, interactive, picker, capsys):
    _session(db, "there", "there", git_repo_root="/elsewhere/other")

    with pytest.raises(SystemExit) as exc:
        main_resume_args.resolve_bare_resume(_args())

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "No sessions in repo." in out
    assert "hermes sessions browse" in out
    assert "ids" not in picker


def test_cancel_exits_without_resuming(db, workspace, interactive, picker, capsys):
    _session(db, "here", "here", git_repo_root=workspace)
    picker["choice"] = None

    with pytest.raises(SystemExit) as exc:
        main_resume_args.resolve_bare_resume(_args())

    assert exc.value.code == 1
    assert "Cancelled." in capsys.readouterr().out

import os
from types import SimpleNamespace

import pytest



def _chat_args(session_id: str, *, allow_parallel_owner: bool = False):
    return SimpleNamespace(
        query=None,
        slash=None,
        query_file=None,
        stdin_query=False,
        image=None,
        continue_last=None,
        resume=session_id,
        tui=False,
        cli=True,
        allow_parallel_owner=allow_parallel_owner,
        model=None,
        provider=None,
        toolsets=None,
        skills=None,
        verbose=None,
        quiet=False,
        worktree=False,
        checkpoints=False,
        pass_session_id=False,
        max_turns=None,
        accept_hooks=False,
        ignore_user_config=False,
        ignore_rules=False,
        safe_mode=False,
        compact=False,
        source=None,
        tui_dev=False,
    )



def test_chat_resume_refuses_live_duplicate_owner_before_startup(tmp_path, monkeypatch, capsys):
    from hermes_cli import active_sessions
    import hermes_cli.main as hermes_main

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(hermes_main, "_resolve_use_tui", lambda _args: False)
    monkeypatch.setattr(
        hermes_main,
        "_has_any_provider_configured",
        lambda: (_ for _ in ()).throw(AssertionError("provider guard should not run")),
    )

    lease, message = active_sessions.try_acquire_active_session(
        session_id="duplicate-session",
        surface="cli",
        config={},
    )
    assert message is None
    assert lease is not None

    try:
        with pytest.raises(SystemExit) as exc:
            hermes_main.cmd_chat(_chat_args("duplicate-session"))
    finally:
        lease.release()

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "live owner already controls this session" in captured.err
    assert f"pid={os.getpid()}" in captured.err
    assert "owner_kind=cli" in captured.err
    assert "command_line_fingerprint=" in captured.err
    assert "duplicate-session" not in captured.err
    assert "cwd_fingerprint" not in captured.err



def test_chat_resume_allows_duplicate_owner_only_with_explicit_flag(tmp_path, monkeypatch):
    from hermes_cli import active_sessions
    import hermes_cli.main as hermes_main

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(hermes_main, "_resolve_use_tui", lambda _args: False)
    monkeypatch.setattr(
        hermes_main,
        "_has_any_provider_configured",
        lambda: (_ for _ in ()).throw(SystemExit(17)),
    )

    lease, message = active_sessions.try_acquire_active_session(
        session_id="parallel-session",
        surface="cli",
        config={},
    )
    assert message is None
    assert lease is not None

    try:
        with pytest.raises(SystemExit) as exc:
            hermes_main.cmd_chat(
                _chat_args("parallel-session", allow_parallel_owner=True)
            )
    finally:
        lease.release()

    assert exc.value.code == 17



def test_chat_parser_accepts_allow_parallel_owner_flag():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat_parser = build_top_level_parser()
    args = parser.parse_args([
        "chat",
        "--resume",
        "some-session",
        "--allow-parallel-owner",
    ])

    assert args.resume == "some-session"
    assert args.allow_parallel_owner is True

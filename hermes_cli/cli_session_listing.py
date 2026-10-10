"""CLI session-list command routing."""


def handle_sessions_command(cli, cmd_original: str) -> None:
    """A named target resumes; the bare command displays recent sessions."""
    from hermes_cli.cli_commands_mixin import _command_arg, _cp, _db_unavailable_line, _t

    arg = _command_arg(cmd_original)
    if arg and arg.lower() not in {"list", "ls", "browse"}:
        cli._handle_resume_command(f"/resume {arg}")
    elif not cli._session_db:
        _cp(_db_unavailable_line())
    elif not cli._show_recent_sessions(reason="sessions"):
        _cp(f"  {_t('sessions.none_yet')}")

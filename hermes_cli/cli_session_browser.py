"""Classic CLI entry into the shared session picker (never a second browser)."""

import asyncio
import shutil
import sys


def can_browse_sessions() -> bool:
    """Only a real input/output terminal may own the keyboard."""
    return bool(sys.stdin and sys.stdout and sys.stdin.isatty() and sys.stdout.isatty())


def is_resume_browser_request(cli, target: str) -> bool:
    """Bare list aliases are verbs unless they identify an existing session."""
    if not target:
        return True
    if target.lower() not in {"list", "ls", "browse"}:
        return False
    db = cli._session_db
    return not db or not (db.get_session(target) or db.get_session_by_title(target))


def render_recent_sessions(sessions, *, reason):
    """Safe, compact static fallback; numeric /resume keeps its ten-row contract."""
    from agent.i18n import t
    from cli import _cli_visible_print
    from hermes_cli.sessions_cmd_browse import _clip, _format_compact_row

    width = shutil.get_terminal_size((80, 24)).columns
    header = t("cli.session.recent_intro") if reason == "history" else f"  {t('cli.session.recent_header')}"
    lines = ["", header, ""]
    for index, session in enumerate(sessions, 1):
        prefix = f"  {index}. "
        lines.extend([prefix + _format_compact_row(session, max(1, width - len(prefix))),
                      f"     {session['id']}"])
    lines.extend(["", f"  {t('cli.session.resume_usage')}", f"  {t('cli.session.resume_example')}", ""])
    for line in lines:
        _cli_visible_print(_clip(line, width))


def show_session_choices(cli, *, reason):
    """One entry point for bare /sessions and /resume, with nonblocking pipe output."""
    from hermes_cli.cli_commands_mixin import _cp, _db_unavailable_line, _t
    from hermes_cli.sessions_cmd_browse import _clip

    if not cli._session_db:
        _cp(_db_unavailable_line())
        return
    if can_browse_sessions():
        shown = browse_recent_sessions(cli)
    else:
        if reason == "resume":
            _cp(_clip(f"  {_t('resume.usage')}", shutil.get_terminal_size((80, 24)).columns))
        shown = cli._show_recent_sessions(reason=reason)
        if shown and reason == "resume":
            cli._pending_resume_sessions = cli._list_recent_sessions(limit=10)
    if not shown:
        key = "resume.tip_find_sessions" if reason == "resume" else "sessions.none_yet"
        _cp(f"  {_t(key)}")


def _pick_or_list(sessions):
    """Unavailable curses gets visible choices, not cancellation or a hidden input prompt."""
    from hermes_cli.sessions_cmd_browse import SessionBrowserUnavailable, _session_browse_picker

    try:
        return _session_browse_picker(sessions, keyboard_navigation=True)
    except SessionBrowserUnavailable:
        render_recent_sessions(sessions[:10], reason="resume")
        return None


async def _pick_in_terminal(app, sessions):
    """Detach prompt_toolkit input/renderer on its owning loop, including stdout paints."""
    from prompt_toolkit.application import in_terminal
    from prompt_toolkit.application.current import set_app

    with set_app(app):
        async with in_terminal():
            # Keep curses on the loop thread: patch_stdout's queued paints cannot interleave
            # with its screen. in_terminal restores raw input and redraws on every exit.
            return _pick_or_list(sessions)


async def _browse_on_loop(cli, sessions):
    selected = await _pick_in_terminal(cli._app, sessions)
    if selected is not None:
        cli._handle_resume_command(f"/resume {selected}")


def browse_recent_sessions(cli) -> bool:
    """Resume the selected snapshot ID; cancellation leaves the active conversation intact."""
    sessions = cli._list_recent_sessions(limit=None)
    cli._listed_resume_sessions = sessions
    if not sessions:
        return False
    cli._pending_resume_sessions = None
    app = getattr(cli, "_app", None)
    if app is not None and app.is_running:
        try:
            caller_loop = asyncio.get_running_loop()
        except RuntimeError:
            caller_loop = None
        if caller_loop is app.loop:
            app.create_background_task(_browse_on_loop(cli, sessions))
            return True
        selected = asyncio.run_coroutine_threadsafe(_pick_in_terminal(app, sessions), app.loop).result()
    else:
        selected = _pick_or_list(sessions)
    if selected is not None:
        cli._handle_resume_command(f"/resume {selected}")
    return True

"""What a local chat launcher does around its client process: the command that runs this Hermes
install, and the lines printed after the client exits. Public for plugins (plugin_api.json); the
Ink launcher (``main_tui_launch.py``) uses the same functions.
"""

import json
import os
import sqlite3
from pathlib import Path

from agent.i18n import t


def hermes_command(args=()) -> list[str]:
    """The argv that runs this Hermes install with *args*, for a child process or a client that
    keeps it: the install's launcher file when it has one, else this interpreter with an inline
    bootstrap. It needs no extra environment (no ``PYTHONPATH``); the child serves the profile
    named by the inherited ``HERMES_HOME``. The first item is always a file a child process can
    start without a shell."""
    from hermes_cli._launchers import installation_command, runtime_command
    from hermes_cli.main import PROJECT_ROOT
    command = installation_command(PROJECT_ROOT, args)
    launcher = Path(command[0])
    if os.name != "nt" or launcher.name != "hermes":
        return command
    # Windows publishes ``hermes.exe``, or ``hermes.cmd`` when it could not make one. Neither is
    # the extensionless path, and a batch file is not startable without a shell.
    exe = launcher.with_name("hermes.exe")
    return [str(exe), *command[1:]] if exe.is_file() else runtime_command(PROJECT_ROOT, args)


# A canonical local session is keyed ``local-`` + 64 hex; main's epilogue printed 22-char stored ids.
_SHORT_RESUME_LEN = 22


def _short_resume_id(db, session_id: str) -> str:
    """The 22-char prefix of a canonical ``local-…`` id when it names only that session (``--resume``
    resolves a unique prefix back); every other id, or an ambiguous prefix, prints whole."""
    if not session_id.startswith("local-") or len(session_id) <= _SHORT_RESUME_LEN:
        return session_id
    short = session_id[:_SHORT_RESUME_LEN]
    try:
        return short if db.resolve_session_id(short) == session_id else session_id
    except sqlite3.Error:  # an unreadable store prints the whole id
        return session_id


def _branch_lineage(db, session: dict) -> list:
    """*session* plus each ``/branch`` ancestor it was cut from. Main's epilogue after ``/branch`` still
    named the parent and so reported the conversation's tokens; a canonical branch child is named
    itself, and its copied history was billed on the ancestors."""
    rows, seen = [session], {session.get("id")}
    while True:
        try:
            config = json.loads(rows[-1].get("model_config") or "{}")
        except (TypeError, ValueError):
            break
        parent_id = config.get("_branched_from") if isinstance(config, dict) else None
        parent = db.get_session(parent_id) if parent_id and parent_id not in seen else None
        if not parent:
            break
        seen.add(parent_id)
        rows.append(parent)
    return rows


def print_exit_summary(session_id: str, *, command: str) -> None:
    """Print the lines a chat launcher shows after its client exits: how to resume the session
    with *command* (for example ``hermes --tui``), its id, title, message count and tokens. Prints
    nothing when the profile's store has no such session or the session has no messages."""
    db = None
    try:
        from hermes_state import SessionDB
        from hermes_cli.config import get_hermes_home
        db = SessionDB(db_path=get_hermes_home() / "state.db", read_only=True)
        session = db.get_session(session_id)
        if not session:
            return

        title = db.get_session_title(session_id)
        resume_id = _short_resume_id(db, session_id)
        message_count = int(session.get("message_count") or 0)
        if message_count == 0:
            return  # No real conversation — don't show resume info
        tokens = {k: 0 for k in ("input", "output", "cache_read", "cache_write", "reasoning")}
        for row in _branch_lineage(db, session):
            for k in tokens:
                tokens[k] += int(row.get(f"{k}_tokens") or 0)
    except Exception:  # health: allow BLE001 -- the epilogue never fails the launcher
        return
    finally:
        if db is not None:
            db.close()

    # Sessions live under the profile's HERMES_HOME — mirror cli_session_mixin::_print_exit_summary.
    from hermes_cli.profiles import get_active_profile_name
    active_profile = get_active_profile_name()
    profile_flag = "" if active_profile in ("default", "custom") else f" -p {active_profile}"
    print(f"\n{t('cli.session.exit_resume_hint')}\n  {command} --resume {resume_id}{profile_flag}")
    if title:
        print(f'  {command} -c "{title}"{profile_flag}')
    print(f"\nSession:        {resume_id}")
    if title:
        print(f"Title:          {title}")
    print(f"Messages:       {message_count}")
    print(
        "Tokens:         "
        f"{sum(tokens.values())} (in {tokens['input']}, out {tokens['output']}, "
        f"cache {tokens['cache_read'] + tokens['cache_write']}, reasoning {tokens['reasoning']})"
    )

"""``hermes --resume`` target helpers: foreign-session import and the owning-profile lookup.

A bare ``hermes --resume <id>`` opens the sticky active profile, so an id minted under another
profile failed with "Session not found" (a terminal multiplexer restoring panes types exactly that
into a fresh shell, with no ``-p`` and no ``HERMES_HOME``). :func:`follow_resume_owner` runs inside
the ``-p`` pre-parse, before ``HERMES_HOME`` is exported or any ``.env`` is loaded, so following
the owner is the same as having typed ``-p <owner>``: no re-exec, and no launch-profile secrets in
the resumed profile's environment. Only modules ``-p`` itself loads (stdlib, the parser,
``hermes_cli.profiles``) may load on that path, and only when an id-shaped value was typed.
"""

from __future__ import annotations

import sys
from contextlib import closing
from pathlib import Path


def _import_foreign_resume(args) -> None:
    """--resume @claude / @codex: import a foreign session and resume it."""
    _resume_foreign = getattr(args, "resume", None)
    if not (isinstance(_resume_foreign, str) and _resume_foreign.strip().lower() in ("@claude", "@codex")):
        return
    from hermes_cli.foreign_sessions import import_foreign_session, pick_foreign_session

    _picked = pick_foreign_session(_resume_foreign.strip().lower().lstrip("@"))
    if _picked is None:
        sys.exit(1)
    try:
        _imported_id = import_foreign_session(_picked.source, _picked.path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"✓ Imported as {_imported_id} — resuming it now.")
    print(f"  (later: hermes --resume {_imported_id})")
    args.resume = _imported_id


def _resume_id_from_argv(argv: list[str]) -> str | None:
    """The ``--resume``/``-r`` value of a chat launch when it has a minted session-id shape.

    Titles (including unquoted multi-word ones), ``latest`` and ``@claude``/``@codex`` never
    qualify, nor does a ``-r`` that belongs to another subcommand.
    """
    from hermes_state_ids import is_known_session_id

    value = None
    for index, token in enumerate(argv):
        if token == "--":
            break
        if token in ("--resume", "-r") and index + 1 < len(argv):
            words_follow = index + 2 < len(argv) and not argv[index + 2].startswith("-")
            value = None if words_follow else argv[index + 1]
        elif token.startswith("--resume="):
            value = token.split("=", 1)[1]
    if not (value and is_known_session_id(value)):
        return None
    from hermes_cli._parser import command_argv

    command = command_argv(argv)
    return value if not command or command[0] == "chat" else None


def _store_has_session(home: Path, session_id: str, *, match_title: bool = False) -> bool | None:
    """Whether ``home/state.db`` holds ``session_id`` (or a title equal to it); None when unreadable.

    ``mode=ro`` behind an ``is_file`` gate: no store is created, scaffolded or migrated.
    """
    import sqlite3

    db_path = home / "state.db"
    if not db_path.is_file():
        return False
    query = "SELECT 1 FROM sessions WHERE id = ?" + (" OR title = ?" if match_title else "")
    params = (session_id, session_id) if match_title else (session_id,)
    try:
        with closing(sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=1.0)) as conn:
            row = conn.execute(query + " LIMIT 1", params).fetchone()
    except sqlite3.Error:  # locked, corrupt or pre-schema
        return None
    return row is not None


def _owning_profiles(session_id: str, active_home: Path) -> list[str]:
    """Live profiles (``default`` included) other than ``active_home`` whose store holds the id."""
    from hermes_cli.profiles import list_profile_names, resolve_profile_env

    current = active_home.resolve()
    owners = []
    for name in list_profile_names():
        home = Path(resolve_profile_env(name))
        if home.resolve() != current and _store_has_session(home, session_id):
            owners.append(name)
    return owners


def follow_resume_owner(argv: list[str], profile_name: str | None, *, pinned: bool,
                        active_home: Path | None = None) -> str | None:
    """Profile to launch: ``profile_name``, or the one other profile owning a ``--resume`` id.

    Looks only when the typed id resolves neither as an id nor a title in the active store.
    ``pinned`` (``-p``/``--profile`` or a profile-shaped ``HERMES_HOME``) is the user's own choice
    and is never overridden: a hint names the owner and the active profile reports "Session not
    found" as before. Several owners are listed, never guessed. A lookup failure keeps the profile.
    """
    session_id = _resume_id_from_argv(argv)
    if session_id is None:
        return profile_name
    from hermes_cli.profiles import resolve_profile_env

    try:
        home = active_home or Path(resolve_profile_env(profile_name or "default"))
        if _store_has_session(home, session_id, match_title=True) is not False:
            return profile_name
        owners = _owning_profiles(session_id, home)
    except (OSError, ValueError):  # stale sticky profile, unreadable root: the normal path reports it
        return profile_name
    if len(owners) > 1:
        print(f"Session {session_id} exists in profiles {', '.join(owners)}; "
              f"pick one: hermes -p <profile> --resume {session_id}", file=sys.stderr)
    elif owners and pinned:
        print(f"Session {session_id} belongs to profile {owners[0]}: "
              f"hermes -p {owners[0]} --resume {session_id}", file=sys.stderr)
    elif owners:
        print(f"Session {session_id} belongs to profile {owners[0]}; resuming there.", file=sys.stderr)
        return owners[0]
    return profile_name

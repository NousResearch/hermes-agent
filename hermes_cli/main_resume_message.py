"""``hermes -r <session> fix the bug``: text after the session target becomes the first turn.

``_coalesce_session_name_args`` joins every bare word after ``-r``/``-c`` into one session name
so unquoted multi-word titles work (``hermes -c Pokemon Agent Dev``). The same join swallowed a
trailing message: ``hermes -r 20261007_191147_38b910 what changed?`` failed with ``Session not
found: 20261007_191147_38b910 what changed?``. Here the joined value is split back at the
longest leading run of words that names a real session (or the ``latest`` keyword), and the rest
is handed to the chat as ``args.query`` — the same channel ``-q`` uses, so a TTY seeds the
interactive session with it and a non-TTY/``-Q`` run answers and exits.

A value that resolves whole always wins, so every name that resumed before still resumes the
same session with no message.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_WORD_GAP = re.compile(r"\s+")


def _coalesce_session_name_args(argv: list) -> list:
    """Join unquoted multi-word session names after -c/--continue and -r/--resume.

    ``hermes -c Pokemon Agent Dev`` → ``['-c', 'Pokemon Agent Dev']``; tokens
    are collected until the next flag (``-*``) or known top-level subcommand.
    """
    _SUBCOMMANDS = {
        "chat", "model", "gateway", "setup", "whatsapp", "whatsapp-cloud", "login", "logout",
        "auth", "status", "cron", "doctor", "config", "pairing", "skills", "tools", "mcp",
        "sessions", "insights", "update", "uninstall", "profile", "dashboard", "serve",
        "desktop", "gui", "honcho", "claw", "plugins", "security", "acp", "webhook", "peer",
        "memory", "dump", "debug", "backup", "import", "completion", "logs", "usage",
    }
    _SESSION_FLAGS = {"-c", "--continue", "-r", "--resume"}

    result = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in _SESSION_FLAGS:
            result.append(token)
            i += 1
            # Collect subsequent non-flag, non-subcommand tokens as one name
            parts: list = []
            while (
                i < len(argv)
                and not argv[i].startswith("-")
                and argv[i] not in _SUBCOMMANDS
            ):
                parts.append(argv[i])
                i += 1
            if parts:
                result.append(" ".join(parts))
        else:
            result.append(token)
            i += 1
    return result


def split_session_target(raw: str, *, allow_latest: bool) -> Optional[Tuple[str, str]]:
    """``(target, message)`` for the longest word prefix of ``raw`` naming a session, else None."""
    from hermes_cli.main import _resolve_session_by_name_or_id

    text = raw.strip()
    gaps = list(_WORD_GAP.finditer(text))
    if not gaps or _resolve_session_by_name_or_id(text):
        return None
    for gap in reversed(gaps):
        target, message = text[:gap.start()], text[gap.end():]
        if (allow_latest and target.lower() == "latest") or _resolve_session_by_name_or_id(target):
            return target, message
    return None


def apply_resume_message(args) -> None:
    """Move a message trailing ``--resume``/``--continue <name>`` into ``args.query``.

    Skipped when the run already has a prompt (``-q``, ``--query-file``, ``-z``) and for
    ``--create-if-missing``, whose callers name a thread that may not exist yet.
    """
    if any(getattr(args, attr, None) for attr in ("query", "query_file", "oneshot", "create_if_missing")):
        return
    for attr, allow_latest in (("resume", True), ("continue_last", False)):
        raw = getattr(args, attr, None)
        if not isinstance(raw, str):
            continue
        split = split_session_target(raw, allow_latest=allow_latest)
        if split:
            setattr(args, attr, split[0])
            args.query = split[1]
        return

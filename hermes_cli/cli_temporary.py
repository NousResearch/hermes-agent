"""``hermes --temporary``: a classic-CLI chat that leaves nothing behind.

The agent side (no memory, no recall, no title, hidden + unsearchable row) is
``agent.temporary_chat``; this module owns the CLI surface: argument checks, the
startup banner, the commands that would leak or adopt history, and deleting the
row on ``/new`` and exit (``_tui_shutdown`` deletes via ``_delete_session_on_exit``).
"""

from __future__ import annotations

import logging

from agent.i18n import t

logger = logging.getLogger(__name__)

# /resume would turn a saved conversation into this process's temporary chat (and delete it on
# exit); /branch would copy the temporary transcript into a normal, persisted session.
TEMPORARY_BLOCKED_COMMANDS = frozenset({"resume", "branch"})


def init_temporary_mode(cli, temporary: bool, resume) -> None:
    if temporary and resume:
        raise ValueError(t("cli.session.temporary_resume_conflict"))
    cli.temporary = bool(temporary)


def temporary_banner(cli) -> None:
    if getattr(cli, "temporary", False):
        cli._console_print(f"[bold magenta]{t('cli.session.temporary_banner')}[/]")


def refuse_in_temporary_chat(cli, canonical: str) -> bool:
    """True (after telling the user) when *canonical* is blocked in this temporary chat."""
    if not getattr(cli, "temporary", False) or canonical not in TEMPORARY_BLOCKED_COMMANDS:
        return False
    cli._console_print(f"[yellow]{t('cli.session.temporary_blocked', command=canonical)}[/]")
    return True


def prompt_history(cli):
    """The input history for the prompt: a temporary chat recalls earlier prompts (up-arrow) but
    never appends its own to ``.hermes_history`` — they live in memory for this chat only."""
    from prompt_toolkit.history import FileHistory

    if not getattr(cli, "temporary", False):
        return FileHistory(str(cli._history_file))

    class _RecallOnlyFileHistory(FileHistory):
        def store_string(self, string: str) -> None:
            return None

    return _RecallOnlyFileHistory(str(cli._history_file))


def discard_temporary_session(session_db, session_id: str) -> bool:
    """Delete the temporary chat *session_id* (and its compression chain): /new boundary, one-shot exit."""
    from hermes_constants import get_hermes_home
    try:
        return session_db.delete_temporary_session(session_id, sessions_dir=get_hermes_home() / "sessions")
    except Exception:
        logger.debug("Could not delete temporary session %s", session_id, exc_info=True)
        return False

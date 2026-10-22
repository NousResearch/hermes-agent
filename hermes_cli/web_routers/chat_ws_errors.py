"""User-facing copy for dashboard chat (``/api/pty``) start failures.

``pty_ws`` writes one red line into the terminal and closes with 1011 when the
``hermes --tui`` child cannot be spawned. The exception text alone is either
raw errno noise (``[Errno 2] No such file or directory: 'node'``), a bare exit
code (``SystemExit(1)`` after ``_make_tui_argv`` already printed to the server
log), or empty (``RegistryFull``). Turn each into what happened + what to do.
"""

from __future__ import annotations

from fastapi import HTTPException

from hermes_cli.pty_session import RegistryFull

CHAT_NEEDS_NODE = (
    "Chat could not start: Hermes needs Node.js to run the terminal chat. "
    "Install Node 18+ (for example from nodejs.org) and reopen this tab."
)
CHAT_TOO_MANY_TERMINALS = (
    "Chat could not start: too many chat terminals are open in other tabs. "
    "Close one and click Start new session."
)
CHAT_PROFILE_UNKNOWN = "Chat could not start: {detail} Pick another profile from the switcher and reopen this tab."
CHAT_START_FAILED = "Chat could not start: {detail} Check the server log (`hermes dashboard` terminal) and click Start new session."


def _node_missing(exc: BaseException) -> bool:
    text = str(exc)
    return isinstance(exc, SystemExit) or "'node'" in text or "'npm'" in text or "Node.js" in text


def chat_start_failure_message(exc: BaseException) -> str:
    """One plain sentence for the terminal pane; never the bare exception."""
    if isinstance(exc, RegistryFull):
        return CHAT_TOO_MANY_TERMINALS
    if isinstance(exc, HTTPException):
        return CHAT_PROFILE_UNKNOWN.format(detail=_sentence(str(exc.detail)))
    if _node_missing(exc):
        return CHAT_NEEDS_NODE
    return CHAT_START_FAILED.format(detail=_sentence(str(exc) or type(exc).__name__))


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return "the terminal process could not be launched."
    return text if text.endswith((".", "!", "?")) else f"{text}."

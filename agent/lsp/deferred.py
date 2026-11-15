"""LSP diagnostics for a write, delivered after the write's tool result instead of inside it.

``write_file`` / ``patch`` return as soon as the bytes are on disk and verified; the language server
check runs on the LSP loop (``agent.lsp.edit_check``).  Its verdict is queued here under the session
that made the edit (bound by ``tools.file_tools`` around the write) and handed to that session's agent
at its next tool result, or its next user turn when no tool runs again
(``agent.plugin_live_notes``: a new row not yet flushed, so no cached byte moves).

A later edit of the same file takes over an undelivered verdict: the newer check diffs against the
older baseline, so an error the first edit introduced is still reported unless the second fixed it.

A verdict must not be lost when the edit was the turn's last tool call (a one-shot ``-q`` run has no
next turn): checks still running are tracked per session (:func:`track_pending`), and before the turn's
final answer is accepted :func:`settle_for_agent` waits for them (bounded) and returns every verdict
still undelivered, which ``agent.turn_stop_gates`` hands to the model as a new message.
"""
from __future__ import annotations

import contextlib
import threading
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional

_OWNER: ContextVar[Optional[str]] = ContextVar("hermes_lsp_deferred_owner", default=None)
_MAX_OWNERS = 64  # sessions that never drain (a one-shot caller) must not grow this forever
_MAX_PER_OWNER = 32
_LOCK = threading.Lock()
_SETTLED = threading.Condition(_LOCK)
# A turn-end wait longer than this hands the verdict to the next user turn instead.
TURN_END_WAIT = 5.0


@dataclass(frozen=True)
class _Note:
    text: str
    baseline_text: Optional[str]  # pre-edit content the verdict was diffed against


# owner -> {abs path -> note}, oldest owner first
_QUEUED: OrderedDict[str, OrderedDict[str, _Note]] = OrderedDict()
# owner -> tokens of its checks still running (their verdict lands in _QUEUED or nowhere)
_PENDING: dict[str, set[object]] = {}


@contextlib.contextmanager
def bind_owner(owner: Optional[str]) -> Iterator[None]:
    """Attribute writes in this block to ``owner`` (the session id, else the task id)."""
    token = _OWNER.set(owner or None)
    try:
        yield
    finally:
        _OWNER.reset(token)


def current_owner() -> Optional[str]:
    return _OWNER.get()


def post(owner: str, path: str, text: str, baseline_text: Optional[str]) -> None:
    """Queue ``text`` (a formatted diagnostics block) for ``owner``.  A note still queued for ``path``
    (an older check finished while a newer one was starting) is kept in front, with its baseline."""
    with _LOCK:
        notes = _QUEUED.pop(owner, None) or OrderedDict()
        older = notes.pop(path, None)
        if older is not None and older.text != text:
            text, baseline_text = f"{older.text}\n\n{text}", older.baseline_text
        notes[path] = _Note(text, baseline_text)
        while len(notes) > _MAX_PER_OWNER:
            notes.popitem(last=False)
        _QUEUED[owner] = notes
        while len(_QUEUED) > _MAX_OWNERS:
            _QUEUED.popitem(last=False)


def take(owner: Optional[str], path: str) -> Optional[_Note]:
    """Withdraw ``owner``'s undelivered note for ``path``: a newer edit of the file supersedes it and
    diffs against its ``baseline_text``.  The newer check carries it and hands it back to
    :func:`restore` only if it ends with no verdict, so the older errors are not lost."""
    if not owner:
        return None
    with _LOCK:
        notes = _QUEUED.get(owner)
        return notes.pop(path, None) if notes else None


def restore(owner: Optional[str], path: str, notes) -> None:
    """Queue ``notes`` (oldest first) withdrawn by :func:`take` again."""
    if owner:
        for note in notes:
            post(owner, path, note.text, note.baseline_text)


def pop_notes(*owners: Optional[str]) -> list[tuple[str, str]]:
    """Every queued ``(path, text)`` for ``owners``, oldest first, and clear them."""
    out: list[tuple[str, str]] = []
    with _LOCK:
        for owner in dict.fromkeys(o for o in owners if o):
            notes = _QUEUED.pop(owner, None)
            if notes:
                out.extend((path, note.text) for path, note in notes.items())
    return out


def track_pending(owner: str) -> object:
    """Record a check of ``owner``'s still running; pass the token to :func:`settled` once its verdict
    has been posted (or dropped)."""
    token = object()
    with _LOCK:
        _PENDING.setdefault(owner, set()).add(token)
    return token


def settled(owner: str, token: object) -> None:
    with _SETTLED:
        tokens = _PENDING.get(owner)
        if tokens is not None:
            tokens.discard(token)
            if not tokens:
                del _PENDING[owner]
        _SETTLED.notify_all()


def _owners(agent: Any) -> tuple[Optional[str], ...]:
    owners = (getattr(agent, "session_id", None), getattr(agent, "_current_task_id", None))
    return tuple(o if isinstance(o, str) else None for o in owners)


def settle_for_agent(agent: Any, timeout: float = TURN_END_WAIT) -> list[tuple[str, str]]:
    """Turn end: wait up to ``timeout`` for ``agent``'s running checks, then pop every undelivered
    ``(path, text)`` verdict.  Returns at once when nothing is running."""
    owners = [o for o in dict.fromkeys(_owners(agent)) if o]
    if not owners:
        return []
    with _SETTLED:
        _SETTLED.wait_for(lambda: not any(_PENDING.get(o) for o in owners), timeout=timeout)
    return pop_notes(*owners)


def forward_to_agent(agent: Any) -> None:
    """Move ``agent``'s finished verdicts onto its live-notes queue, which rides the next tool result
    (or the next user turn).  Called right before that tool result is built."""
    notes = pop_notes(*_owners(agent))
    if not notes:
        return
    from agent.plugin_live_notes import queue_plugin_notice
    for path, text in notes:
        queue_plugin_notice(agent, f"lsp:{path}", text)


__all__ = ["TURN_END_WAIT", "bind_owner", "current_owner", "forward_to_agent", "pop_notes", "post", "restore",
           "settle_for_agent", "settled", "take", "track_pending"]

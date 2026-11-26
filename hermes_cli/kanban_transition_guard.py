"""Fail-closed board transition guard for explicitly listed Kanban boards.

Some boards are coordinated by an owner process with its own admission and control ledger
(the ``fi-plus`` board is the first). On such a board a ``blocked``/``scheduled`` card may
only leave that column — and a ``todo`` card may only become ``ready`` — when a *release
control file* exists on disk that a human wrote for exactly that transition. Nothing else
authorizes it: not a chat message, not a prompt, not an agent's claim that the owner
approved something, and not a read-only audit. The owner side's own cooperative checks are
not the boundary this module defends; this one does not depend on the caller's goodwill.

Where the gate lives
    In the database layer. ``unblock_task`` and ``promote_task`` read and *spend* the
    release inside their own ``BEGIN IMMEDIATE`` transaction, so every caller of those
    functions is covered: the ``kanban_unblock`` model tool, ``hermes kanban unblock`` /
    ``hermes kanban promote``, the dashboard PATCH / bulk / drag-drop status verbs, and any
    in-process caller. ``set_task_status`` (the raw status writer dashboard edits use)
    refuses a transition into ``ready`` on an armed board as well.

    Out of scope by construction: paths that never go through those functions — the
    dispatcher's own auto-promotion, direct SQL against ``kanban.db``, a separate sqlite3
    client, ``hermes kanban repair``, and board import. This gate refuses a transition; it
    is not an access-control boundary, and whoever opens the database file bypasses it.

What a release proves
    ``{"task_id", "request_id", "board", "scope", "requested_at", "requested_by"}`` where
    ``request_id`` is a seven-digit owner request code and ``requested_at`` lies within
    :data:`RELEASE_WINDOW_SECONDS` of the transition; ``board`` must be the armed board and
    ``task_id`` the card. The file is consumed — renamed into ``consumed/`` inside the same
    transaction, before the status flips — so one release authorizes one transition. Config
    (``kanban.transition_guard.boards``) decides *which* boards are armed; it cannot author
    a release, and an older read-only audit names no ``request_id`` so it is never authority.

    A malformed ``boards`` value is not "off": every board is treated as armed and every
    release is refused until an operator fixes the setting.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_cli.kanban_db_connect import _main_db_file

CONFIG_SECTION = "transition_guard"
CONFIG_KEY = "boards"

#: A release names when the human was asked. An older one was written for a different
#: attempt (its owner ledger row has most likely expired), so asking again is the only
#: accurate answer.
RELEASE_WINDOW_SECONDS = 30 * 60
#: Tolerance for a clock that ran slightly ahead when the release was written.
FUTURE_TOLERANCE_SECONDS = 5 * 60

RELEASE_FIELDS = ("task_id", "request_id", "board", "scope", "requested_at", "requested_by")
RELEASE_SUFFIX = ".release.json"
INTERPRETATIONS = ("unblock", "promote")

#: Returned by :func:`armed_boards` for a configured but unusable value: arms every board.
UNREADABLE = ("*",)

#: The owner side issues seven-digit request codes. A four-digit or abbreviated "id" is not
#: one, so a stale document quoting one cannot be promoted into a release.
_REQUEST_ID_RE = re.compile(r"^[0-9]{7}$")

#: Provenance of the call being authorized (which surface attempted the transition).
_TRANSITION_SOURCE: ContextVar[str] = ContextVar("hermes_kanban_transition_source", default="direct")


@dataclass(frozen=True)
class ReleaseRequest:
    """What an armed-board transition must find a release file for."""

    board: str
    task_id: str
    interpretation: str


class BoardTransitionRefused(ValueError):
    """A transition an armed board refuses; the message is the operator-facing reason."""


def armed_boards(config: Optional[dict[str, Any]] = None) -> tuple[str, ...]:
    """Slugs whose ``blocked``/``todo`` exits need a release file; ``()`` when unconfigured.

    Read from ``kanban.transition_guard.boards`` (a list of board slugs); an empty list
    disarms, an absent key leaves every board on normal transitions, and anything else
    unreadable returns :data:`UNREADABLE`.
    """
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    if not isinstance(config, dict):
        return UNREADABLE
    kanban_cfg = config.get("kanban")
    if kanban_cfg is None:
        return ()
    if not isinstance(kanban_cfg, dict):
        return UNREADABLE
    section = kanban_cfg.get(CONFIG_SECTION)
    if section is None:
        return ()
    raw = section.get(CONFIG_KEY) if isinstance(section, dict) else section
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        return UNREADABLE
    slugs: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            return UNREADABLE
        slugs.append(item.strip().lower())
    return tuple(slugs)


def board_is_armed(board: str, armed: tuple[str, ...]) -> bool:
    return "*" in armed or board in armed


def transition_source() -> str:
    """Provenance recorded with a consumed release (which surface asked for it)."""
    return _TRANSITION_SOURCE.get()


class TransitionSource:
    """``with TransitionSource("dashboard"): ...`` — label the call for the audit trail."""

    def __init__(self, label: str):
        self._label = label
        self._token: Any = None

    def __enter__(self) -> TransitionSource:
        self._token = _TRANSITION_SOURCE.set(self._label)
        return self

    def __exit__(self, *exc: object) -> bool:
        _TRANSITION_SOURCE.reset(self._token)
        return False


def _canonical(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return str(Path(path).resolve(strict=False)).lower()
    except OSError:
        return str(path).lower()


def board_of_connection(conn: sqlite3.Connection) -> str:
    """Slug of the board *conn* actually opened, resolved by its file.

    The resolved path outranks env pins: ``HERMES_KANBAN_DB`` collapses every slug onto one
    file, so the path is the identity that cannot be argued with. Falls back to the
    current-board resolution when the file is in-memory or names no known board.
    """
    from hermes_cli import kanban_db as kb

    actual = _canonical(_main_db_file(conn))
    if actual is not None:
        try:
            dirs = [p.name for p in kb.boards_root().iterdir() if p.is_dir()]
        except OSError:
            dirs = []
        for slug in [kb.DEFAULT_BOARD, *dirs]:
            try:
                if _canonical(str(kb.kanban_db_path(board=slug))) == actual:
                    return slug
            except (ValueError, OSError):
                continue
    return kb._slug_or_default(kb.get_current_board())


def releases_dir(board: str) -> Optional[Path]:
    """``<kanban home>/kanban/boards/<slug>/controls/releases/`` — the owner's inbox.

    ``None`` when the shared kanban root cannot be resolved: without a board directory no
    release can be attributed, so transitions stay refused.
    """
    from hermes_cli import kanban_db as kb

    try:
        root = kb.kanban_home()
    except (OSError, RuntimeError, ValueError):
        return None
    return root / "kanban" / "boards" / board / "controls" / "releases" if str(root) else None


def read_release(path: Path) -> Optional[dict[str, str]]:
    """The release at *path*, or ``None`` when it is unreadable or incomplete."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    if not all(isinstance(raw.get(f), str) and raw[f].strip() for f in RELEASE_FIELDS):
        return None  # an empty field asserts nothing; every field is an assertion
    return {f: raw[f].strip() for f in RELEASE_FIELDS}


def _rejection(release: dict[str, str], request: ReleaseRequest, now: float) -> Optional[str]:
    """Why this release does not authorize *request*, or ``None`` when it does."""
    if release["board"].lower() != request.board:
        return f"names board {release['board']!r}, not {request.board!r}"
    if release["task_id"] != request.task_id:
        return f"names task {release['task_id']!r}, not {request.task_id!r}"
    if request.interpretation not in release["scope"].lower().split(":"):
        return f"scope {release['scope']!r} does not grant {request.interpretation} on a blocked card"
    if not _REQUEST_ID_RE.match(release["request_id"]):
        return f"request_id {release['request_id']!r} is not a seven-digit owner request code"
    try:
        requested_at = float(release["requested_at"])
    except ValueError:
        return f"requested_at {release['requested_at']!r} is not unix seconds"
    if requested_at > now + FUTURE_TOLERANCE_SECONDS:
        return f"requested_at is {int(requested_at - now)}s in the future"
    if now - requested_at > RELEASE_WINDOW_SECONDS:
        asked = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(requested_at))
        return (f"was requested at {asked}, over {RELEASE_WINDOW_SECONDS // 60} minutes ago; "
                "that approval has expired — request a new release for this attempt")
    return None


def _scan(directory: Path, request: ReleaseRequest, now: float) -> tuple[Optional[Path], list[str]]:
    """First release in *directory* that authorizes *request*; plus why others did not."""
    try:
        entries = sorted(p for p in directory.iterdir() if p.name.endswith(RELEASE_SUFFIX) and p.is_file())
    except OSError:
        return None, []
    reasons: list[str] = []
    for path in entries:
        release = read_release(path)
        if release is None:
            reasons.append(f"{path.name}: unreadable, or missing {', '.join(RELEASE_FIELDS)}")
            continue
        why = _rejection(release, request, now)
        if why is None:
            return path, reasons
        reasons.append(f"{path.name}: {why}")
    return None, reasons


def refusal_message(request: ReleaseRequest, reasons: list[str]) -> str:
    """The refusal text, naming the exact release a new request has to produce."""
    inbox = releases_dir(request.board)
    where = str(inbox) if inbox is not None else "<kanban home>/kanban/boards/<board>/controls/releases"
    lines = [
        f"{request.board} board: {request.interpretation} of {request.task_id} refused — no valid release.",
        "Leaving blocked/todo on this board needs a one-use release file written by a human for",
        f"this task, board and scope, requested within the last {RELEASE_WINDOW_SECONDS // 60} minutes:",
        f"  {where}/<name>{RELEASE_SUFFIX}",
        f'  {{"task_id": "{request.task_id}", "request_id": "<7-digit owner request code>", '
        f'"board": "{request.board}", "scope": "{request.interpretation}", '
        f'"requested_at": "<unix seconds>", "requested_by": "<owner profile>"}}',
        "A chat message, a prompt, an agent's claim of approval, an audit document, or a config",
        "or env value is not a release. Ask the FI+ owner for a new request and have it record",
        "the release file; an earlier request's approval does not carry over.",
    ]
    if reasons:
        lines.append("Present but not authorizing: " + "; ".join(reasons))
    return "\n".join(lines)


def release_request_for(conn: sqlite3.Connection, task_id: str, interpretation: str) -> Optional[ReleaseRequest]:
    """``ReleaseRequest`` when *conn*'s board is armed, else ``None`` (normal behavior)."""
    board = board_of_connection(conn)
    if not board_is_armed(board, armed_boards()):
        return None
    return ReleaseRequest(board=board, task_id=task_id, interpretation=interpretation)


def refuse_without_release(conn: sqlite3.Connection, request: ReleaseRequest, *, now: Optional[float] = None) -> None:
    """Raise :class:`BoardTransitionRefused` unless a release authorizes *request* (read-only)."""
    directory = releases_dir(request.board)
    if directory is None:
        raise BoardTransitionRefused(refusal_message(request, ["the shared kanban home could not be resolved"]))
    found, reasons = _scan(directory, request, now if now is not None else time.time()) if directory.is_dir() else (None, [])
    if found is None:
        raise BoardTransitionRefused(refusal_message(request, reasons))


def consume_release(conn: sqlite3.Connection, request: ReleaseRequest, *, now: Optional[float] = None) -> Path:
    """Spend the authorizing release, or raise :class:`BoardTransitionRefused`.

    Call inside the transition's own write transaction: the rename into ``consumed/`` and
    the status flip then commit together, so a losing race leaves the release intact for
    the one caller that does change the row — and a second caller finds it spent.
    """
    stamp = now if now is not None else time.time()
    directory = releases_dir(request.board)
    if directory is None:
        raise BoardTransitionRefused(refusal_message(request, ["the shared kanban home could not be resolved"]))
    found, reasons = _scan(directory, request, stamp) if directory.is_dir() else (None, [])
    if found is None:
        raise BoardTransitionRefused(refusal_message(request, reasons))
    release = read_release(found) or {}
    consumed_dir = directory / "consumed"
    try:
        consumed_dir.mkdir(parents=True, exist_ok=True)
        consumed = consumed_dir / f"{int(stamp)}-{found.name}"
        os.replace(found, consumed)
    except OSError as exc:
        raise BoardTransitionRefused(
            f"{request.board} board: could not spend release {found.name} ({exc}); "
            "refusing rather than reusing an unspent release") from exc
    from hermes_cli import kanban_db as kb

    kb._append_event(conn, request.task_id, "transition_released", {
        "board": request.board,
        "request_id": release.get("request_id"),
        "requested_by": release.get("requested_by"),
        "scope": release.get("scope"),
        "source": transition_source(),
        "release": consumed.name,
    })
    return consumed


def pending_releases(board: str) -> list[dict[str, str]]:
    """Well-formed releases currently sitting in *board*'s inbox (for status reporting)."""
    directory = releases_dir(board)
    if directory is None or not directory.is_dir():
        return []
    try:
        entries = sorted(p for p in directory.iterdir() if p.name.endswith(RELEASE_SUFFIX) and p.is_file())
    except OSError:
        return []
    out: list[dict[str, str]] = []
    for path in entries:
        release = read_release(path)
        if release is not None:
            out.append({"file": path.name, **release})
    return out

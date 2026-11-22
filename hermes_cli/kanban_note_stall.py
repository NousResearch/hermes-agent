"""Note-stall detection for RUNNING cards (ADR 0020 §2.2/§3.2/§3.3/§3.6 — build card t_68d5b5a4).

A running worker keeps its claim alive by calling ``kanban_heartbeat`` every ~60 s. That pump
only feeds the *claim* backstop (``release_stale_claims`` at
``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS``), so a worker that is stuck in application logic
while still heartbeating never trips anything, and the board renders "working" forever
(SPECIMEN-14: ~75 min of heartbeats, zero human-readable notes, first signal at ~49 min).

This module adds the missing signal: ``stalled_suspect`` — a first-class ``task_events`` kind
emitted when a ``running`` card has heartbeats but **no milestone note** for a configurable
window. A *milestone note* is a ``task_comments`` row whose body starts with one of
:data:`NOTE_PREFIXES` (ADR §3.6 — the cheap "still working" mitigation the notifier contract
encourages). Arbitrary chatter does not reset the window; heartbeats never do.

Design contracts (ADR §4.1, frozen across the two build lanes):

* Pure DB helpers — no gateway import, so ``hermes-upstream:fresh`` (no gateway) and the
  replay harness can drive it directly against a temp ``kanban.db``.
* One event per window per running episode: emission requires BOTH "silent for >= window"
  and "no earlier ``stalled_suspect`` in this run within ``window`` seconds". Closing the run
  (completed/blocked/reclaimed) starts a fresh episode because the guard is run-scoped.
* Post-a-note reset: the silence anchor is the last milestone note, so a ``NOTE:`` line
  postpones the next ping by a full window.
* Never double-alert with the estate ``stall-sentinel.py`` watchdog (ADR §3.3): a card that
  already carries a ``STALL-SENTINEL:`` receipt for the current episode is skipped, and our
  own receipt prefix ``NOTE-STALL:`` is the token the sentinel's ``IGNORED_COMMENT_RE`` skips.
* Delivery is the existing notification path: the event rides the per-subscription cursor
  (``kanban_db_notify.claim_unseen_events_for_sub``) once the notifier allowlist includes the
  kind. Nothing here talks to Telegram.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from typing import Any, Iterable, Optional

# --- Contract constants (ADR §3.6 / §4.1) ---

#: Comment-body prefixes that count as a milestone note and reset the silence window.
NOTE_PREFIXES: tuple[str, ...] = ("NOTE:", "STILL-WORKING:", "HANDOFF:", "RECHECK:")

#: Receipt prefix this watcher writes next to the event (ADR §3.3 dedup token). Deliberately
#: NOT a :data:`NOTE_PREFIXES` member (``"NOTE-STALL:".startswith("NOTE:")`` is False), so a
#: stall receipt can never silence the very window it is reporting.
STALL_RECEIPT_PREFIX = "NOTE-STALL:"

#: Author stamped on the receipt comment.
STALL_RECEIPT_AUTHOR = "note-stall-watcher"

#: ``task_events.kind`` emitted here.
EVENT_KIND = "stalled_suspect"

#: Shipped default window (ADR §2.2 pins 30 min; operator guidance was 25–30).
DEFAULT_WINDOW_SECONDS = 1800

#: The estate cron's own receipt prefix (read-only marker: its presence means the parked-card
#: watchdog already owns this episode — ADR §3.3 "whichever watchdog fires first wins").
SENTINEL_RECEIPT_RE = re.compile(r"^\s*(?:\*\*)?STALL-SENTINEL:", re.IGNORECASE | re.MULTILINE)


def is_milestone_note(body: Optional[str]) -> bool:
    """True when a comment body is an operator-ratified milestone note (ADR §3.6).

    Leading markdown emphasis/whitespace is tolerated because workers write ``**NOTE:**`` as
    often as ``NOTE:``; the match is still prefix-scoped, so a mid-body mention does not count.
    """
    if not body:
        return False
    head = body.lstrip().lstrip("*_ `>").lstrip()
    return any(head.upper().startswith(p) for p in NOTE_PREFIXES)


def _last_heartbeat_note_at(conn: sqlite3.Connection, task_id: str, since: int) -> Optional[int]:
    """``created_at`` of the newest heartbeat **carrying a note** at/after ``since``.

    ``kanban_heartbeat``'s own tool contract tells workers to pass a one-line plain-language
    note at each real milestone, and the board bot relays exactly those notes to the operator.
    A card doing that is visibly working, so it must not be called stalled — this is the one
    widening of the ADR §2.2 predicate (comments only), taken because the alternative is a
    systematic false positive on the estate's documented heartbeat protocol. Note-less technical
    heartbeats — SPECIMEN-14's shape — still count as silence.
    """
    rows = conn.execute(
        "SELECT payload, created_at FROM task_events WHERE task_id = ? AND kind = 'heartbeat' "
        "AND created_at >= ? ORDER BY id DESC",
        (task_id, int(since)),
    ).fetchall()
    for r in rows:
        raw = r["payload"]
        if not raw:
            continue
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError):
            continue
        note = payload.get("note") if isinstance(payload, dict) else None
        if isinstance(note, str) and note.strip():
            return int(r["created_at"])
    return None


def last_progress_note_at(conn: sqlite3.Connection, task_id: str, since: int) -> Optional[int]:
    """Newest human-visible progress signal since ``since`` (comment note or noted heartbeat)."""
    candidates = [
        value for value in (
            last_milestone_note_at(conn, task_id, since),
            _last_heartbeat_note_at(conn, task_id, since),
        ) if value is not None
    ]
    return max(candidates) if candidates else None


def set_note_stall_window(
    conn: sqlite3.Connection, task_id: str, seconds: int, *, run_id: Optional[int] = None,
) -> bool:
    """Write the per-card ``note_stall_window_seconds`` override (ADR §3.2's task metadata).

    The kernel has no task-level metadata column, so the override lives on the active run's
    ``metadata`` JSON — the same column ``kanban_complete`` writes its handoff into — merged
    rather than replaced so it never clobbers a handoff. This is the writer the replay harness
    uses to compress the window to seconds instead of sleeping, and the lever an operator pulls
    for a card with a legitimately long phase.
    """
    seconds = int(seconds)
    if seconds <= 0:
        raise ValueError("note_stall_window_seconds must be > 0")
    from hermes_cli import kanban_db as _kb

    with _kb.write_txn(conn):
        if run_id is None:
            row = conn.execute(
                "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            run_id = row["current_run_id"] if row else None
        if run_id is None:
            return False
        row = conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (int(run_id),)).fetchone()
        if row is None:
            return False
        try:
            metadata = json.loads(str(row["metadata"])) if row["metadata"] else {}
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["note_stall_window_seconds"] = seconds
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps(metadata, ensure_ascii=False), int(run_id)),
        )
    return True


def _config(board: Optional[str]) -> dict:
    """``kanban.note_stall`` block, ``{}`` when config is unreadable (fail-open: no watcher).

    Read through the same singleton the rest of the kernel uses so a per-test monkeypatch of
    ``hermes_cli.config.load_config`` steers it, exactly like ``_successor_guard_config``.
    """
    try:
        from hermes_cli.config import load_config

        kanban = (load_config() or {}).get("kanban", {})
    # An unreadable config disables the watcher (absent ⇒ no pings) rather than raising inside
    # the dispatcher's dispatch tick, so this boundary is deliberately blind.
    # health: allow BLE001 -- fail-open config read; absence of config means "do not ping"
    except Exception:  # pragma: no cover - config unreadable ⇒ watcher disabled by absence
        return {}
    if not isinstance(kanban, dict):
        return {}
    block = kanban.get("note_stall")
    if not isinstance(block, dict):
        return {}
    return block


def note_stall_enabled(board: Optional[str] = None) -> bool:
    """Whether the watcher runs at all (``kanban.note_stall.enabled``, default True)."""
    return bool(_config(board).get("enabled", True))


def resolve_window_seconds(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    run_id: Optional[int],
    board: Optional[str] = None,
    override: Optional[int] = None,
    cfg: Optional[dict] = None,
) -> int:
    """Window for one card, most-specific first (ADR §3.2).

    1. ``override`` — explicit caller kwarg (harness/replay).
    2. ``task.metadata.note_stall_window_seconds`` — per-card override. The kernel has no
       task-level metadata column, so this lands where per-attempt metadata actually lives:
       the active ``task_runs.metadata`` JSON (what ``kanban_heartbeat``/``kanban_complete``
       writers use). Read defensively: non-integer/<=0 values are ignored.
    3. ``kanban.note_stall.by_board.<slug>`` — per-board override for noisy boards.
    4. ``kanban.note_stall.window_seconds`` — global default.
    5. :data:`DEFAULT_WINDOW_SECONDS`.
    """
    if override is not None:
        try:
            value = int(override)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value

    if run_id is not None:
        try:
            row = conn.execute(
                "SELECT metadata FROM task_runs WHERE id = ?", (int(run_id),)
            ).fetchone()
        except sqlite3.Error:  # pragma: no cover - table always present post-init
            row = None
        raw = row["metadata"] if row is not None and "metadata" in row.keys() else (row[0] if row else None)
        if raw:
            try:
                meta = json.loads(str(raw))
            except (TypeError, ValueError):
                meta = None
            if isinstance(meta, dict):
                candidate = meta.get("note_stall_window_seconds")
                try:
                    value = int(candidate) if candidate is not None else 0
                except (TypeError, ValueError):
                    value = 0
                if value > 0:
                    return value

    cfg = _config(board) if cfg is None else cfg
    by_board = cfg.get("by_board")
    if board and isinstance(by_board, dict):
        candidate = by_board.get(board)
        try:
            value = int(candidate) if candidate is not None else 0
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    try:
        value = int(cfg.get("window_seconds", DEFAULT_WINDOW_SECONDS))
    except (TypeError, ValueError):
        value = DEFAULT_WINDOW_SECONDS
    return value if value > 0 else DEFAULT_WINDOW_SECONDS


def _running_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT t.id, t.title, t.started_at, t.created_at, t.current_run_id, "
        "       t.last_heartbeat_at, r.started_at AS run_started_at "
        "  FROM tasks t "
        "  LEFT JOIN task_runs r ON r.id = t.current_run_id "
        " WHERE t.status = 'running' "
        "   AND t.id NOT LIKE 'board:%:liveness' "
        " ORDER BY t.created_at ASC"
    ).fetchall()


def _episode_start(row: sqlite3.Row) -> int:
    """Start of the current running episode: run start, else task start, else creation."""
    for key in ("run_started_at", "started_at", "created_at"):
        try:
            value = row[key]
        except (IndexError, KeyError):
            value = None
        if value:
            return int(value)
    return 0


def last_milestone_note_at(conn: sqlite3.Connection, task_id: str, since: int) -> Optional[int]:
    """``created_at`` of the newest milestone note at/after ``since`` (None when silent).

    Prefixes are matched in Python rather than SQL ``LIKE`` so the tolerance in
    :func:`is_milestone_note` (markdown emphasis, case) stays the single source of truth; the
    comment set for one episode is small by construction.
    """
    rows = conn.execute(
        "SELECT body, created_at FROM task_comments WHERE task_id = ? AND created_at >= ? "
        "ORDER BY id DESC",
        (task_id, int(since)),
    ).fetchall()
    for r in rows:
        if is_milestone_note(r["body"]):
            return int(r["created_at"])
    return None


def has_sentinel_receipt(conn: sqlite3.Connection, task_id: str, since: int) -> bool:
    """True when the estate ``stall-sentinel`` watchdog already claimed this episode."""
    rows = conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ? AND created_at >= ? ORDER BY id DESC",
        (task_id, int(since)),
    ).fetchall()
    return any(SENTINEL_RECEIPT_RE.search(r["body"] or "") for r in rows)


def _prior_ping_at(conn: sqlite3.Connection, task_id: str, run_id: Optional[int]) -> Optional[int]:
    """Clock of this episode's newest ``stalled_suspect`` (None when never pinged).

    The row's ``created_at`` is the wall clock it was written under; the payload's
    ``evaluated_at`` is the clock the *verdict* used, which a replay driver controls. Taking the
    later of the two keeps the one-ping-per-window rule correct under a synthetic clock without
    changing anything in production, where the two are equal.
    """
    if run_id is None:
        row = conn.execute(
            "SELECT created_at, payload FROM task_events WHERE task_id = ? AND kind = ? "
            "AND run_id IS NULL ORDER BY id DESC LIMIT 1",
            (task_id, EVENT_KIND),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT created_at, payload FROM task_events WHERE task_id = ? AND kind = ? "
            "AND run_id = ? ORDER BY id DESC LIMIT 1",
            (task_id, EVENT_KIND, int(run_id)),
        ).fetchone()
    if row is None:
        return None
    best = int(row["created_at"])
    try:
        payload = json.loads(str(row["payload"])) if row["payload"] else {}
    except (TypeError, ValueError):
        payload = {}
    if isinstance(payload, dict):
        try:
            best = max(best, int(payload.get("evaluated_at") or 0))
        except (TypeError, ValueError):
            pass
    return best


def _ping_count(conn: sqlite3.Connection, task_id: str, run_id: Optional[int]) -> int:
    if run_id is None:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? AND kind = ? "
            "AND run_id IS NULL",
            (task_id, EVENT_KIND),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? AND kind = ? AND run_id = ?",
            (task_id, EVENT_KIND, int(run_id)),
        ).fetchone()
    return int(row["n"]) if row else 0


def check_running_notes(
    conn: sqlite3.Connection,
    *,
    board: Optional[str] = None,
    now: Optional[int] = None,
    window_s: Optional[int] = None,
    emit: bool = True,
    require_heartbeat: bool = True,
    config: Optional[dict] = None,
) -> list[dict]:
    """Scan running cards and open ``stalled_suspect`` episodes (ADR §2.2).

    ``now`` is injectable so tests/replays advance the clock instead of sleeping; the shipped
    default stays 30 min (``kanban.note_stall.window_seconds``) and the per-card
    ``metadata.note_stall_window_seconds`` override is the replay's short-window lever.

    Returns one dict per card that crossed the line, ``{"task_id", "run_id", "window_s",
    "silence_s", "last_note_at", "dedup_key", "emitted"}`` — with ``emitted=False`` when
    ``emit`` is false or the config gate is closed, so callers (and dry-run replays) see the
    verdict either way.
    """
    now = int(time.time()) if now is None else int(now)
    cfg = _config(board) if config is None else config
    verdicts: list[dict] = []
    if not cfg.get("enabled", True):
        return verdicts

    from hermes_cli import kanban_db as _kb

    for row in _running_rows(conn):
        task_id = str(row["id"])
        run_id = row["current_run_id"]
        run_id = int(run_id) if run_id is not None else None
        since = _episode_start(row)
        if not since:
            continue  # no episode anchor ⇒ nothing measurable

        window = resolve_window_seconds(
            conn, task_id=task_id, run_id=run_id, board=board, override=window_s, cfg=cfg
        )
        last_note = last_progress_note_at(conn, task_id, since)
        silence_anchor = last_note if last_note is not None else since
        silence = now - silence_anchor
        if silence < window:
            continue

        hb = row["last_heartbeat_at"]
        hb = int(hb) if hb else None
        if require_heartbeat and not (hb and hb >= since):
            # No heartbeat in this episode ⇒ heartbeat liveness (claim reclaim / the estate
            # NOHB watchdog) owns this card, not note-stall. ADR §3.3 keeps them disjoint.
            continue
        if has_sentinel_receipt(conn, task_id, since):
            continue  # sentinel already owns the episode (ADR §3.3)

        epoch = _ping_count(conn, task_id, run_id) + 1
        prior = _prior_ping_at(conn, task_id, run_id)
        if prior is not None and now - prior < window:
            # Already pinged inside the current window — one event per window per run.
            continue

        dedup_key = f"stalled:{task_id}:{run_id if run_id is not None else 'task'}:{epoch}"
        payload = {
            "dedup_key": dedup_key,
            "running_since": since,
            "window_s": window,
            "window_epoch": epoch,
            "silence_s": silence,
            "last_note_at": last_note,
            "last_heartbeat_at": hb,
            "evaluated_at": now,
            "note_prefixes": list(NOTE_PREFIXES),
        }
        verdicts.append({
            "task_id": task_id, "run_id": run_id, "window_s": window, "silence_s": silence,
            "last_note_at": last_note, "dedup_key": dedup_key, "emitted": bool(emit),
        })
        if not emit:
            continue
        minutes = max(1, round(silence / 60))
        receipt = (
            f"{STALL_RECEIPT_PREFIX} no milestone note for ~{minutes} min while heartbeating "
            f"(window {window}s). Post a line like `NOTE: still working — <what you did>` "
            f"to reset the window; a stuck worker should `kanban_block` instead."
        )
        with _kb.write_txn(conn):
            _kb._append_event(conn, task_id, EVENT_KIND, payload, run_id=run_id)
            _kb._insert_comment(conn, task_id, STALL_RECEIPT_AUTHOR, receipt, now)
    return verdicts


__all__: Iterable[str] = (
    "NOTE_PREFIXES",
    "STALL_RECEIPT_PREFIX",
    "STALL_RECEIPT_AUTHOR",
    "EVENT_KIND",
    "DEFAULT_WINDOW_SECONDS",
    "SENTINEL_RECEIPT_RE",
    "is_milestone_note",
    "note_stall_enabled",
    "resolve_window_seconds",
    "last_milestone_note_at",
    "has_sentinel_receipt",
    "check_running_notes",
)

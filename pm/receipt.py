"""pm sync receipts: the machine-readable surface for venv operations.

Every pm venv sync — startup, plugin install, update rebuild — writes a
receipt with the SAME schema the updater's receipts use
(hermes_cli.update_receipt), into the same
``<HERMES_HOME>/logs/update_receipts/`` dir with a ``kind`` field
separating kinds. One reader (``hermes pm status``, desktop IPC) serves
every surface: a failed venv rebuild is as reportable as a failed
update.

The in-flight receipt lives in a ContextVar, not a module global: the
sync cadence and an ensure bisect can overlap across threads, and a
shared global lets one run's begin/finalize clobber another's record.

Two hazards of ContextVar state are handled explicitly:
- Copied contexts (``contextvars.copy_context()``, asyncio.to_thread)
  share the SAME dict object — so every record_* COPY-ON-WRITES: it
  deep-copies, mutates the copy, and re-sets it in the current context.
  A child task records into its own copy; the parent's receipt is
  untouched.
- A nested ``begin`` in the same context would silently drop the outer
  receipt. ``begin`` therefore returns the ContextVar token; passing it
  to ``finalize(..., token=...)`` restores the OUTER receipt instead of
  discarding it. The ambient no-token begin→finalize stays as-is (the
  existing linear consumers — pm sync, the plugin-check cadence).

Correlation: ``begin`` stamps the receipt with the ambient update
correlation id (derived from the open update receipt's own identity in
the same context). ``finalize`` files the finished receipt under that
id in a per-context map, and the updater embeds only the entry carrying
ITS OWN id (``last_for_update``), never latest.json — a sync that
finished before the update began, a standalone sync, or one from a
concurrent thread cannot be misattributed to it, and a nested update's
sync never displaces the outer update's entry.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
import copy
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_RECEIPT_KEEP = 20

# Scoped current receipt — per-context (threads get their own via
# context isolation), same pattern as agent/context_compressor's pin.
_current: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "pm_receipt_current", default=None
)

# Completed sync receipts, keyed by the update correlation id they were
# begun under (copy-on-written per finalize — copied contexts never share
# a mutated dict). The updater embeds ONLY the entry carrying ITS id, so
# a nested update's sync can never displace the outer update's, and a
# standalone sync (update_id None) is never embedded. Bounded: one entry
# per distinct update id seen in this context (nesting depth).
_completed_by_update: contextvars.ContextVar[Optional[dict[str, dict[str, Any]]]] = (
    contextvars.ContextVar("pm_receipt_completed_by_update", default=None)
)


_worker_update: contextvars.ContextVar[tuple[Optional[str]] | None] = contextvars.ContextVar(
    "pm_worker_update", default=None
)
_last_completed: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "pm_last_completed", default=None
)


@contextmanager
def worker_context(update_id: Optional[str]):
    """Carry correlation across the worker seam without consulting disk state."""
    token = _worker_update.set((update_id,))
    completed = _last_completed.set(None)
    try:
        yield
    finally:
        _last_completed.reset(completed)
        _worker_update.reset(token)


def last_completed() -> Optional[dict[str, Any]]:
    return copy.deepcopy(_last_completed.get())


def accept_worker_receipt(data: Optional[dict[str, Any]], update_id: Optional[str]) -> None:
    if data is None:
        return
    if data.get("update_id") != update_id:
        raise ValueError("PM worker receipt correlation mismatch")
    if update_id:
        completed = dict(_completed_by_update.get() or {})
        completed[update_id] = copy.deepcopy(data)
        _completed_by_update.set(completed)


def _ambient_update_id() -> Optional[str]:
    """The update correlation id in force in this context, or None.

    Lazy import: hermes_cli.update_receipt imports pm.receipt at embed
    time, so this direction must stay function-scoped. Never raises."""
    worker = _worker_update.get()
    if worker is not None:
        return worker[0]
    try:
        from hermes_cli.update_receipt import current_correlation_id

        return current_correlation_id()
    except Exception:
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _receipt_dir() -> Path:
    """The receipts dir — a pure path computation, NO mkdir side
    effect: readers (latest()) must not create state."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "logs" / "update_receipts"


def begin(kind: str) -> contextvars.Token:
    """Start recording a sync. ``kind``: 'sync' | 'update' | 'plugin-check'.
    Returns the ContextVar token — pass it to ``finalize(token=...)`` when
    this begin nests inside an outer begin in the same context, so the
    outer receipt survives the inner finalize.

    The receipt is stamped with the ambient update correlation id (the
    ``hermes update`` this sync belongs to), or None for a standalone
    sync — the id is what makes the updater's embed selective."""
    return _current.set(
        {
            "schema": 1,
            "kind": kind,
            "update_id": _ambient_update_id(),
            "started_at": _utc_now_iso(),
            "steps": [],
            "venv_rebuild": None,
            "plugin_bisect": [],
            "feature_list": None,
            "platform": None,
            "outcome": None,
            "warnings": [],
            "refusal": None,
        }
    )


def _record(mutate) -> None:
    """Copy-on-write record: the ContextVar may be shared with copied
    contexts, so mutate a deep copy and re-set it in THIS context only."""
    current = _current.get()
    if current is None:
        return
    updated = dict(current)
    mutate(updated)
    _current.set(copy.deepcopy(updated))


def record_step(name: str, ok: bool, detail: str = "") -> None:
    _record(lambda r: r.update(steps=[*r["steps"],
        {"name": name, "ok": ok, "detail": detail, "at": _utc_now_iso()}]))


def record_venv_rebuild(ok: bool, reason: str = "") -> None:
    _record(lambda r: r.__setitem__("venv_rebuild", {"ok": ok, "reason": reason}))


def record_bisect(decisions: list[dict]) -> None:
    _record(lambda r: r.__setitem__("plugin_bisect", decisions))


def record_feature_list(extras: Optional[list[str]]) -> None:
    _record(lambda r: r.__setitem__("feature_list", extras))


def record_platform(platform_id: str) -> None:
    _record(lambda r: r.__setitem__("platform", platform_id))


def record_plugin_checks(results: list) -> None:
    """Plugin update-check results (the cadence's receipt section).
    Each item is a plugins_updates.CheckResult.to_json() dict."""
    _record(
        lambda r: r.__setitem__(
            "plugin_checks", [r.to_json() if hasattr(r, "to_json") else r for r in results]
        )
    )


def record_warning(message: str) -> None:
    """Record a warning surfaced to the user during this sync — a warning
    that reached the operator's eyes must reach the receipt too."""
    _record(
        lambda r: r.update(
            warnings=[*r.get("warnings", []),
                      {"message": str(message), "at": _utc_now_iso()}]
        )
    )


def record_refusal(code: str, detail: str = "") -> None:
    """Record WHY this sync refused to act (e.g. the lazy-install policy).
    ``outcome`` stays whatever ``finalize`` is given (refusals finalize as
    ``failed``/``refused``) — this names the refusal class."""
    _record(
        lambda r: r.__setitem__(
            "refusal", {"code": str(code), "detail": str(detail), "at": _utc_now_iso()}
        )
    )


def snapshot() -> Optional[dict[str, Any]]:
    """The in-flight receipt data — for the updater to EMBED its sync
    sections into its own receipt (one schema, one directory). A deep
    COPY: the authoritative in-flight dict is never exposed for the
    caller to mutate."""
    current = _current.get()
    return copy.deepcopy(current) if current is not None else None


def finalize(
    outcome: str, exit_code: int = 0, token: Optional[contextvars.Token] = None
) -> Optional[Path]:
    """Write the receipt (``outcome``: ok | refused | failed | bisected)
    and rotate. Returns its path; None when nothing was begun.

    With ``token`` (from the matching ``begin``): pops only this begin's
    receipt and restores the outer one. Without: pops whatever is
    current (the ambient linear-consumer form)."""
    current = snapshot()
    if current is None:
        return None
    if token is not None:
        _current.reset(token)
    else:
        _current.set(None)
    current["outcome"] = outcome
    current["exit_code"] = exit_code
    current["finished_at"] = _utc_now_iso()
    _last_completed.set(copy.deepcopy(current))
    # Correlation: file this completion under its update id (copy-on-write
    # — a deep-copied entry in a freshly copied map, never a shared dict).
    update_id = current.get("update_id")
    if update_id:
        completed = dict(_completed_by_update.get() or {})
        completed[update_id] = copy.deepcopy(current)
        _completed_by_update.set(completed)
    try:
        path = _write_rotated(current)
    except OSError:
        return None
    return path


def last_for_update(update_id: Optional[str], *, consume: bool = False) -> Optional[dict[str, Any]]:
    """The last sync receipt completed in THIS context under ``update_id``
    — the correlation surface for an invoking update's embed. A sync from
    before this update, a standalone sync, or one from a concurrent
    context cannot be returned; a nested update's sync is filed under the
    nested id and never displaces the outer update's entry. Deep copy;
    None when no matching completion exists."""
    if not update_id:
        return None
    entries = dict(_completed_by_update.get() or {})
    completed = entries.get(update_id)
    if consume:
        entries.pop(update_id, None)
        _completed_by_update.set(entries)
    return copy.deepcopy(completed) if completed is not None else None


def latest() -> Optional[dict[str, Any]]:
    """The newest receipt (any kind) — the reader surface for
    ``hermes pm status`` + the desktop. Pure read: never creates the
    receipts dir."""
    try:
        point = _receipt_dir() / "latest.json"
        if point.is_file():
            return json.loads(point.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return None


def _receipt_name(data: dict[str, Any]) -> str:
    """Unique name for concurrent writers: stamp + pid + full random
    uuid4 (same pid+random convention as the updater's receipts)."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    kind = data.get("kind") or "sync"
    return f"pm_{stamp}-{kind}-{os.getpid()}-{uuid.uuid4().hex}.json"


def _write_rotated(data: dict[str, Any]) -> Path:
    """Use the same stdlib-only atomic writer as PM's installed facts."""
    from hermes_cli.runtime_state import _lock
    from pm.lock import _write

    d = _receipt_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / _receipt_name(data)
    # Concurrent completions share latest.json; serialize its replacement.
    with (d / ".pm-write.lock").open("a+b") as lock:
        _lock(lock.fileno(), wait=True)
        _write(path, data)
        _write(d / "latest.json", data)
        _rotate(d)
    return path


def _rotate(d: Path) -> None:
    """Keep the newest _RECEIPT_KEEP PM receipts. The dir also holds the
    updater's ``update_*.json`` receipts — this rotates ONLY pm's own
    (``pm_*.json``); the updater rotates its own."""
    receipts = sorted(
        (p for p in d.glob("pm_*.json") if p.is_file()),
        key=lambda p: p.name,
    )
    for stale in receipts[:-_RECEIPT_KEEP]:
        try:
            stale.unlink()
        except OSError:
            pass

"""Pre-claim dispatch admission gates: which assignees this home may claim
(``kanban.dispatch_profiles``) and ``kanban.require_model_selection``.

Model selection: when enabled, the dispatcher refuses to claim a ready/review card unless it
carries an explicit task-level model, provider and reasoning effort, so a
worker can never silently start on the assignee profile's default model.
Checked by :func:`hermes_cli.kanban_db_dispatch._dispatch_lane_task` before the
claim, which is the single spawn boundary for gateway ticks, ``hermes kanban
dispatch``, dashboard dispatch, retries/reclaims requeued to ``ready`` and the
review lane. Running cards are never re-read, so in-flight workers are
untouched.

Origin-resident helpers are reached late-bound via ``_kb``.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional

logger = logging.getLogger(__name__)

# Task column -> short name reported in events and CLI output.
_SELECTION_FIELDS = (
    ("model_override", "model"),
    ("provider_override", "provider"),
    ("reasoning_effort", "reasoning"),
)


def require_model_selection_enabled() -> bool:
    """``kanban.require_model_selection`` (default false = upstream behavior).

    An unreadable config is treated as enabled (fail-closed), the same stance
    as ``kanban.dispatch_profiles``.
    """
    try:
        from hermes_cli.config_effective import load_user_config_effective
        kanban = (load_user_config_effective(fail_closed=True) or {}).get("kanban", {})
    except Exception:  # config layer may raise anything; fail closed with traceback
        logger.warning(
            "kanban: could not read kanban.require_model_selection — holding "
            "cards without a task-level model selection", exc_info=True,
        )
        return True
    return isinstance(kanban, Mapping) and bool(kanban.get("require_model_selection", False))


def missing_model_selection(row: Any) -> list[str]:
    """Names of the task-level selections ``row`` lacks; empty = fully selected."""
    keys = set(row.keys())
    return [
        name for column, name in _SELECTION_FIELDS
        if not str((row[column] if column in keys else None) or "").strip()
    ]


def append_event_once(conn: sqlite3.Connection, task_id: str, kind: str, payload: dict) -> None:
    """Append ``kind`` unless it is already the card's latest event with the
    same payload: a persistent hold is written once, not once per tick."""
    from hermes_cli import kanban_db as _kb

    with _kb.write_txn(conn):
        last = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? "
            "ORDER BY created_at DESC, id DESC LIMIT 1", (task_id,)).fetchone()
        if last is None or last["kind"] != kind or last["payload"] != _kb._json_or_null(payload):
            _kb._append_event(conn, task_id, kind, payload)


def hold_unselected(conn: sqlite3.Connection, row: Any, result: Any, *, dry_run: bool) -> bool:
    """Record and return True when ``row`` lacks a task-level selection."""
    missing = missing_model_selection(row)
    if not missing:
        return False
    result.skipped_model_unselected.append((row["id"], missing))
    if not dry_run:
        append_event_once(conn, row["id"], "skipped_model_unselected", {"missing": missing})
    return True


def _profile_exists_fn() -> Optional[Callable[[str], bool]]:
    """``hermes_cli.profiles.profile_exists``, or ``None`` when it cannot be
    imported (local import avoids a cycle; callers fall back to trusting the
    assignee).

    When ``kanban.dispatch_profiles`` is set (#110995) the returned predicate
    additionally requires the assignee to be listed, fail-closed — so a card
    assigned to ``default`` is only claimable by homes that opted into it.
    Foreign assignees land in the existing ``skipped_nonspawnable`` bucket.
    """
    try:
        from hermes_cli.profiles import normalize_profile_name, profile_exists
    except Exception:
        return None
    allowlist = _dispatch_profile_allowlist(normalize_profile_name)
    if allowlist is None:
        return profile_exists

    def _gated(name: str) -> bool:
        try:
            canon = normalize_profile_name(name)
        except ValueError:
            return False
        return canon in allowlist and bool(profile_exists(name))

    return _gated


def _dispatch_profile_allowlist(normalize_profile_name) -> Optional[frozenset]:
    """Per-home claim allowlist ``kanban.dispatch_profiles`` (#110995).

    On a shared board (one ``kanban.db`` mounted across several Hermes homes),
    every home's ``profile_exists`` returns True for ``default`` — the root
    profile every home has — so a card assigned to ``default`` is claimable by
    every home's dispatcher. A home opts out of foreign claims by declaring
    which assignees it may claim::

        kanban:
          dispatch_profiles: ["sage", "researcher"]   # or "sage,researcher"

    Returns ``None`` only when the key is absent from the user config (upstream
    behavior: any existing profile is claimable). A present value is
    fail-closed: an empty list, ``null`` or a bare ``dispatch_profiles:`` claims
    nothing. The user layer is read without the ``DEFAULT_CONFIG`` merge (whose
    ``None`` placeholder would make the key look present in every home), and a
    config read that raises also claims nothing — a corrupt config on a shared
    board must never widen this home's claim scope silently (#113620).
    """
    from hermes_cli import kanban_db as _kb

    try:
        from hermes_cli.config_effective import load_user_config_effective
        kanban = (load_user_config_effective(fail_closed=True) or {}).get("kanban", {})
    except Exception as exc:
        _kb._log.warning(
            "kanban: could not read kanban.dispatch_profiles (%s: %s) — "
            "this home claims no cards until the config is readable",
            type(exc).__name__, exc,
        )
        return frozenset()
    if not isinstance(kanban, Mapping) or "dispatch_profiles" not in kanban:
        return None
    raw = kanban["dispatch_profiles"]
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        _kb._log.warning(
            "kanban: kanban.dispatch_profiles is present but empty — this home "
            "claims no cards; omit the key to allow any existing profile"
        )
        return frozenset()
    names = [str(n) for n in raw] if isinstance(raw, (list, tuple)) else str(raw).split(",")
    allowed = set()
    for n in names:
        try:
            allowed.add(normalize_profile_name(n))
        except ValueError:
            continue
    return frozenset(allowed)


def dispatch_profile_allowlist_summary() -> str:
    """Human-readable resolution of ``kanban.dispatch_profiles`` for this home.

    Surfaced by ``hermes kanban diagnostics`` so an operator on a shared board
    can see what a home believes it may claim (#113620): ``any`` (key absent),
    the sorted allowed names, or ``none (fail-closed: ...)``.
    """
    try:
        from hermes_cli.profiles import normalize_profile_name
    except Exception as exc:
        return f"none (fail-closed: profiles unavailable: {exc})"
    allowlist = _dispatch_profile_allowlist(normalize_profile_name)
    if allowlist is None:
        return "any"
    if allowlist:
        return ", ".join(sorted(allowlist))
    return ("none (fail-closed: kanban.dispatch_profiles is present but names no valid "
            "profile, or the config could not be read — omit the key to allow any)")

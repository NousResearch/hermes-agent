"""Offline-only admission policy for CoStar-targeted Kanban cards.

This module stores only card metadata and receipts.  It never opens a browser,
connects to CDP, or retries a different endpoint.
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

TERRA_MODEL = "gpt-5.6-terra"
CODEX_PROVIDER = "openai-codex"
CONTRACT = {
    "browser": "HermesCDP-CoStar",
    "cdp_url": "http://127.0.0.1:9224",
    "no_fallback": True,
}
_SOL_EXCEPTION_IDS = frozenset({"t_8928b684", "t_b3e7a3ac", "t_e9071b8d"})
_DO_NOT_START_RE = re.compile(r"\bdo\s+not\s+start\s+until\b", re.IGNORECASE)
_DENIAL_RE = re.compile(r"\b(access\s+denied|unusual[_\s-]?activity)\b", re.IGNORECASE)


def is_costar_card(title: str | None, body: str | None) -> bool:
    return "costar" in f"{title or ''}\n{body or ''}".casefold()


def has_costar_contract(conn: sqlite3.Connection, task_id: str) -> bool:
    """Return whether a card carries the durable CoStar creation contract."""
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'created' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None:
        return False
    try:
        payload = json.loads(row["payload"] or "{}")
    except (TypeError, ValueError):
        return False
    return payload.get("costar_contract") == CONTRACT


def is_costar_task(conn: sqlite3.Connection, task_id: str, title: str | None, body: str | None) -> bool:
    """Recognize CoStar work by text or the inherited durable contract."""
    return is_costar_card(title, body) or has_costar_contract(conn, task_id)


def has_costar_parent_lineage(conn: sqlite3.Connection, parents: tuple[str, ...]) -> bool:
    """Return whether any direct or transitive parent establishes a CoStar contract."""
    if not parents:
        return False
    placeholders = ", ".join("?" for _ in parents)
    rows = conn.execute(
        f"""
        WITH RECURSIVE ancestors(id) AS (
            SELECT id FROM tasks WHERE id IN ({placeholders})
            UNION
            SELECT link.parent_id
            FROM task_links AS link
            JOIN ancestors ON link.child_id = ancestors.id
        )
        SELECT task.title, task.body, event.payload
        FROM tasks AS task
        LEFT JOIN task_events AS event
            ON event.task_id = task.id AND event.kind = 'created'
        WHERE task.id IN ancestors
        """,
        parents,
    ).fetchall()
    for row in rows:
        if is_costar_card(row["title"], row["body"]):
            return True
        try:
            payload = json.loads(row["payload"] or "{}")
        except (TypeError, ValueError):
            continue
        if payload.get("costar_contract") == CONTRACT:
            return True
    return False


def is_warden_saved_search(title: str | None, body: str | None) -> bool:
    text = f"{title or ''}\n{body or ''}".casefold()
    return "warden" in text and "saved-search" in text and ("hunt" in text or "refresh" in text)


def has_supervisor_hold(title: str | None, body: str | None) -> bool:
    return bool(_DO_NOT_START_RE.search(f"{title or ''}\n{body or ''}"))


def apply_create_policy(
    *, task_id: str, title: str, body: str | None, model: str | None, provider: str | None,
    initial_status: str, costar_target: bool | None = None,
) -> tuple[str | None, str | None, str, dict[str, Any] | None, dict[str, Any] | None]:
    """Return normalized routing and durable receipt payloads for one new card."""
    if costar_target is None:
        costar_target = is_costar_card(title, body)
    if not costar_target:
        return model, provider, initial_status, None, None
    exception = None
    if model is None:
        model, provider = TERRA_MODEL, CODEX_PROVIDER
    elif task_id not in _SOL_EXCEPTION_IDS and model != TERRA_MODEL:
        exception = {
            "kind": "warden_saved_search" if is_warden_saved_search(title, body) and model == "gpt-5.6-luna" else "user_explicit",
            "model": model,
            "provider": provider,
        }
    hold = None
    if initial_status == "blocked":
        hold = {"reason": "initial_status"}
    elif has_supervisor_hold(title, body):
        initial_status = "blocked"
        hold = {"reason": "do_not_start"}
    return model, provider, initial_status, exception, hold


def persist_creation_receipts(append_event, conn, task_id: str, exception: dict[str, Any] | None, hold: dict[str, Any] | None) -> None:
    """Append optional creation receipts without adding branches to the DB facade."""
    if exception:
        append_event(conn, task_id, "costar_model_exception", exception)
    if hold:
        append_event(conn, task_id, "supervisor_hold", hold)


def supervisor_hold_active(conn: sqlite3.Connection, task_id: str) -> bool:
    """A creation hold may only be released by the explicit unblock transition."""
    row = conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'supervisor_hold' "
        "AND id > COALESCE((SELECT MAX(id) FROM task_events "
        "WHERE task_id = ? AND kind = 'unblocked'), 0) LIMIT 1",
        (task_id, task_id),
    ).fetchone()
    return row is not None


def provider_hold_active(conn: sqlite3.Connection) -> bool:
    """A denial receipt holds all CoStar dispatch until its owning card is unblocked."""
    events = conn.execute(
        "SELECT kind FROM task_events WHERE kind IN ('costar_provider_hold', 'costar_provider_released') "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return bool(events and events["kind"] == "costar_provider_hold")


def provider_hold_owner(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE kind = 'costar_provider_hold' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"] or "{}")
    except (TypeError, ValueError):
        return None
    return payload.get("task_id") if isinstance(payload.get("task_id"), str) else None


def admission_reason(conn: sqlite3.Connection, task_id: str, title: str | None, body: str | None) -> str | None:
    """Return a dispatch refusal without touching browser/CDP state."""
    if not is_costar_task(conn, task_id, title, body):
        return None
    if provider_hold_active(conn):
        return "provider_hold"
    owners = conn.execute(
        "SELECT id, title, body FROM tasks WHERE status = 'running' AND id != ?",
        (task_id,),
    ).fetchall()
    return "single_owner" if any(
        is_costar_task(conn, owner["id"], owner["title"], owner["body"])
        for owner in owners
    ) else None


def is_denial(reason: str | None) -> bool:
    return bool(_DENIAL_RE.search(reason or ""))

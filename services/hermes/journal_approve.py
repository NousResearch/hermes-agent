"""Journal dev-todo approve-to-backlog handler.

Reads dev-category todos from journal_brief.latest.json and handles
one-click promotion to Commander's backlog via POST /api/tickets/create.

Design:
- Idempotent per todo id: a second approval for the same id is a no-op.
- Every attempt (success or duplicate-skip) is written to the audit log.
- The origin:journal label is forwarded in the request; it is never
  created at runtime (must pre-exist in the target repo — AC8).
- Non-2xx / unreachable endpoint → clear error string; never raises.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional, TypedDict

from services.hermes import audit as _audit
from services.hermes.config import get_commander_api_url

_log = logging.getLogger(__name__)

_TICKET_CREATE_PATH = "/api/tickets/create"
_IDEMPOTENCY_FILE = "journal-approvals.json"
_idempotency_lock = threading.Lock()

# Daily-dispatcher date guard: prevents re-posting the same day's [Approve]
# embeds when the Discord gateway restarts mid-day. Separate from
# _IDEMPOTENCY_FILE (which tracks approved *tickets*, not posted *embeds*).
_APPROVALS_POSTED_FILE = "journal-approvals-posted.date"


class ApproveResult(TypedDict):
    success: bool
    duplicate: bool
    error: Optional[str]
    ticket_number: Optional[int]


# ---------------------------------------------------------------------------
# Journal brief loading
# ---------------------------------------------------------------------------

def load_dev_todos(brief_path: str) -> list[dict]:
    """Return dev-category todos from journal_brief.latest.json.

    Returns an empty list when the file is missing, empty, or malformed.
    Never raises.
    """
    try:
        text = Path(brief_path).read_text(encoding="utf-8")
        data = json.loads(text)
        todos = data.get("todos") or []
        return [t for t in todos if t.get("category") == "dev"]
    except (FileNotFoundError, OSError):
        return []
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        _log.warning("journal_brief parse error: %s", exc)
        return []


def map_dev_todos_for_send(todos: list[dict]) -> list[dict]:
    """Map journal-contract dev todos to the {id, title, body} shape that
    ``DiscordAdapter.send_journal_dev_todos`` expects.

    title = todo["content"], falling back to todo["text"] when content is
    empty. body = todo["note"], defaulting to "". Malformed entries (e.g.
    non-dict items) are skipped rather than raising.
    """
    mapped: list[dict] = []
    for todo in todos:
        try:
            todo_id = str(todo.get("id") or "")
            title = str(todo.get("content") or todo.get("text") or "")
            body = str(todo.get("note") or "")
        except AttributeError:
            continue
        mapped.append({"id": todo_id, "title": title, "body": body})
    return mapped


# ---------------------------------------------------------------------------
# Idempotency store
# ---------------------------------------------------------------------------

def _resolve_approval_store() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / _IDEMPOTENCY_FILE


def _load_approved_ids() -> set[str]:
    path = _resolve_approval_store()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("approved_ids") or [])
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return set()


def _save_approved_id(todo_id: str) -> None:
    path = _resolve_approval_store()
    with _idempotency_lock:
        approved = _load_approved_ids()
        approved.add(todo_id)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"approved_ids": sorted(approved)}, separators=(",", ":")),
                encoding="utf-8",
            )
        except Exception as exc:
            _log.warning("Could not persist journal approval for %s: %s", todo_id, exc)


def is_todo_approved(todo_id: str) -> bool:
    """Return True when this todo id has already been approved and a ticket created."""
    return todo_id in _load_approved_ids()


# ---------------------------------------------------------------------------
# Daily-dispatcher date guard
# ---------------------------------------------------------------------------

def _resolve_approvals_posted_file() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / _APPROVALS_POSTED_FILE


def has_posted_approvals_today(today: str) -> bool:
    """Return True when ``today`` (a "YYYY-MM-DD" UTC date string) was
    already recorded as posted.

    Never raises: a missing or unreadable file is treated as "not yet
    posted" so the dispatcher fires normally.
    """
    path = _resolve_approvals_posted_file()
    try:
        last = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return False
    return last == today


def mark_approvals_posted(today: str) -> None:
    """Record ``today`` (a "YYYY-MM-DD" UTC date string) as posted.

    Never raises; a write failure is logged and otherwise ignored — worst
    case the next gateway restart re-posts the same day's embeds, which is
    the existing idempotency-store behavior for the underlying approvals.
    """
    path = _resolve_approvals_posted_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(today, encoding="utf-8")
    except Exception as exc:
        _log.warning("Could not persist journal-approvals posted date: %s", exc)


# ---------------------------------------------------------------------------
# Approval handler
# ---------------------------------------------------------------------------

def handle_journal_approve(
    *,
    todo_id: str,
    title: str,
    body: str,
    project: str,
    user_id: str,
) -> ApproveResult:
    """Promote a dev journal todo to the Commander backlog.

    Returns a dict with:
      success       bool
      duplicate     bool   (True when already approved — no ticket created)
      error         str | None
      ticket_number int | None
    """
    # ── Idempotency check ─────────────────────────────────────────────────
    if is_todo_approved(todo_id):
        _audit.log_journal_approve_attempt(
            todo_id=todo_id,
            actor=user_id,
            outcome="skipped-duplicate",
        )
        return {"success": True, "duplicate": True, "error": None, "ticket_number": None}

    # ── Resolve Commander API URL ──────────────────────────────────────────
    base_url = get_commander_api_url()
    if not base_url:
        return {
            "success": False,
            "duplicate": False,
            "error": (
                "Commander API is not configured. "
                "Set COMMANDER_API_URL to enable journal approvals."
            ),
            "ticket_number": None,
        }

    # ── Build multipart/form POST ──────────────────────────────────────────
    url = base_url.rstrip("/") + _TICKET_CREATE_PATH
    form_fields: list[tuple[str, str]] = [
        ("title", title.strip()),
        ("body", body),
        ("project", project.strip()),
        ("extra_labels", "origin:journal"),
    ]
    encoded = urllib.parse.urlencode(form_fields).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        method="POST",
    )
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    # ── POST ───────────────────────────────────────────────────────────────
    http_outcome: str
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            http_outcome = f"{resp.status} OK"

        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}

        ticket_number: Optional[int] = payload.get("number") or None

        _save_approved_id(todo_id)
        _audit.log_journal_approve_attempt(
            todo_id=todo_id,
            actor=user_id,
            outcome=f"created:{ticket_number}" if ticket_number else "created",
        )
        return {
            "success": True,
            "duplicate": False,
            "error": None,
            "ticket_number": ticket_number,
        }

    except urllib.error.HTTPError as exc:
        http_outcome = f"{exc.code} {exc.reason}"
        _audit.log_journal_approve_attempt(
            todo_id=todo_id,
            actor=user_id,
            outcome=f"error:{exc.code}",
        )
        _log.warning("ticket create returned %d for todo=%s: %s", exc.code, todo_id, exc.reason)
        return {
            "success": False,
            "duplicate": False,
            "error": (
                f"Could not create ticket (HTTP {exc.code} {exc.reason}). "
                "Please try again later."
            ),
            "ticket_number": None,
        }

    except urllib.error.URLError as exc:
        http_outcome = f"Connection Error: {exc.reason}"
        _audit.log_journal_approve_attempt(
            todo_id=todo_id,
            actor=user_id,
            outcome="error:connection",
        )
        _log.warning("Commander API unreachable for todo=%s: %s", todo_id, exc.reason)
        return {
            "success": False,
            "duplicate": False,
            "error": (
                "Could not reach the Commander API — please try again later."
            ),
            "ticket_number": None,
        }

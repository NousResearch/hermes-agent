"""Google Calendar busy-block sync for Torben calendar alignment."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .google_auth import GoogleAccount, load_google_accounts
from .google_evidence import CALENDAR_API_ROOT, _parse_time, _read_token


@dataclass
class CalendarSyncAudit:
    dry_run: bool = False
    google_write_api_calls: int = 0
    external_mutations: int = 0
    created: list[dict[str, Any]] = field(default_factory=list)
    would_create: list[dict[str, Any]] = field(default_factory=list)
    already_exists: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    mutation_cap: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "google_write_api_calls": self.google_write_api_calls,
            "external_mutations": self.external_mutations,
            "created": self.created,
            "would_create": self.would_create,
            "already_exists": self.already_exists,
            "skipped": self.skipped,
            "errors": self.errors,
            "mutation_cap": self.mutation_cap,
        }


def _window(candidate: dict[str, Any]) -> tuple[Any, Any]:
    return _parse_time(candidate.get("start_at")), _parse_time(candidate.get("end_at"))


def _overlaps_window(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_start, first_end = _window(first)
    second_start, second_end = _window(second)
    if not first_start or not first_end or not second_start or not second_end:
        return False
    return first_start < second_end and second_start < first_end


def _alignment_event_id(candidate: dict[str, Any], target_alias: str) -> str:
    evidence = "|".join(str(item) for item in (candidate.get("evidence_ids") or []))
    raw = "|".join(
        [
            "torben-calendar-alignment-v1",
            str(candidate.get("source_account") or ""),
            str(target_alias),
            str(candidate.get("start_at") or ""),
            str(candidate.get("end_at") or ""),
            evidence,
        ]
    )
    return "torben" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _event_body(candidate: dict[str, Any], target_alias: str, event_id: str) -> dict[str, Any]:
    source_account = str(candidate.get("source_account") or "unknown")
    source_hash = hashlib.sha256(
        json.dumps(candidate.get("evidence_ids") or [], sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    body: dict[str, Any] = {
        "id": event_id,
        "summary": "Busy",
        "description": (
            "Torben auto calendar alignment block.\n"
            f"Source account: {source_account}.\n"
            "This is a synthetic busy block; do not treat it as the source meeting."
        ),
        "transparency": "opaque",
        "visibility": "private",
        "reminders": {"useDefault": False},
        "extendedProperties": {
            "private": {
                "torben_alignment": "true",
                "torben_alignment_version": "1",
                "source_account": source_account,
                "target_account": target_alias,
                "source_evidence_hash": source_hash,
            }
        },
    }
    if candidate.get("all_day"):
        start, end = _window(candidate)
        body["start"] = {"date": start.date().isoformat() if start else str(candidate.get("start_at") or "")[:10]}
        body["end"] = {"date": end.date().isoformat() if end else str(candidate.get("end_at") or "")[:10]}
    else:
        body["start"] = {"dateTime": str(candidate.get("start_at") or ""), "timeZone": "UTC"}
        body["end"] = {"dateTime": str(candidate.get("end_at") or ""), "timeZone": "UTC"}
    return body


def _google_insert_event(account: GoogleAccount, token: str, event_body: dict[str, Any]) -> dict[str, Any]:
    calendar_path = urllib.parse.quote("primary", safe="")
    params = urllib.parse.urlencode({"sendUpdates": "none"})
    request = urllib.request.Request(
        f"{CALENDAR_API_ROOT}/calendars/{calendar_path}/events?{params}",
        data=json.dumps(event_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def sync_calendar_alignment_blocks(
    *,
    config_path: str | Path,
    candidates: list[dict[str, Any]],
    dry_run: bool = False,
    max_mutations: int = 60,
) -> dict[str, Any]:
    accounts = {account.alias: account for account in load_google_accounts(config_path).values() if account.enabled}
    audit = CalendarSyncAudit(dry_run=dry_run, mutation_cap=max_mutations)
    created_windows_by_target: dict[str, list[dict[str, Any]]] = {}
    tokens: dict[str, str] = {}

    for candidate in candidates:
        if not candidate.get("start_at") or not candidate.get("end_at"):
            audit.skipped.append({"reason": "missing_time", "candidate": candidate})
            continue
        for target_alias in candidate.get("target_accounts") or []:
            target_alias = str(target_alias)
            target = accounts.get(target_alias)
            if not target:
                audit.errors.append({"target_account": target_alias, "error": "target account not configured"})
                continue
            if any(_overlaps_window(candidate, existing) for existing in created_windows_by_target.get(target_alias, [])):
                audit.skipped.append(
                    {
                        "target_account": target_alias,
                        "reason": "covered_by_batch",
                        "start_at": candidate.get("start_at"),
                        "end_at": candidate.get("end_at"),
                    }
                )
                continue
            event_id = _alignment_event_id(candidate, target_alias)
            item = {
                "target_account": target_alias,
                "source_account": candidate.get("source_account"),
                "event_id": event_id,
                "start_at": candidate.get("start_at"),
                "end_at": candidate.get("end_at"),
            }
            if dry_run:
                audit.would_create.append(item)
                created_windows_by_target.setdefault(target_alias, []).append(candidate)
                continue
            if audit.external_mutations >= max_mutations:
                audit.skipped.append({**item, "reason": "mutation_cap_reached"})
                continue
            token = tokens.get(target_alias)
            if token is None:
                token = _read_token(target)
                tokens[target_alias] = token
            body = _event_body(candidate, target_alias, event_id)
            try:
                created = _google_insert_event(target, token, body)
                audit.google_write_api_calls += 1
            except urllib.error.HTTPError as exc:
                audit.google_write_api_calls += 1
                if exc.code == 409:
                    audit.already_exists.append(item)
                    created_windows_by_target.setdefault(target_alias, []).append(candidate)
                    continue
                audit.errors.append({**item, "error": f"HTTP {exc.code}"})
                continue
            except Exception as exc:  # pragma: no cover - live API/network path
                audit.errors.append({**item, "error": type(exc).__name__})
                continue
            audit.external_mutations += 1
            audit.created.append({**item, "html_link_present": bool(created.get("htmlLink"))})
            created_windows_by_target.setdefault(target_alias, []).append(candidate)

    return audit.to_dict()


def calendar_alignment_sync_needs_attention(sync: dict[str, Any]) -> bool:
    """Return True when a calendar sync result should be delivered to Signal."""
    if sync.get("dry_run") and sync.get("would_create"):
        return True
    if sync.get("errors"):
        return True
    for skipped in sync.get("skipped") or []:
        if skipped.get("reason") == "mutation_cap_reached":
            return True
    return False


def render_calendar_alignment_sync(payload: dict[str, Any]) -> str:
    diagnostics = ((payload.get("source_diagnostics") or {}).get("google") or {})
    audit = diagnostics.get("audit") or {}
    sync = ((payload.get("ea") or {}).get("calendar_alignment_sync") or {})
    created = list(sync.get("created") or [])
    would_create = list(sync.get("would_create") or [])
    errors = list(sync.get("errors") or [])
    already_exists = list(sync.get("already_exists") or [])
    skipped = list(sync.get("skipped") or [])
    dry_run = bool(sync.get("dry_run"))
    action_word = "Would create" if dry_run else "Created"
    action_count = len(would_create) if dry_run else len(created)

    lines = [
        f"Torben calendar alignment: {action_word.lower()} {action_count} busy block(s).",
        (
            f"Checked {len(diagnostics.get('accounts') or [])} account(s), "
            f"{diagnostics.get('calendar_events_collected', 0)} event(s), "
            f"{diagnostics.get('calendar_block_candidates', 0)} drift candidate(s)."
        ),
        (
            f"Google reads: {audit.get('google_read_api_calls', 0)}. "
            f"Google writes: {sync.get('google_write_api_calls', 0)}. "
            f"External mutations: {sync.get('external_mutations', 0)}."
        ),
    ]
    if already_exists:
        lines.append(f"Already existed: {len(already_exists)} block(s).")
    if skipped:
        lines.append(f"Skipped: {len(skipped)} duplicate/capped/unactionable block(s).")
    if errors:
        lines.append(f"Blocked: {len(errors)} write error(s); see JSON artifact.")
        for error in errors[:5]:
            lines.append(f"- {error.get('target_account') or 'unknown'}: {error.get('error') or 'unknown error'}")
    elif action_count == 0 and not already_exists:
        lines.append("No calendar drift needed action.")
    else:
        lines.append("Synthetic blocks are private, reminder-free, and titled Busy.")
    return "\n".join(lines).strip() + "\n"

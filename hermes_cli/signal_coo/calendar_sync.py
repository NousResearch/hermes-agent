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
    google_read_api_calls: int = 0
    google_write_api_calls: int = 0
    external_mutations: int = 0
    created: list[dict[str, Any]] = field(default_factory=list)
    would_create: list[dict[str, Any]] = field(default_factory=list)
    deleted: list[dict[str, Any]] = field(default_factory=list)
    would_delete: list[dict[str, Any]] = field(default_factory=list)
    already_exists: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    circuit_breakers: list[dict[str, Any]] = field(default_factory=list)
    mutation_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    mutation_cap: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "google_read_api_calls": self.google_read_api_calls,
            "google_write_api_calls": self.google_write_api_calls,
            "external_mutations": self.external_mutations,
            "created": self.created,
            "would_create": self.would_create,
            "deleted": self.deleted,
            "would_delete": self.would_delete,
            "already_exists": self.already_exists,
            "skipped": self.skipped,
            "circuit_breakers": self.circuit_breakers,
            "mutation_records": self.mutation_records,
            "errors": self.errors,
            "mutation_cap": self.mutation_cap,
        }


def _window(candidate: dict[str, Any]) -> tuple[Any, Any]:
    return _parse_time(candidate.get("start_at")), _parse_time(candidate.get("end_at"))


def _normalized_time_text(value: Any) -> str:
    if value is None:
        return ""
    raw = value.isoformat() if hasattr(value, "isoformat") else str(value)
    parsed = _parse_time(raw)
    if parsed is None:
        return raw
    return parsed.isoformat().replace("+00:00", "Z")


def _event_key(*, target_alias: str, start_at: Any, end_at: Any) -> str:
    return "|".join([target_alias, _normalized_time_text(start_at), _normalized_time_text(end_at)])


def _mutation_record(action: str, item: dict[str, Any], *, external_mutation: bool) -> dict[str, Any]:
    return {
        "action": action,
        "event_key": item.get("event_key"),
        "event_id": item.get("event_id"),
        "target_account": item.get("target_account"),
        "source_account": item.get("source_account"),
        "start_at": item.get("start_at"),
        "end_at": item.get("end_at"),
        "external_mutation": external_mutation,
    }


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
            _normalized_time_text(candidate.get("start_at")),
            _normalized_time_text(candidate.get("end_at")),
            evidence,
        ]
    )
    return "torben" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _is_torben_alignment_block(event: dict[str, Any]) -> bool:
    private = ((event.get("extended_properties") or event.get("extendedProperties") or {}).get("private") or {})
    if str(private.get("torben_alignment") or "").lower() == "true":
        return True
    return str(event.get("id") or event.get("event_id") or "").startswith("torben")


def _candidate_from_event(event: dict[str, Any], target_alias: str) -> dict[str, Any]:
    return {
        "source_account": event.get("account_alias"),
        "source_calendar": event.get("calendar_summary"),
        "target_accounts": [target_alias],
        "already_blocked_accounts": [],
        "summary": str(event.get("summary") or "Busy"),
        "start_at": event.get("start_at"),
        "end_at": event.get("end_at"),
        "all_day": bool(event.get("all_day")),
        "evidence_ids": list(event.get("evidence_ids") or []),
    }


def _desired_alignment_ids_from_events(
    *,
    source_events: list[dict[str, Any]],
    accounts: dict[str, GoogleAccount],
) -> dict[str, set[str]]:
    real_events_by_account: dict[str, list[dict[str, Any]]] = {alias: [] for alias in accounts}
    for event in source_events:
        alias = str(event.get("account_alias") or "")
        if not alias or alias not in accounts:
            continue
        if str(event.get("transparency") or "opaque") == "transparent":
            continue
        if _is_torben_alignment_block(event):
            continue
        if not event.get("start_at") or not event.get("end_at"):
            continue
        real_events_by_account.setdefault(alias, []).append(event)

    desired: dict[str, set[str]] = {alias: set() for alias in accounts}
    desired_windows_by_target: dict[str, list[dict[str, Any]]] = {alias: [] for alias in accounts}
    for event in sorted(
        (item for items in real_events_by_account.values() for item in items),
        key=lambda item: str(item.get("start_at") or ""),
    ):
        source_alias = str(event.get("account_alias") or "")
        for target_alias in accounts:
            if target_alias == source_alias:
                continue
            if any(_overlaps_window(event, existing) for existing in real_events_by_account.get(target_alias, [])):
                continue
            if any(_overlaps_window(event, existing) for existing in desired_windows_by_target.get(target_alias, [])):
                continue
            candidate = _candidate_from_event(event, target_alias)
            desired[target_alias].add(_alignment_event_id(candidate, target_alias))
            desired_windows_by_target.setdefault(target_alias, []).append(candidate)
    return desired


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
        body["start"] = {"dateTime": _normalized_time_text(candidate.get("start_at")), "timeZone": "UTC"}
        body["end"] = {"dateTime": _normalized_time_text(candidate.get("end_at")), "timeZone": "UTC"}
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


def _google_list_alignment_events(
    account: GoogleAccount,
    token: str,
    *,
    time_min: str,
    time_max: str,
) -> tuple[list[dict[str, Any]], int]:
    calendar_path = urllib.parse.quote("primary", safe="")
    page_token: str | None = None
    events: list[dict[str, Any]] = []
    calls = 0
    while True:
        params = {
            "timeMin": time_min,
            "timeMax": time_max,
            "singleEvents": "true",
            "showDeleted": "false",
            "maxResults": "2500",
            "privateExtendedProperty": "torben_alignment=true",
        }
        if page_token:
            params["pageToken"] = page_token
        url = f"{CALENDAR_API_ROOT}/calendars/{calendar_path}/events?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="GET")
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        calls += 1
        events.extend(payload.get("items") or [])
        page_token = str(payload.get("nextPageToken") or "")
        if not page_token:
            break
    return events, calls


def _google_delete_event(account: GoogleAccount, token: str, event_id: str) -> None:
    calendar_path = urllib.parse.quote("primary", safe="")
    event_path = urllib.parse.quote(event_id, safe="")
    params = urllib.parse.urlencode({"sendUpdates": "none"})
    request = urllib.request.Request(
        f"{CALENDAR_API_ROOT}/calendars/{calendar_path}/events/{event_path}?{params}",
        headers={"Authorization": f"Bearer {token}"},
        method="DELETE",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()


def sync_calendar_alignment_blocks(
    *,
    config_path: str | Path,
    candidates: list[dict[str, Any]],
    source_events: list[dict[str, Any]] | None = None,
    cleanup_stale: bool = False,
    cleanup_window_start: str | None = None,
    cleanup_window_end: str | None = None,
    dry_run: bool = False,
    max_mutations: int = 20,
) -> dict[str, Any]:
    accounts = {account.alias: account for account in load_google_accounts(config_path).values() if account.enabled}
    max_mutations = max(0, int(max_mutations))
    audit = CalendarSyncAudit(dry_run=dry_run, mutation_cap=max_mutations)
    created_windows_by_target: dict[str, list[dict[str, Any]]] = {}
    tokens: dict[str, str] = {}
    created_event_ids_by_key: dict[str, list[str]] = {}
    desired_ids_by_target = (
        _desired_alignment_ids_from_events(source_events=source_events, accounts=accounts)
        if source_events is not None
        else {alias: set() for alias in accounts}
    )

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
            desired_ids_by_target.setdefault(target_alias, set()).add(event_id)
            start_at = _normalized_time_text(candidate.get("start_at"))
            end_at = _normalized_time_text(candidate.get("end_at"))
            item = {
                "target_account": target_alias,
                "source_account": candidate.get("source_account"),
                "event_id": event_id,
                "event_key": _event_key(target_alias=target_alias, start_at=start_at, end_at=end_at),
                "start_at": start_at,
                "end_at": end_at,
            }
            if dry_run:
                audit.would_create.append(item)
                created_event_ids_by_key.setdefault(str(item["event_key"]), []).append(event_id)
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
            created_item = {**item, "html_link_present": bool(created.get("htmlLink"))}
            audit.created.append(created_item)
            audit.mutation_records.append(_mutation_record("create", created_item, external_mutation=True))
            created_event_ids_by_key.setdefault(str(item["event_key"]), []).append(event_id)
            created_windows_by_target.setdefault(target_alias, []).append(candidate)

    if cleanup_stale:
        if not cleanup_window_start or not cleanup_window_end:
            audit.skipped.append({"reason": "missing_cleanup_window"})
        else:
            for target_alias, target in accounts.items():
                token = tokens.get(target_alias)
                if token is None:
                    token = _read_token(target)
                    tokens[target_alias] = token
                try:
                    alignment_events, read_calls = _google_list_alignment_events(
                        target,
                        token,
                        time_min=cleanup_window_start,
                        time_max=cleanup_window_end,
                    )
                    audit.google_read_api_calls += read_calls
                except Exception as exc:  # pragma: no cover - live API/network path
                    audit.errors.append({"target_account": target_alias, "error": type(exc).__name__})
                    continue

                desired_ids = desired_ids_by_target.get(target_alias, set())
                for event in alignment_events:
                    event_id = str(event.get("id") or "")
                    if not event_id or event_id in desired_ids:
                        continue
                    start = (event.get("start") or {}).get("dateTime") or (event.get("start") or {}).get("date")
                    end = (event.get("end") or {}).get("dateTime") or (event.get("end") or {}).get("date")
                    start_at = _normalized_time_text(start)
                    end_at = _normalized_time_text(end)
                    item = {
                        "target_account": target_alias,
                        "event_id": event_id,
                        "event_key": _event_key(target_alias=target_alias, start_at=start_at, end_at=end_at),
                        "start_at": start_at,
                        "end_at": end_at,
                    }
                    created_event_ids = created_event_ids_by_key.get(str(item["event_key"])) or []
                    if created_event_ids:
                        breaker = {
                            **item,
                            "reason": "same_event_key_create_delete",
                            "threshold": 2,
                            "created_event_ids": sorted(created_event_ids),
                        }
                        audit.circuit_breakers.append(breaker)
                        audit.skipped.append({**item, "reason": "circuit_breaker_same_event_key_create_delete"})
                        audit.mutation_records.append(
                            _mutation_record("circuit_breaker", breaker, external_mutation=False)
                        )
                        continue
                    if dry_run:
                        audit.would_delete.append(item)
                        continue
                    if audit.external_mutations >= max_mutations:
                        audit.skipped.append({**item, "reason": "mutation_cap_reached"})
                        continue
                    try:
                        _google_delete_event(target, token, event_id)
                        audit.google_write_api_calls += 1
                    except urllib.error.HTTPError as exc:
                        audit.google_write_api_calls += 1
                        audit.errors.append({**item, "error": f"HTTP {exc.code}"})
                        continue
                    except Exception as exc:  # pragma: no cover - live API/network path
                        audit.errors.append({**item, "error": type(exc).__name__})
                        continue
                    audit.external_mutations += 1
                    audit.deleted.append(item)
                    audit.mutation_records.append(_mutation_record("delete", item, external_mutation=True))

    return audit.to_dict()


def calendar_alignment_sync_needs_attention(sync: dict[str, Any]) -> bool:
    """Return True when a calendar sync result should be delivered to Signal."""
    if sync.get("dry_run") and (sync.get("would_create") or sync.get("would_delete")):
        return True
    if sync.get("errors"):
        return True
    if sync.get("circuit_breakers"):
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
    deleted = list(sync.get("deleted") or [])
    would_delete = list(sync.get("would_delete") or [])
    circuit_breakers = list(sync.get("circuit_breakers") or [])
    dry_run = bool(sync.get("dry_run"))
    create_count = len(would_create) if dry_run else len(created)
    delete_count = len(would_delete) if dry_run else len(deleted)
    action_line = (
        f"would create {create_count} busy block(s) and delete {delete_count} stale block(s)"
        if dry_run
        else f"created {create_count} busy block(s) and deleted {delete_count} stale block(s)"
    )

    lines = [
        f"Torben calendar alignment: {action_line}.",
        (
            f"Checked {len(diagnostics.get('accounts') or [])} account(s), "
            f"{diagnostics.get('calendar_events_collected', 0)} event(s), "
            f"{diagnostics.get('calendar_block_candidates', 0)} drift candidate(s)."
        ),
        (
            f"Google reads: {audit.get('google_read_api_calls', 0)}. "
            f"Sync reads: {sync.get('google_read_api_calls', 0)}. "
            f"Google writes: {sync.get('google_write_api_calls', 0)}. "
            f"External mutations: {sync.get('external_mutations', 0)}."
        ),
    ]
    if already_exists:
        lines.append(f"Already existed: {len(already_exists)} block(s).")
    if skipped:
        lines.append(f"Skipped: {len(skipped)} duplicate/capped/unactionable block(s).")
    if circuit_breakers:
        lines.append(f"Circuit breaker: {len(circuit_breakers)} same-window create/delete loop(s) suppressed.")
    if errors:
        lines.append(f"Blocked: {len(errors)} write error(s); see JSON artifact.")
        for error in errors[:5]:
            lines.append(f"- {error.get('target_account') or 'unknown'}: {error.get('error') or 'unknown error'}")
    elif circuit_breakers:
        lines.append("Calendar sync needs review before more same-window mutations are attempted.")
    elif create_count == 0 and delete_count == 0 and not already_exists:
        lines.append("No calendar drift needed action.")
    else:
        lines.append("Synthetic blocks are private, reminder-free, and titled Busy.")
    return "\n".join(lines).strip() + "\n"

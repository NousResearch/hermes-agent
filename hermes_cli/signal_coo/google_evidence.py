"""Bounded Google evidence collection for Torben EA canaries."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .google_auth import GoogleAccount, check_account, load_google_accounts
from .morning_brief import build_morning_brief_scope

GMAIL_API_ROOT = "https://gmail.googleapis.com/gmail/v1/users/me"
CALENDAR_API_ROOT = "https://www.googleapis.com/calendar/v3"


@dataclass
class GoogleEvidenceAudit:
    generated_at: str
    accounts_checked: list[str] = field(default_factory=list)
    google_read_api_calls: int = 0
    google_write_api_calls: int = 0
    external_mutations: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "accounts_checked": self.accounts_checked,
            "google_read_api_calls": self.google_read_api_calls,
            "google_write_api_calls": self.google_write_api_calls,
            "external_mutations": self.external_mutations,
            "warnings": self.warnings,
        }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_token(account: GoogleAccount) -> str:
    status = check_account(account)
    if not status.status.startswith("authenticated"):
        raise RuntimeError(f"Google account {account.alias} is not authenticated: {status.status} {status.reason or ''}")
    payload = json.loads(account.token_path.read_text(encoding="utf-8"))
    token = str(payload.get("token") or "")
    if not token:
        raise RuntimeError(f"Google account {account.alias} has no access token after refresh.")
    return token


def _google_get(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _calendar_sources(
    account: GoogleAccount,
    token: str,
    *,
    include_secondary_calendars: bool,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    primary = {
        "id": "primary",
        "summary": "Primary",
        "primary": True,
        "access_role": "owner",
    }
    if not include_secondary_calendars:
        return [primary], 0, []

    try:
        payload = _google_get(f"{CALENDAR_API_ROOT}/users/me/calendarList", token)
    except Exception as exc:  # pragma: no cover - live API fallback path
        return [primary], 1, [f"{account.alias}: calendarList unavailable; fell back to primary ({exc})"]

    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload.get("items") or []:
        calendar_id = str(item.get("id") or "").strip()
        if not calendar_id or calendar_id in seen:
            continue
        if item.get("hidden"):
            continue
        access_role = str(item.get("accessRole") or "")
        is_primary = bool(item.get("primary"))
        if not is_primary and access_role not in {"owner", "writer"}:
            continue
        seen.add(calendar_id)
        sources.append(
            {
                "id": calendar_id,
                "summary": str(item.get("summary") or ("Primary" if is_primary else "Calendar")),
                "primary": is_primary,
                "access_role": access_role or ("owner" if is_primary else "unknown"),
            }
        )

    if not sources:
        sources = [primary]
    elif "primary" not in seen and not any(source.get("primary") for source in sources):
        sources.insert(0, primary)
    return sources, 1, []


def _calendar_events(
    account: GoogleAccount,
    token: str,
    *,
    calendar_source: dict[str, Any],
    now: datetime,
    days: int,
    max_results: int,
) -> tuple[list[dict[str, Any]], int]:
    calendar_id = str(calendar_source.get("id") or "primary")
    params = urllib.parse.urlencode(
        {
            "timeMin": _iso(now),
            "timeMax": _iso(now + timedelta(days=days)),
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": str(max_results),
        }
    )
    calendar_path = urllib.parse.quote(calendar_id, safe="")
    url = f"{CALENDAR_API_ROOT}/calendars/{calendar_path}/events?{params}"
    payload = _google_get(url, token)
    events: list[dict[str, Any]] = []
    for item in payload.get("items") or []:
        if item.get("status") == "cancelled":
            continue
        start_raw = (item.get("start") or {}).get("dateTime") or (item.get("start") or {}).get("date")
        end_raw = (item.get("end") or {}).get("dateTime") or (item.get("end") or {}).get("date")
        start = _parse_time(start_raw)
        end = _parse_time(end_raw)
        if not start or not end:
            continue
        attendees = [
            {
                "email": str(attendee.get("email") or ""),
                "display_name": str(attendee.get("displayName") or ""),
                "response_status": str(attendee.get("responseStatus") or ""),
                "organizer": bool(attendee.get("organizer")),
            }
            for attendee in (item.get("attendees") or [])[:10]
        ]
        description = re.sub(r"\s+", " ", str(item.get("description") or "")).strip()
        events.append(
            {
                "account_alias": account.alias,
                "account_email": account.email,
                "calendar_id": calendar_id,
                "calendar_summary": str(calendar_source.get("summary") or "Primary"),
                "calendar_primary": bool(calendar_source.get("primary")),
                "calendar_access_role": str(calendar_source.get("access_role") or "unknown"),
                "title": str(item.get("summary") or "Busy"),
                "person": str(item.get("summary") or "Busy"),
                "organization": account.email,
                "summary": str(item.get("summary") or "Busy"),
                "description": description[:1000],
                "start_at": _iso(start),
                "end_at": _iso(end),
                "all_day": "date" in (item.get("start") or {}),
                "goal": "to protect the calendar commitment and arrive prepared",
                "last_conversation": "calendar context only; no prior conversation captured",
                "recommended_line": "confirm the next decision and any blocker before the call ends",
                "transparency": str(item.get("transparency") or "opaque"),
                "attendees_count": len(item.get("attendees") or []),
                "attendees": attendees,
                "organizer": item.get("organizer") or {},
                "hangout_link_present": bool(item.get("hangoutLink") or item.get("conferenceData")),
                "extended_properties": item.get("extendedProperties") or {},
                "evidence_ids": [f"google-calendar:{account.alias}:{calendar_id}:{item.get('id', 'unknown')}"],
            }
        )
    return events, 1


def _message_headers(headers: list[dict[str, Any]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for header in headers:
        name = str(header.get("name") or "").lower()
        if name in {"from", "subject", "date"}:
            values[name] = str(header.get("value") or "")
    return values


def _staged_response_detail(*, sender: str, subject: str, snippet: str) -> str:
    clean_snippet = re.sub(r"\s+", " ", snippet).strip()
    if clean_snippet:
        clean_snippet = clean_snippet[:180].rstrip()
        return (
            f"Acknowledge the thread from {sender}, respond to the concrete ask in "
            f"\"{subject}\", and use this source context: {clean_snippet}"
        )
    return f"Acknowledge the thread from {sender} and answer or ask for the next step on \"{subject}\"."


def _gmail_candidates(account: GoogleAccount, token: str, *, max_results: int) -> tuple[list[dict[str, Any]], int]:
    if max_results <= 0:
        return [], 0
    query = "newer_than:2d -category:promotions -category:social"
    params = urllib.parse.urlencode({"q": query, "maxResults": str(max_results)})
    list_url = f"{GMAIL_API_ROOT}/messages?{params}"
    listed = _google_get(list_url, token)
    read_calls = 1
    candidates: list[dict[str, Any]] = []
    for item in listed.get("messages") or []:
        message_id = str(item.get("id") or "")
        if not message_id:
            continue
        get_params = urllib.parse.urlencode(
            {
                "format": "metadata",
                "metadataHeaders": ["From", "Subject", "Date"],
            },
            doseq=True,
        )
        payload = _google_get(f"{GMAIL_API_ROOT}/messages/{message_id}?{get_params}", token)
        read_calls += 1
        headers = _message_headers(((payload.get("payload") or {}).get("headers") or []))
        label_ids = [str(label) for label in (payload.get("labelIds") or [])]
        sender = headers.get("from", "unknown sender")
        subject = headers.get("subject", "(no subject)")
        combined = f"{sender} {subject} {payload.get('snippet') or ''}".lower()
        noisy_terms = (
            "no-reply",
            "noreply",
            "donotreply",
            "do-not-reply",
            "one time passcode",
            "one-time passcode",
            "verification code",
            "security code",
            "authentication code",
            "login code",
        )
        if any(term in combined for term in noisy_terms):
            continue
        candidates.append(
            {
                "account_alias": account.alias,
                "account_email": account.email,
                "sender": sender,
                "subject": subject,
                "date": headers.get("date", ""),
                "snippet": str(payload.get("snippet") or ""),
                "context_line": (
                    f"Recent message in {account.email} from {sender} about "
                    f"\"{subject}\"."
                ),
                "staged_response_detail": _staged_response_detail(
                    sender=sender,
                    subject=subject,
                    snippet=str(payload.get("snippet") or ""),
                ),
                "labels": label_ids,
                "evidence_ids": [f"gmail:{account.alias}:{message_id}"],
            }
        )
    return candidates, read_calls


def _overlaps(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_start = _parse_time(first.get("start_at"))
    first_end = _parse_time(first.get("end_at"))
    second_start = _parse_time(second.get("start_at"))
    second_end = _parse_time(second.get("end_at"))
    if not first_start or not first_end or not second_start or not second_end:
        return False
    return first_start < second_end and second_start < first_end


def _is_torben_alignment_block(event: dict[str, Any]) -> bool:
    summary = str(event.get("summary") or "").lower()
    return "torben" in summary and ("block" in summary or "busy" in summary or "alignment" in summary)


def build_calendar_block_candidates(
    events: list[dict[str, Any]],
    accounts: list[GoogleAccount],
    *,
    max_candidates: int | None = 3,
) -> list[dict[str, Any]]:
    by_account: dict[str, list[dict[str, Any]]] = {account.alias: [] for account in accounts}
    for event in events:
        by_account.setdefault(str(event.get("account_alias") or ""), []).append(event)

    candidates: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: str(item.get("start_at") or "")):
        if str(event.get("transparency") or "opaque") == "transparent":
            continue
        if _is_torben_alignment_block(event):
            continue
        source_alias = str(event.get("account_alias") or "")
        target_aliases: list[str] = []
        already_blocked_aliases: list[str] = []
        for account in accounts:
            if account.alias == source_alias:
                continue
            if any(_overlaps(event, existing) for existing in by_account.get(account.alias, [])):
                already_blocked_aliases.append(account.alias)
            else:
                target_aliases.append(account.alias)
        if target_aliases:
            candidates.append(
                {
                    "source_account": source_alias,
                    "source_calendar": event.get("calendar_summary"),
                    "target_accounts": target_aliases,
                    "already_blocked_accounts": already_blocked_aliases,
                    "summary": str(event.get("summary") or "Busy"),
                    "start_at": event.get("start_at"),
                    "end_at": event.get("end_at"),
                    "all_day": bool(event.get("all_day")),
                    "evidence_ids": list(event.get("evidence_ids") or []),
                }
            )
        if max_candidates is not None and len(candidates) >= max_candidates:
            break
    return candidates


def collect_google_ea_evidence(
    *,
    config_path: str | Path,
    now: datetime | None = None,
    days: int = 2,
    max_calendar_events: int = 8,
    max_email_messages: int = 5,
    max_calendar_block_candidates: int | None = 3,
    include_secondary_calendars: bool = False,
) -> dict[str, Any]:
    now = (now or utc_now()).astimezone(timezone.utc)
    accounts = [account for account in load_google_accounts(config_path).values() if account.enabled]
    audit = GoogleEvidenceAudit(generated_at=_iso(now), accounts_checked=[account.alias for account in accounts])
    all_events: list[dict[str, Any]] = []
    all_messages: list[dict[str, Any]] = []

    for account in accounts:
        token = _read_token(account)
        sources, read_calls, warnings = _calendar_sources(
            account,
            token,
            include_secondary_calendars=include_secondary_calendars,
        )
        audit.google_read_api_calls += read_calls
        audit.warnings.extend(warnings)
        for source in sources:
            events, read_calls = _calendar_events(
                account,
                token,
                calendar_source=source,
                now=now,
                days=days,
                max_results=max_calendar_events,
            )
            audit.google_read_api_calls += read_calls
            all_events.extend(events)
        messages, read_calls = _gmail_candidates(account, token, max_results=max_email_messages)
        audit.google_read_api_calls += read_calls
        all_messages.extend(messages)

    all_events.sort(key=lambda item: str(item.get("start_at") or ""))
    calendar_block_candidates = build_calendar_block_candidates(
        all_events,
        accounts,
        max_candidates=max_calendar_block_candidates,
    )
    ea_payload = {
        "calendar_events": all_events,
        "email_reply_candidates": all_messages,
        "calendar_block_candidates": calendar_block_candidates,
        "open_loops": [],
        "calendar_alignment_policy": {
            "lookahead_days": days,
            "mode": "auto_private_busy_block",
            "rule": (
                "When a non-transparent event appears on one enabled account, "
                "create private Busy blocks for enabled accounts that do not already have an overlapping event."
            ),
            "mutation_boundary": (
                "Only synthetic private Busy blocks are created; source events are never edited or deleted."
            ),
        },
    }
    ea_payload["morning_brief"] = build_morning_brief_scope(ea_payload, now=now)
    return {
        "ea": ea_payload,
        "source_diagnostics": {
            "google": {
                "accounts": [
                    {
                        "alias": account.alias,
                        "email": account.email,
                        "role": account.role,
                    }
                    for account in accounts
                ],
                "secondary_calendar_collection": bool(include_secondary_calendars),
                "calendar_events_collected": len(all_events),
                "email_messages_collected": len(all_messages),
                "calendar_block_candidates": len(calendar_block_candidates),
                "audit": audit.to_dict(),
            }
        },
    }


def write_json_artifact(payload: dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path

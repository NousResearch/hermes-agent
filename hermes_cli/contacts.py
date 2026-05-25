"""CLI helpers for Hermes data-isolation contacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No contacts found.")
        return
    headers = ("identity_key", "level", "display_name", "source")
    widths = {
        header: max(len(header), *(len(str(row.get(header, "") or "")) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row.get(header, "") or "").ljust(widths[header]) for header in headers))


def _parse_tools(value: str) -> list[str]:
    tools = [item.strip() for item in (value or "").split(",") if item.strip()]
    if not tools:
        raise SystemExit("At least one tool is required.")
    return tools


def _normalize_expiry(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SystemExit("--expires-at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def contacts_command(args) -> None:
    from gateway import data_isolation

    action = getattr(args, "contacts_action", None) or "list"
    if action == "list":
        _print_table(data_isolation.list_contacts(include_profiles=not getattr(args, "configured_only", False)))
        return

    if action == "set-level":
        contact = data_isolation.set_contact_level(
            args.identity_key,
            args.level,
            display_name=getattr(args, "display_name", "") or "",
        )
        print(f"Set {args.identity_key} to {contact['level']}.")
        return

    if action == "grant":
        grant = data_isolation.add_project_grant(
            identity_key=args.identity_key,
            tools=_parse_tools(args.tools),
            path=str(Path(args.path).expanduser()) if args.path else None,
            expires_at=_normalize_expiry(args.expires_at),
        )
        scope = f" under {grant['path']}" if grant.get("path") else ""
        expiry = f" until {grant['expires_at']}" if grant.get("expires_at") else ""
        print(f"Granted {', '.join(grant['tools'])} to {args.identity_key}{scope}{expiry}.")
        return

    raise SystemExit(f"Unknown contacts action: {action}")

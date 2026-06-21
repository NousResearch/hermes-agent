from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from typing import Any

from agent.evolution_log import (
    clear_older_than,
    ensure_evolution_dir,
    filter_events,
    get_events_path,
    read_events,
    resolve_event_id,
)
from hermes_cli.config import load_config, save_config


def register_cli(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(func=evolution_command)
    sub = parser.add_subparsers(dest="evolution_command")
    sub.add_parser("enable", help="Enable self-evolution logging")
    sub.add_parser("disable", help="Disable self-evolution logging")
    list_p = sub.add_parser("list", help="Show evolution timeline")
    _add_list_args(list_p)
    timeline_p = sub.add_parser("timeline", help="Alias for list")
    _add_list_args(timeline_p)
    show_p = sub.add_parser("show", help="Show one evolution event")
    show_p.add_argument("event_id")
    stats_p = sub.add_parser("stats", help="Show evolution statistics")
    stats_p.add_argument("--days", type=int, default=30)
    clear_p = sub.add_parser("clear", help="Clear old evolution events")
    clear_p.add_argument(
        "--older-than", dest="older_than", type=int, required=True, metavar="DAYS"
    )
    clear_p.add_argument("--yes", action="store_true")


def _add_list_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--type", dest="event_type", default=None)
    parser.add_argument("--target", default=None)


def _set_enabled(enabled: bool) -> dict[str, Any]:
    config = load_config()
    evolution = config.setdefault("evolution", {})
    evolution["enabled"] = enabled
    save_config(config)
    return config


def _is_enabled() -> bool:
    cfg = load_config()
    evolution = cfg.get("evolution") if isinstance(cfg, dict) else {}
    return bool(evolution.get("enabled")) if isinstance(evolution, dict) else False


def _short_id(event: dict[str, Any]) -> str:
    return str(event.get("id", ""))[-6:]


def _display_time(value: str | None) -> str:
    if not value:
        return "unknown"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%b %d %H:%M")
    except ValueError:
        return str(value)


def _list_events(args) -> int:
    events, warnings = read_events()
    for warning in warnings:
        print(f"Warning: {warning}")

    enabled = _is_enabled()
    if not events:
        if not enabled:
            print("Self-evolution logging is disabled.")
            print("No evolution events found.")
            print("Enable it with: hermes evolution enable")
        else:
            print("No evolution events found.")
        return 0

    if not enabled:
        print("Self-evolution logging is disabled; showing preserved events.")
    days = getattr(args, "days", 30)
    print(f"Hermes Evolution - last {days} days")
    events = sorted(
        events, key=lambda event: str(event.get("timestamp", "")), reverse=True
    )
    events = filter_events(
        events,
        days=days,
        event_type=getattr(args, "event_type", None),
        target_query=getattr(args, "target", None),
        limit=getattr(args, "limit", 50),
    )
    if not events:
        print("No evolution events found.")
        return 0
    for event in events:
        print(
            f"{_display_time(event.get('timestamp'))}  {_short_id(event)}  "
            f"{event.get('type', ''):<14}  {event.get('target_name', ''):<20}  {event.get('summary', '')}"
        )
    return 0


def _show_event(args) -> int:
    events, warnings = read_events()
    for warning in warnings:
        print(f"Warning: {warning}")
    event, matches = resolve_event_id(events, getattr(args, "event_id", ""))
    if event is None:
        if matches:
            print("Ambiguous event ID. Use the full event ID:")
            for match in matches:
                print(f"  {match.get('id', '')}")
        else:
            print("Evolution event not found.")
        return 1

    print(f"ID: {event.get('id', '')}")
    print(f"Time: {_display_time(event.get('timestamp'))}")
    print(f"Type: {event.get('type', '')}")
    print(f"Actor: {event.get('actor', '')}")
    print(f"Source tool: {event.get('source_tool', '')}")
    print(f"Target: {event.get('target', '')}")
    print(
        f"Target kind/name: {event.get('target_kind', '')} / {event.get('target_name', '')}"
    )
    print(f"Summary: {event.get('summary', '')}")
    print(f"Reason: {event.get('reason') or ''}")
    if event.get("profile"):
        print(f"Profile: {event.get('profile')}")
    if event.get("platform"):
        print(f"Platform: {event.get('platform')}")
    if event.get("session_id"):
        print(f"Session: {event.get('session_id')}")
    print(
        f"Redaction: enabled={event.get('redaction_enabled')} applied={event.get('redaction_applied')}"
    )
    print(f"Diff format: {event.get('diff_format', '')}")
    print(f"Truncated: {event.get('diff_truncated')}")
    print("Diff:")
    print(event.get("diff") or "")
    return 0


def _stats(args) -> int:
    days = getattr(args, "days", 30)
    events, warnings = read_events()
    for warning in warnings:
        print(f"Warning: {warning}")
    events = sorted(events, key=lambda event: str(event.get("timestamp", "")))
    events = filter_events(events, days=days)

    print(f"Hermes Evolution Stats - last {days} days")
    print("\nOverview:")
    print(f"  Total events: {len(events)}")
    if events:
        print(f"  First event:  {_display_time(events[0].get('timestamp'))}")
        print(f"  Latest event: {_display_time(events[-1].get('timestamp'))}")
    else:
        print("  First event:  none")
        print("  Latest event: none")

    type_counts = Counter(str(event.get("type", "")) for event in events)
    category_counts = Counter(
        str(event.get("type", "")).split(".", 1)[0] for event in events if event.get("type")
    )
    print(f"  Memory:  {category_counts.get('memory', 0)}")
    print(f"  Skills:  {category_counts.get('skill', 0)}")
    print(f"  Curator: {category_counts.get('curator', 0)}")

    print("\nBy type:")
    for event_type, count in sorted(type_counts.items()):
        print(f"  {event_type:<18} {count}")

    print("\nMost evolved targets:")
    target_counts = Counter(
        str(event.get("target", "")) for event in events if event.get("target")
    )
    for target, count in target_counts.most_common(10):
        print(f"  {target:<40} {count}")

    print("\nActivity by day:")
    day_counts = Counter(
        str(event.get("timestamp", ""))[:10]
        for event in events
        if event.get("timestamp")
    )
    for day, count in sorted(day_counts.items()):
        print(f"  {day}  {'#' * count} {count}")
    return 0


def _clear(args) -> int:
    deleted, retained = clear_older_than(
        getattr(args, "older_than"), apply=bool(getattr(args, "yes", False))
    )
    if getattr(args, "yes", False):
        print(f"Deleted {deleted} evolution events; retained {retained}.")
    else:
        print(
            f"Would delete {deleted} evolution events; retained {retained}. Re-run with --yes to apply."
        )
    return 0


def evolution_command(args) -> int:
    command = getattr(args, "evolution_command", None)
    if command == "enable":
        _set_enabled(True)
        ensure_evolution_dir()
        print(
            f"Self-evolution logging enabled. Future events will be written to {get_events_path()}."
        )
        return 0
    if command == "disable":
        _set_enabled(False)
        print("Self-evolution logging disabled. Existing events are preserved.")
        return 0
    if command in {"list", "timeline"}:
        return _list_events(args)
    if command == "show":
        return _show_event(args)
    if command == "stats":
        return _stats(args)
    if command == "clear":
        return _clear(args)
    print("Usage: hermes evolution enable|disable|list|timeline|show|stats|clear")
    return 0

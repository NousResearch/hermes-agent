"""Shared ``/proactive`` command logic for CLI and gateway."""

from __future__ import annotations

import shlex
from typing import Any, Optional


def _parse_args(args: str) -> dict[str, Any]:
    try:
        parts = shlex.split(args or "")
    except ValueError:
        parts = (args or "").split()

    parsed: dict[str, Any] = {
        "days": 30,
        "source": None,
        "limit": 5,
        "min_messages": 2,
        "json": False,
    }
    i = 0
    while i < len(parts):
        item = parts[i]
        if item == "--days" and i + 1 < len(parts):
            parsed["days"] = int(parts[i + 1])
            i += 2
            continue
        if item == "--source" and i + 1 < len(parts):
            parsed["source"] = parts[i + 1]
            i += 2
            continue
        if item == "--limit" and i + 1 < len(parts):
            parsed["limit"] = int(parts[i + 1])
            i += 2
            continue
        if item == "--min-messages" and i + 1 < len(parts):
            parsed["min_messages"] = int(parts[i + 1])
            i += 2
            continue
        if item == "--json":
            parsed["json"] = True
            i += 1
            continue
        if item.isdigit():
            parsed["days"] = int(item)
        i += 1
    return parsed


def handle_proactive_command(args: str, *, db: Optional[Any] = None) -> str:
    """Run proactive opportunity detection and return display text."""
    try:
        opts = _parse_args(args)
    except ValueError as exc:
        return f"Invalid proactive command arguments: {exc}"

    created_db = False
    if db is None:
        from hermes_state import SessionDB

        db = SessionDB()
        created_db = True
    try:
        from agent.proactive import ProactiveEngine, dumps_report

        engine = ProactiveEngine(db)
        report = engine.generate(
            days=opts["days"],
            source=opts["source"],
            limit=opts["limit"],
            min_messages=opts["min_messages"],
        )
        return dumps_report(report) if opts["json"] else engine.format_terminal(report)
    except Exception as exc:
        return f"Proactive command failed: {exc}"
    finally:
        if created_db:
            try:
                db.close()
            except Exception:
                pass

"""Lightweight runtime event log for operator diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home


EVENTS_FILE = "runtime_events.jsonl"
_MAX_DETAIL_CHARS = 1200


def _events_path(hermes_home: str | Path | None = None) -> Path:
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    return home / EVENTS_FILE


def _clean_detail(detail: Any) -> str:
    text = "" if detail is None else str(detail)
    if len(text) > _MAX_DETAIL_CHARS:
        return text[: _MAX_DETAIL_CHARS - 1] + "…"
    return text


def append_runtime_event(
    *,
    kind: str,
    status: str,
    name: str,
    detail: Any = None,
    payload: dict[str, Any] | None = None,
    hermes_home: str | Path | None = None,
) -> None:
    """Append one runtime event as JSONL.

    This is deliberately best-effort. Runtime diagnostics must never break the
    tool, provider, cron, or gateway path they observe.
    """
    try:
        path = _events_path(hermes_home)
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": str(kind),
            "status": str(status),
            "name": str(name),
        }
        cleaned = _clean_detail(detail)
        if cleaned:
            event["detail"] = cleaned
        if payload:
            event["payload"] = payload
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        return


def iter_runtime_events(
    *,
    limit: int = 50,
    kind: str | None = None,
    status: str | None = None,
    hermes_home: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Read recent runtime events, newest last."""
    path = _events_path(hermes_home)
    if not path.exists():
        return []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    events: list[dict[str, Any]] = []
    for line in lines[-max(1, limit * 4):]:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if kind and event.get("kind") != kind:
            continue
        if status and event.get("status") != status:
            continue
        events.append(event)
    return events[-limit:]


def _format_event(event: dict[str, Any]) -> str:
    detail = event.get("detail")
    suffix = f" — {detail}" if detail else ""
    return (
        f"{event.get('ts', '?')} "
        f"{event.get('kind', '?')}/{event.get('status', '?')} "
        f"{event.get('name', '?')}{suffix}"
    )


def run_runtime_events(args) -> None:
    """CLI entrypoint for ``hermes runtime-events``."""
    events = iter_runtime_events(
        limit=max(1, int(getattr(args, "limit", 50) or 50)),
        kind=getattr(args, "kind", None),
        status=getattr(args, "status", None),
    )
    if getattr(args, "json", False):
        print(json.dumps(events, ensure_ascii=False, indent=2))
        return
    if not events:
        print("No runtime events recorded.")
        return
    for event in events:
        print(_format_event(event))

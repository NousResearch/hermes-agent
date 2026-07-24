"""Bounded agent telemetry (Jarvis Layer E / Phase 3).

Records two kinds of events to a small newline-delimited JSON log:
- "route" — a model-routing decision (task, tier, model), written server-side
- "tool"  — a tool-call outcome (name, tier, ok, approved), reported by the
            client after it runs each gated tool

The log is capped at MAX events (oldest dropped) and mirrored in memory so the
status widget can read recent activity and summary stats cheaply. Nothing here
is personal — tool *names* and tiers, not their arguments or results.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path


class Telemetry:
    MAX = 500

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._events: list[dict] = []
        if path.exists():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        self._events.append(json.loads(line))
                self._events = self._events[-self.MAX:]
            except (OSError, json.JSONDecodeError):
                self._events = []  # corrupted log: start fresh

    def record(self, event: dict) -> dict:
        ev = {"at": round(time.time(), 3), **event}
        with self._lock:
            self._events.append(ev)
            if len(self._events) > self.MAX:
                self._events = self._events[-self.MAX:]
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(
                    "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in self._events),
                    encoding="utf-8")
            except OSError:
                pass  # telemetry is best-effort; never break a request over it
        return ev

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self._events[-limit:])

    def summary(self) -> dict:
        with self._lock:
            evs = list(self._events)
        tools = [e for e in evs if e.get("kind") == "tool"]
        routes = [e for e in evs if e.get("kind") == "route"]
        return {
            "total": len(evs),
            "tool_calls": len(tools),
            "denied": sum(1 for e in tools if e.get("approved") is False),
            "escalations": sum(1 for e in evs if e.get("kind") == "advisor"),
            "by_tier": dict(Counter(e.get("tier") for e in routes if e.get("tier"))),
            "by_tool": dict(Counter(e.get("name") for e in tools if e.get("name"))),
        }

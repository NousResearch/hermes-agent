from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEFAULT_LOG_DIR = Path.home() / ".hermes" / "swarm" / "logs"


class SwarmMonitor:
    def __init__(self, log_dir: Path | None = None) -> None:
        self._log_dir = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record_event(
        self,
        event_type: str,
        task_id: str | None = None,
        worker_id: str | None = None,
        data: dict | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
        }
        if task_id is not None:
            event["task_id"] = task_id
        if worker_id is not None:
            event["worker_id"] = worker_id
        if data is not None:
            event["data"] = data
        with self._lock:
            self._events.append(event)

    def record_api_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_s: float,
        cost_usd: float,
        task_id: str | None = None,
    ) -> None:
        self.record_event(
            "api_call",
            task_id=task_id,
            data={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "duration_s": duration_s,
                "cost_usd": cost_usd,
            },
        )

    def record_tool_call(
        self,
        tool_name: str,
        duration_s: float,
        success: bool,
        task_id: str | None = None,
    ) -> None:
        self.record_event(
            "tool_call",
            task_id=task_id,
            data={
                "tool_name": tool_name,
                "duration_s": duration_s,
                "success": success,
            },
        )

    def record_task_transition(
        self, task_id: str, from_state: str, to_state: str
    ) -> None:
        self.record_event(
            "task_transition",
            task_id=task_id,
            data={"from_state": from_state, "to_state": to_state},
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_timeline(self, last_n: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events[-last_n:])

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)

        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        api_durations: list[float] = []
        by_model: dict[str, dict[str, Any]] = {}
        by_tool: dict[str, dict[str, int]] = {}

        task_states: dict[str, str] = {}  # latest state per task

        for ev in events:
            etype = ev["type"]
            data = ev.get("data", {})

            if etype == "api_call":
                model = data.get("model", "unknown")
                cost = data.get("cost_usd", 0.0)
                inp = data.get("input_tokens", 0)
                out = data.get("output_tokens", 0)
                dur = data.get("duration_s", 0.0)

                total_cost += cost
                total_input_tokens += inp
                total_output_tokens += out
                api_durations.append(dur)

                entry = by_model.setdefault(model, {
                    "calls": 0, "cost_usd": 0.0,
                    "input_tokens": 0, "output_tokens": 0,
                })
                entry["calls"] += 1
                entry["cost_usd"] += cost
                entry["input_tokens"] += inp
                entry["output_tokens"] += out

            elif etype == "tool_call":
                tool = data.get("tool_name", "unknown")
                entry = by_tool.setdefault(tool, {
                    "calls": 0, "successes": 0, "failures": 0,
                })
                entry["calls"] += 1
                if data.get("success"):
                    entry["successes"] += 1
                else:
                    entry["failures"] += 1

            elif etype == "task_transition":
                tid = ev.get("task_id", "")
                task_states[tid] = data.get("to_state", "")

        completed = sum(1 for s in task_states.values() if s == "completed")
        failed = sum(1 for s in task_states.values() if s == "failed")
        avg_duration = (
            sum(api_durations) / len(api_durations) if api_durations else 0.0
        )

        return {
            "total_tasks": len(task_states),
            "completed": completed,
            "failed": failed,
            "total_cost_usd": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "avg_api_duration_s": round(avg_duration, 4),
            "total_events": len(events),
            "by_model": by_model,
            "by_tool": by_tool,
        }

    def get_cost_breakdown(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)

        by_model: dict[str, float] = {}
        by_role: dict[str, float] = {}
        by_task: dict[str, float] = {}

        for ev in events:
            if ev["type"] != "api_call":
                continue
            data = ev.get("data", {})
            cost = data.get("cost_usd", 0.0)

            model = data.get("model", "unknown")
            by_model[model] = by_model.get(model, 0.0) + cost

            role = data.get("role", "")
            if role:
                by_role[role] = by_role.get(role, 0.0) + cost

            tid = ev.get("task_id")
            if tid:
                by_task[tid] = by_task.get(tid, 0.0) + cost

        total = sum(by_model.values())
        return {
            "total_cost_usd": total,
            "by_model": by_model,
            "by_role": by_role,
            "by_task": by_task,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, path: Path | None = None) -> Path:
        if path is None:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = self._log_dir / f"telemetry_{ts}.json"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            events_copy = list(self._events)

        summary = self.get_summary()
        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "events": events_copy,
        }

        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def print_dashboard(self) -> None:
        s = self.get_summary()
        cb = self.get_cost_breakdown()

        _hr = "-" * 60
        lines = [
            "",
            _hr,
            "  SWARM MONITOR DASHBOARD",
            _hr,
            "",
            f"  Total events:   {s['total_events']}",
            f"  Total tasks:    {s['total_tasks']}",
            f"  Completed:      {s['completed']}",
            f"  Failed:         {s['failed']}",
            "",
            f"  Total cost:     ${s['total_cost_usd']:.6f}",
            f"  Total tokens:   {s['total_tokens']:,}",
            f"    Input:        {s['total_input_tokens']:,}",
            f"    Output:       {s['total_output_tokens']:,}",
            f"  Avg API call:   {s['avg_api_duration_s']:.4f}s",
            "",
        ]

        # Model breakdown
        if s["by_model"]:
            lines.append("  MODEL BREAKDOWN")
            lines.append(f"  {'Model':<20} {'Calls':>6} {'Cost':>12} {'Tokens':>12}")
            lines.append(f"  {'-'*20} {'-'*6} {'-'*12} {'-'*12}")
            for model, info in sorted(s["by_model"].items()):
                tokens = info["input_tokens"] + info["output_tokens"]
                lines.append(
                    f"  {model:<20} {info['calls']:>6} "
                    f"${info['cost_usd']:>11.6f} {tokens:>12,}"
                )
            lines.append("")

        # Tool breakdown
        if s["by_tool"]:
            lines.append("  TOOL BREAKDOWN")
            lines.append(f"  {'Tool':<25} {'Calls':>6} {'OK':>6} {'Fail':>6}")
            lines.append(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6}")
            for tool, info in sorted(s["by_tool"].items()):
                lines.append(
                    f"  {tool:<25} {info['calls']:>6} "
                    f"{info['successes']:>6} {info['failures']:>6}"
                )
            lines.append("")

        # Cost by task
        if cb["by_task"]:
            lines.append("  COST BY TASK")
            lines.append(f"  {'Task ID':<20} {'Cost':>12}")
            lines.append(f"  {'-'*20} {'-'*12}")
            for tid, cost in sorted(cb["by_task"].items(), key=lambda x: -x[1]):
                lines.append(f"  {tid:<20} ${cost:>11.6f}")
            lines.append("")

        lines.append(_hr)
        print("\n".join(lines))

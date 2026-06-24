"""
hermes heal — self-healing engine CLI.

Subcommands:
  hermes heal [status]   Pretty-print healer status (default)
  hermes heal history    Readable audit trail from incidents.jsonl
  hermes heal trigger    Run one healer pass now
  hermes heal summary    Weekly-style incident rollup (default 7 days)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home


def _healer_dir() -> Path:
    return get_hermes_home() / "healer"


def _status_path() -> Path:
    return _healer_dir() / "status.json"


def _incidents_path() -> Path:
    return _healer_dir() / "incidents.jsonl"


def _healer_script() -> Path:
    return get_hermes_home() / "scripts" / "hermes-healer.ps1"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _iter_incidents(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = _incidents_path()
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if limit is not None:
        lines = lines[-limit:]
    out: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                out.append(row)
        except json.JSONDecodeError:
            continue
    return out


def _parse_ts(raw: object) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _overall_color(overall: str) -> str:
    return {
        "healthy": "32",
        "degraded": "33",
        "failed": "31",
        "circuit_open": "31",
    }.get(overall, "36")


def cmd_heal_status(args) -> int:
    status = _load_json(_status_path())
    if not status:
        print("No healer status yet. Run: hermes heal trigger")
        return 1

    overall = str(status.get("overall") or "unknown")
    print(_color(f"Overall: {overall}", _overall_color(overall)))
    summary = status.get("summary")
    if summary:
        print(f"Summary: {summary}")

    last = status.get("last_run") or {}
    if isinstance(last, dict):
        print(
            f"Last run: {last.get('finished_at', '?')} "
            f"(actions={last.get('actions_taken', 0)}, exit={last.get('exit_code', '?')})"
        )

    components = status.get("components") or {}
    if isinstance(components, dict):
        print("\nComponents:")
        for name in sorted(components):
            comp = components[name]
            if not isinstance(comp, dict):
                continue
            health = comp.get("health", "?")
            issue = comp.get("issue")
            flags: List[str] = []
            if comp.get("in_cooldown"):
                flags.append(f"cooldown→{comp.get('cooldown_until', '?')}")
            if comp.get("circuit_open"):
                flags.append("CIRCUIT OPEN")
            restarts = comp.get("restart_count_24h", 0)
            if restarts:
                flags.append(f"restarts24h={restarts}")
            extra = f" ({', '.join(flags)})" if flags else ""
            line = f"  {name:12} {health}"
            if issue:
                line += f" — {issue}"
            line += extra
            print(line)

    if status.get("user_alert_pending"):
        print(f"\n{_color('USER ALERT PENDING', '31')}: {status.get('user_alert_path')}")

    print(f"\nStatus file: {_status_path()}")
    return 0 if overall == "healthy" else 1


def cmd_heal_history(args) -> int:
    limit = int(getattr(args, "limit", 30) or 30)
    rows = _iter_incidents(limit=limit)
    if not rows:
        print("No incidents logged yet.")
        return 0

    for row in rows:
        ts = row.get("ts", "?")
        kind = row.get("type", "?")
        if kind == "action":
            print(
                f"{ts}  ACTION  {row.get('component', '?')} "
                f"tier {row.get('tier', '?')}: {row.get('detail', '')}"
            )
        elif kind == "probe":
            ok = row.get("ok")
            mark = "OK" if ok else "FAIL"
            issues = row.get("issues") or {}
            if isinstance(issues, dict) and issues:
                detail = "; ".join(f"{k}: {v}" for k, v in issues.items())
            else:
                detail = "all clear"
            print(f"{ts}  PROBE   {mark}  {detail}")
        else:
            print(f"{ts}  {kind.upper()}  {row}")
    return 0


def cmd_heal_trigger(args) -> int:
    script = _healer_script()
    if not script.is_file():
        print(f"Healer script not found: {script}")
        return 1

    ps = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if not ps.is_file():
        ps = Path("powershell")

    proc = subprocess.run(
        [
            str(ps),
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
        ],
        env={**os.environ, "HERMES_HOME": str(get_hermes_home())},
    )
    return int(proc.returncode)


def _build_summary(days: int) -> Dict[str, Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = _iter_incidents()
    recent = [r for r in rows if (_parse_ts(r.get("ts")) or datetime.min.replace(tzinfo=timezone.utc)) >= cutoff]

    probes = [r for r in recent if r.get("type") == "probe"]
    actions = [r for r in recent if r.get("type") == "action"]
    failed_probes = [r for r in probes if not r.get("ok")]

    issue_counter: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()
    tier_counter: Counter[int] = Counter()

    for probe in failed_probes:
        issues = probe.get("issues") or {}
        if isinstance(issues, dict):
            for key in issues:
                issue_counter[key] += 1

    for action in actions:
        comp = str(action.get("component") or "unknown")
        action_counter[comp] += 1
        try:
            tier_counter[int(action.get("tier") or 0)] += 1
        except (TypeError, ValueError):
            pass

    status = _load_json(_status_path()) or {}
    overall = status.get("overall", "unknown")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "overall_now": overall,
        "probe_count": len(probes),
        "failed_probe_count": len(failed_probes),
        "action_count": len(actions),
        "top_issues": issue_counter.most_common(10),
        "actions_by_component": action_counter.most_common(),
        "actions_by_tier": sorted(tier_counter.items()),
        "uptime_pct": round(100.0 * (len(probes) - len(failed_probes)) / len(probes), 1) if probes else None,
    }


def _format_summary_text(summary: Dict[str, Any]) -> str:
    lines = [
        f"Hermes Healer Summary ({summary.get('window_days')} days)",
        f"Generated: {summary.get('generated_at')}",
        f"Current overall: {summary.get('overall_now')}",
    ]
    uptime = summary.get("uptime_pct")
    if uptime is not None:
        lines.append(
            f"Probe uptime: {uptime}% "
            f"({summary.get('probe_count') - summary.get('failed_probe_count')}/{summary.get('probe_count')} clean)"
        )
    lines.append(f"Repair actions taken: {summary.get('action_count')}")

    top_issues = summary.get("top_issues") or []
    if top_issues:
        lines.append("\nTop issues:")
        for name, count in top_issues:
            lines.append(f"  - {name}: {count} failed probe(s)")

    by_comp = summary.get("actions_by_component") or []
    if by_comp:
        lines.append("\nActions by component:")
        for name, count in by_comp:
            lines.append(f"  - {name}: {count}")

    tiers = summary.get("actions_by_tier") or []
    if tiers:
        lines.append("\nActions by tier:")
        for tier, count in tiers:
            lines.append(f"  - tier {tier}: {count}")

    return "\n".join(lines) + "\n"


def cmd_heal_summary(args) -> int:
    days = int(getattr(args, "days", 7) or 7)
    summary = _build_summary(days)
    text = _format_summary_text(summary)

    if getattr(args, "json", False):
        print(json.dumps(summary, indent=2))
    else:
        print(text)

    if getattr(args, "write", False):
        out_dir = _healer_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d")
        md_path = out_dir / f"weekly-summary-{stamp}.md"
        json_path = out_dir / "weekly-summary-latest.json"
        md_path.write_text(text, encoding="utf-8")
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")

    return 0


def cmd_heal(args) -> int:
    sub = getattr(args, "heal_command", None)
    if sub in (None, "status"):
        return cmd_heal_status(args)
    if sub == "history":
        return cmd_heal_history(args)
    if sub == "trigger":
        return cmd_heal_trigger(args)
    if sub == "summary":
        return cmd_heal_summary(args)
    print("Usage: hermes heal [status|history|trigger|summary]")
    return 1


def build_heal_parser(subparsers, cmd_heal=cmd_heal) -> None:
    heal_parser = subparsers.add_parser(
        "heal",
        help="Self-healing engine status, history, and manual trigger",
        description="Inspect and control the local Hermes self-healing engine.",
    )
    heal_sub = heal_parser.add_subparsers(dest="heal_command")

    status_p = heal_sub.add_parser("status", help="Show live healer status (default)")
    status_p.set_defaults(func=cmd_heal)

    hist_p = heal_sub.add_parser("history", help="Show recent incidents from the audit log")
    hist_p.add_argument("--limit", type=int, default=30, help="Number of incident lines (default 30)")
    hist_p.set_defaults(func=cmd_heal)

    trigger_p = heal_sub.add_parser("trigger", help="Run one healer pass immediately")
    trigger_p.set_defaults(func=cmd_heal)

    summary_p = heal_sub.add_parser("summary", help="Roll up incidents over the last N days")
    summary_p.add_argument("--days", type=int, default=7, help="Window in days (default 7)")
    summary_p.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    summary_p.add_argument(
        "--write",
        action="store_true",
        help="Write healer/weekly-summary-YYYYMMDD.md and weekly-summary-latest.json",
    )
    summary_p.set_defaults(func=cmd_heal)

    heal_parser.set_defaults(func=cmd_heal, heal_command=None)

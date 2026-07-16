#!/usr/bin/env python3
"""Per-profile boot-prompt + ctx/call audit for the token-economy program.

Two numbers per profile:

  * ``boot_prompt`` -- the stable+volatile system-prompt tiers a profile
    ships on EVERY fresh session regardless of workspace: identity
    (SOUL.md), tool-use guidance, the kanban worker lifecycle protocol,
    the skills index, and injected memory/user-profile blocks. Built via
    an offline inspection ``AIAgent`` (no network call -- same technique
    as ``hermes prompt-size`` / ``agent/context_breakdown.py``) with
    ``skip_context_files=True`` so the workspace-dependent ``context``
    tier (AGENTS.md et al, which varies per task and isn't a profile-
    level lever) is excluded from the comparison.
  * ``ctx_per_call`` -- context volume per API call over a trailing
    window, read from ``profiles/<p>/state.db``'s ``sessions`` table:
    ``(input + cache_read + cache_write tokens) / api_call_count``. This
    is the same KPI the hermes-metrics dashboard surfaces
    (``plugins/hermes-metrics/dashboard/plugin_api.py::compute_usage``).

Read-only: does not modify any profile state. Use ``--save`` to snapshot
a baseline and ``--diff`` to compare a later run against it.

Usage:
    scripts/context_audit.py                       # all profiles, table
    scripts/context_audit.py --profile pm           # one profile
    scripts/context_audit.py --json                 # machine-readable
    scripts/context_audit.py --save baseline.json   # snapshot for diffing
    scripts/context_audit.py --diff baseline.json   # compare vs a snapshot
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hermes_constants import (  # noqa: E402
    get_default_hermes_root,
    reset_hermes_home_override,
    set_hermes_home_override,
)


def _discover_profiles(root: Path) -> List[str]:
    """Return profile names with a state.db -- 'default' plus each dir
    under profiles/ that has one."""
    names: List[str] = []
    if (root / "state.db").exists():
        names.append("default")
    profiles_dir = root / "profiles"
    if profiles_dir.is_dir():
        for entry in sorted(profiles_dir.iterdir()):
            if entry.is_dir() and (entry / "state.db").exists():
                names.append(entry.name)
    return names


def _profile_home(root: Path, name: str) -> Path:
    return root if name == "default" else root / "profiles" / name


def measure_boot_prompt(profile_home: Path) -> Dict[str, Any]:
    """Build an offline inspection agent scoped to *profile_home* and
    return char/token counts for its boot-prompt (stable+volatile tiers).

    Sets ``HERMES_KANBAN_TASK`` for the duration of the build so the
    kanban worker-lifecycle guidance (only injected for dispatcher-
    spawned workers) is included for profiles that run as kanban
    workers -- matching what they actually see in production.
    """
    from tools.registry import invalidate_check_fn_cache

    token = set_hermes_home_override(str(profile_home))
    prior_kanban_task = os.environ.get("HERMES_KANBAN_TASK")
    os.environ["HERMES_KANBAN_TASK"] = "context-audit-dummy"
    invalidate_check_fn_cache()
    try:
        from agent.model_metadata import estimate_tokens_rough
        from agent.system_prompt import build_system_prompt_parts
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools
        from run_agent import AIAgent

        cfg = load_config()
        model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
        model = model_cfg.get("default") or model_cfg.get("model") or ""
        enabled_toolsets = sorted(_get_platform_tools(cfg, "cli"))
        agent_cfg = cfg.get("agent") or {}
        disabled_toolsets = agent_cfg.get("disabled_toolsets") or None

        agent = AIAgent(
            model=model,
            api_key="inspect-only",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            save_trajectories=False,
            platform="cli",
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            # Boot-prompt excludes cwd-dependent AGENTS.md/.cursorrules --
            # those vary per task/workspace, not per profile -- but DOES
            # include the profile's own SOUL.md (persona + standing orders,
            # e.g. PRODUCTION-CHANGE POLICY), which is what the spec means
            # by "boot-prompt ... incl. SOUL.md". Same combination cron
            # sessions use (see cron/scheduler.py) to keep persona while
            # dropping cwd project files.
            skip_context_files=True,
            load_soul_identity=True,
        )
        parts = build_system_prompt_parts(agent)
        stable = parts.get("stable", "") or ""
        volatile = parts.get("volatile", "") or ""
        boot_prompt = "\n\n".join(p for p in (stable, volatile) if p)
        return {
            "chars": len(boot_prompt),
            "tokens_est": estimate_tokens_rough(boot_prompt),
            "model": getattr(agent, "model", "") or "",
            "tool_count": len(getattr(agent, "tools", None) or []),
        }
    finally:
        if prior_kanban_task is None:
            os.environ.pop("HERMES_KANBAN_TASK", None)
        else:
            os.environ["HERMES_KANBAN_TASK"] = prior_kanban_task
        reset_hermes_home_override(token)
        invalidate_check_fn_cache()


def measure_ctx_per_call(state_db: Path, window_days: int = 7) -> Dict[str, Any]:
    """Read the ctx/call KPI straight from a profile's sessions table.

    Opens SQLite in read-only (mode=ro) so this never contends with a
    live gateway/CLI writer -- same pattern as the hermes-metrics plugin.
    """
    floor = time.time() - window_days * 86400
    try:
        conn = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        return {"error": str(exc)}
    try:
        row = conn.execute(
            "SELECT COUNT(*) sessions, COALESCE(SUM(input_tokens),0) inp,"
            " COALESCE(SUM(cache_read_tokens),0) cr,"
            " COALESCE(SUM(cache_write_tokens),0) cw,"
            " COALESCE(SUM(api_call_count),0) api"
            " FROM sessions WHERE started_at >= ?",
            (floor,),
        ).fetchone()
    except sqlite3.Error as exc:
        return {"error": str(exc)}
    finally:
        conn.close()
    sessions = int(row["sessions"]) if row else 0
    api_calls = int(row["api"]) if row else 0
    if not api_calls:
        return {"sessions": sessions, "api_calls": 0, "context_per_call": None}
    ctx = row["inp"] + row["cr"] + row["cw"]
    return {
        "sessions": sessions,
        "api_calls": api_calls,
        "context_per_call": int(ctx / api_calls),
    }


def run_audit(
    profiles: Optional[List[str]] = None, window_days: int = 7
) -> Dict[str, Any]:
    root = get_default_hermes_root()
    names = profiles or _discover_profiles(root)
    results: Dict[str, Any] = {}
    for name in names:
        home = _profile_home(root, name)
        boot = measure_boot_prompt(home)
        ctx = measure_ctx_per_call(home / "state.db", window_days=window_days)
        results[name] = {"boot_prompt": boot, "ctx_per_call": ctx}
    return {
        "window_days": window_days,
        "generated_at": int(time.time()),
        "profiles": results,
    }


def render_table(data: Dict[str, Any]) -> str:
    lines = [f"Context audit ({data['window_days']}d window)", ""]
    header = (
        f"{'profile':<18} {'boot chars':>11} {'boot tok~':>10} "
        f"{'tools':>6} {'ctx/call':>10} {'api calls':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, r in sorted(data["profiles"].items()):
        boot = r["boot_prompt"]
        ctx = r["ctx_per_call"]
        ctx_call = ctx.get("context_per_call")
        ctx_call_s = f"{ctx_call:,}" if ctx_call is not None else "\u2014"
        api_calls = ctx.get("api_calls", 0)
        lines.append(
            f"{name:<18} {boot['chars']:>11,} {boot['tokens_est']:>10,} "
            f"{boot['tool_count']:>6} {ctx_call_s:>10} {api_calls:>10,}"
        )
    return "\n".join(lines)


def render_diff(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    lines = ["Context audit diff (before -> after)", ""]
    header = (
        f"{'profile':<18} {'boot tok before':>16} {'boot tok after':>15} "
        f"{'delta':>8} {'pct':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    names = sorted(set(before["profiles"]) & set(after["profiles"]))
    for name in names:
        b = before["profiles"][name]["boot_prompt"]["tokens_est"]
        a = after["profiles"][name]["boot_prompt"]["tokens_est"]
        delta = a - b
        pct = (delta / b * 100) if b else 0.0
        lines.append(
            f"{name:<18} {b:>16,} {a:>15,} {delta:>+8,} {pct:>+6.1f}%"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", action="append",
        help="Limit to specific profile(s); repeatable.",
    )
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--save", metavar="PATH",
        help="Write the audit snapshot as JSON to PATH.",
    )
    parser.add_argument(
        "--diff", metavar="PATH",
        help="Compare current audit against a saved snapshot (boot-prompt tokens).",
    )
    args = parser.parse_args()

    data = run_audit(profiles=args.profile, window_days=args.window_days)

    if args.save:
        Path(args.save).write_text(json.dumps(data, indent=2), encoding="utf-8")

    if args.diff:
        before = json.loads(Path(args.diff).read_text(encoding="utf-8"))
        print(render_diff(before, data))
        return

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(render_table(data))


if __name__ == "__main__":
    main()

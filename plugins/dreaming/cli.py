"""CLI commands for the Dreaming plugin: ``hermes dream <subcommand>``."""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from typing import Callable

from plugins.dreaming import (
    is_enabled,
    run_cycle,
    _dreams_path,
    _memory_path,
    _last_user_activity,
    _is_quiet,
    _cfg,
)

logger = logging.getLogger(__name__)


def setup_cli(subparser) -> None:
    """Register ``hermes dream`` subcommands."""
    sub = subparser.add_subparsers(dest="dream_command")

    run_p = sub.add_parser("run", help="Run a dreaming cycle now")
    run_p.add_argument("--force", "-f", action="store_true",
                       help="Skip the quiet-hours check")
    run_p.add_argument("--verbose", "-v", action="store_true",
                       help="Detailed output")

    sub.add_parser("status", help="Show dreaming status")

    diary_p = sub.add_parser("diary", help="Show recent dream diary entries")
    diary_p.add_argument("--limit", "-n", type=int, default=5,
                         help="Number of entries (default: 5)")

    sub.add_parser("enable", help="Enable dreaming in config")
    sub.add_parser("disable", help="Disable dreaming in config")


def handle_cli(args: Namespace) -> None:
    cmd = getattr(args, "dream_command", None) or "status"
    if cmd == "run":
        _cmd_run(args)
    elif cmd == "status":
        _cmd_status()
    elif cmd == "diary":
        _cmd_diary(args)
    elif cmd == "enable":
        _cmd_toggle(True)
    elif cmd == "disable":
        _cmd_toggle(False)


def _cmd_run(args: Namespace) -> None:
    force = getattr(args, "force", False)
    verbose = getattr(args, "verbose", False)
    print("🌙 Starting dreaming cycle...")
    report = run_cycle(force=force, verbose=verbose)
    if report is None:
        print("   Skipped (user active or disabled).")
        return
    print(f"   ✅ Complete — {report.light_count} staged, "
          f"{len(report.promoted)} promoted, {len(report.skipped)} skipped")
    if report.promoted:
        print("   Promoted:")
        for p in report.promoted:
            print(f"     • {p[:100]}")


def _cmd_status() -> None:
    last = _last_user_activity()
    print("🌙 Dreaming Status")
    print(f"   Enabled:     {'yes' if is_enabled() else 'no'}")
    print(f"   Frequency:   {_cfg('frequency', '0 3 * * *')}")
    print(f"   Quiet mins:  {_cfg('quiet_minutes', 60)}")
    print(f"   Lookback:    {_cfg('lookback_days', 7)} days")
    print(f"   Threshold:   {_cfg('promotion_threshold', 0.6)}")
    print(f"   Last active: {last or 'unknown'}")
    print(f"   User quiet:  {'yes' if _is_quiet() else 'no'}")
    print(f"   Dreams:      {_dreams_path()} ({'exists' if _dreams_path().exists() else 'not yet'})")
    print(f"   Memory:      {_memory_path()} ({'exists' if _memory_path().exists() else 'not yet'})")


def _cmd_diary(args: Namespace) -> None:
    p = _dreams_path()
    if not p.exists():
        print("No dream diary yet. Run: hermes dream run --force")
        return
    content = p.read_text(encoding="utf-8")
    entries = ["## Dream Cycle" + e for e in content.split("## Dream Cycle")[1:]]
    limit = getattr(args, "limit", 5)
    entries = entries[-limit:]
    if not entries:
        print("Dream diary is empty.")
        return
    print(f"🌙 Dream Diary (last {len(entries)} entries):\n")
    for e in entries:
        print(e.strip())
        print()


def _cmd_toggle(enabled: bool) -> None:
    key = "enabled"
    val = enabled
    try:
        from hermes_cli.config import load_config, save_config
        config = load_config()
        cfg = config.setdefault("plugins", {}).setdefault("entries", {}).setdefault("dreaming", {}).setdefault("config", {})
        cfg[key] = val
        save_config(config)
        state = "enabled" if enabled else "disabled"
        print(f"✅ Dreaming {state}. Restart the gateway to apply.")
    except Exception as exc:
        print(f"Error: {exc}")

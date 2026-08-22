"""``hermes fed`` subcommand — federation management CLI.

Commands:
    hermes fed status          Show federation status and connected peers
    hermes fed tasks           List all federation tasks
    hermes fed handoff TASK    Hand off a running task to another device
    hermes fed compute         Show compute capability of all peers
"""
from __future__ import annotations

import json
import sys
from typing import Callable, Optional


def build_fed_parser(subparsers, *, cmd_fed: Optional[Callable] = None) -> None:
    """Attach the ``fed`` subcommand to ``subparsers``."""
    fed_parser = subparsers.add_parser(
        "fed",
        help="Federation management (multi-device coordination)",
        description="Manage Hermes P2P federation — device discovery, task relay, and cross-device collaboration.",
    )
    fed_sub = fed_parser.add_subparsers(dest="fed_command", help="Federation command")

    # ── fed status ──────────────────────────────────────────────────
    p_status = fed_sub.add_parser(
        "status",
        help="Show federation status and connected peers",
    )
    p_status.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON",
    )
    p_status.set_defaults(func=_cmd_fed_status)

    # ── fed tasks ───────────────────────────────────────────────────
    p_tasks = fed_sub.add_parser(
        "tasks",
        help="List all federation tasks",
    )
    p_tasks.add_argument(
        "--active", action="store_true",
        help="Show only active tasks",
    )
    p_tasks.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON",
    )
    p_tasks.set_defaults(func=_cmd_fed_tasks)

    # ── fed handoff ─────────────────────────────────────────────────
    p_handoff = fed_sub.add_parser(
        "handoff",
        help="Hand off a running task to another device",
    )
    p_handoff.add_argument(
        "task_id",
        help="Task ID to hand off",
    )
    p_handoff.add_argument(
        "--reason", default="manual handoff",
        help="Reason for handoff",
    )
    p_handoff.set_defaults(func=_cmd_fed_handoff)

    # ── fed compute ─────────────────────────────────────────────────
    p_compute = fed_sub.add_parser(
        "compute",
        help="Show compute capability of all peers",
    )
    p_compute.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON",
    )
    p_compute.set_defaults(func=_cmd_fed_compute)

    fed_parser.set_defaults(func=_cmd_fed_default)


# ========================================================================
# Command implementations
# ========================================================================

def _cmd_fed_default(args) -> int:
    """Default handler — show help."""
    print("Usage: hermes fed <command>")
    print()
    print("Commands:")
    print("  status    Show federation status and connected peers")
    print("  tasks     List all federation tasks")
    print("  handoff   Hand off a running task to another device")
    print("  compute   Show compute capability of all peers")
    print()
    print("Use 'hermes fed <command> --help' for more info.")
    return 0


def _cmd_fed_status(args) -> int:
    """Show federation status."""
    from gateway.federation.federation_adapter import create_federation_adapter
    from gateway.config import FederationConfig

    cfg = FederationConfig()  # Will be overridden by config loading
    adapter = create_federation_adapter(enabled=True, mode="lan")

    peers = adapter.get_peers()
    is_connected = adapter.is_connected()

    if getattr(args, "json_output", False):
        output = {
            "device_id": adapter.device_id,
            "connected": is_connected,
            "peer_count": len(peers),
            "peers": [
                {
                    "device_id": p.device_id,
                    "hostname": p.hostname,
                    "status": p.status,
                    "score": round(p.compute_score, 1),
                }
                for p in peers
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"🔗 Federation Status")
        print(f"  Device: {adapter.device_id}")
        print(f"  Connected: {'✅ Yes' if is_connected else '❌ No'}")
        print(f"  Peers: {len(peers)}")
        print()
        if peers:
            print(f"  {'Device':<20} {'Hostname':<20} {'Status':<10} {'Score':<8}")
            print(f"  {'─' * 58}")
            for p in peers:
                print(f"  {p.device_id:<20} {p.hostname:<20} {p.status:<10} {p.compute_score:<8.1f}")

    return 0


def _cmd_fed_tasks(args) -> int:
    """List all federation tasks."""
    from gateway.federation.federation_adapter import create_federation_adapter

    adapter = create_federation_adapter(enabled=True, mode="lan")
    states = adapter.get_all_task_states()

    if getattr(args, "active", False):
        states = {k: v for k, v in states.items() if v.get("status") in ("claimed", "in_progress")}

    if getattr(args, "json_output", False):
        print(json.dumps(states, indent=2, default=str))
    else:
        if not states:
            print("📋 No federation tasks.")
            return 0

        print(f"📋 Federation Tasks ({len(states)} total)")
        print()
        print(f"  {'Task ID':<12} {'Status':<14} {'Title':<30} {'Progress':<10}")
        print(f"  {'─' * 66}")
        for task_id, state in states.items():
            title = state.get("title", "")[:28]
            status = state.get("status", "unknown")
            progress = state.get("progress", 0)
            progress_str = f"{progress * 100:.0f}%" if progress > 0 else "-"
            print(f"  {task_id:<12} {status:<14} {title:<30} {progress_str:<10}")

    return 0


def _cmd_fed_handoff(args) -> int:
    """Hand off a task to another device."""
    import asyncio
    from gateway.federation.federation_adapter import create_federation_adapter

    adapter = create_federation_adapter(enabled=True, mode="lan")

    async def _handoff():
        return await adapter.handoff_task(args.task_id, reason=args.reason)

    result = asyncio.run(_handoff())
    if result:
        print(f"✅ Task {args.task_id} handed off successfully.")
    else:
        print(f"❌ Failed to hand off task {args.task_id}.")
        return 1

    return 0


def _cmd_fed_compute(args) -> int:
    """Show compute capability of all peers."""
    from gateway.federation.federation_adapter import create_federation_adapter

    adapter = create_federation_adapter(enabled=True, mode="lan")
    peers = adapter.get_peers()

    if getattr(args, "json_output", False):
        output = {
            "peers": [
                {
                    "device_id": p.device_id,
                    "hostname": p.hostname,
                    "cpu_cores": p.cpu_cores,
                    "memory_gb": p.memory_gb,
                    "load_avg": p.load_avg,
                    "gpu_type": p.gpu_type,
                    "score": round(p.compute_score, 1),
                }
                for p in sorted(peers, key=lambda p: p.compute_score, reverse=True)
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        if not peers:
            print("💻 No peers connected.")
            return 0

        sorted_peers = sorted(peers, key=lambda p: p.compute_score, reverse=True)

        print(f"💻 Compute Capability ({len(peers)} peers)")
        print()
        print(f"  {'Device':<20} {'CPU':<6} {'Mem':<8} {'Load':<8} {'GPU':<15} {'Score':<8}")
        print(f"  {'─' * 65}")
        for p in sorted_peers:
            cpu = f"{p.cpu_cores}c"
            mem = f"{p.memory_gb:.0f}G" if p.memory_gb else "-"
            load = f"{p.load_avg:.1f}" if p.load_avg else "-"
            gpu = p.gpu_type[:13] if p.gpu_type else "-"
            print(f"  {p.device_id:<20} {cpu:<6} {mem:<8} {load:<8} {gpu:<15} {p.compute_score:<8.1f}")

    return 0

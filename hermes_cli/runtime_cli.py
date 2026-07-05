"""Runtime recovery CLI helpers."""

from __future__ import annotations

import argparse
import json as json_module
from pathlib import Path
from typing import Any


_OWNER_SUMMARY_ORDER = (
    "pid",
    "session_id",
    "session_id_fingerprint",
    "session_key",
    "session_key_fingerprint",
    "surface",
    "owner_kind",
    "cwd_fingerprint",
    "command_line_fingerprint",
)


def _format_owner_summary(summary: Any) -> str:
    if not isinstance(summary, dict):
        return "owner=unknown"
    parts = []
    for key in _OWNER_SUMMARY_ORDER:
        if key in summary:
            parts.append(f"{key}={summary[key]}")
    return " ".join(parts) if parts else "owner=unknown"


def _cmd_active_sessions_status(_args) -> int:
    from hermes_cli import active_sessions

    report = active_sessions.active_session_registry_status()
    _print_active_sessions_report(report)
    return 0


def _print_active_sessions_report(report: dict[str, Any]) -> None:
    lock_status = report.get("lock_status")
    read_mode = report.get("read_mode")
    lock_bits = ""
    if lock_status:
        lock_bits += f" lock_status={lock_status}"
    if read_mode:
        lock_bits += f" read_mode={read_mode}"
    print(
        "active sessions: "
        f"checked={report['checked']} live={report['live']} stale={report['stale']}"
        f"{lock_bits}"
    )
    if report.get("lock_owner_summary"):
        print(f"  lock_owner: {_format_owner_summary(report.get('lock_owner_summary'))}")
    for entry in report.get("entries", []):
        status = entry.get("runtime_status", "unknown")
        print(f"  {status}: {_format_owner_summary(entry.get('owner_summary'))}")


def _cmd_active_sessions_diagnose(args) -> int:
    from hermes_cli import active_sessions

    report = active_sessions.active_session_registry_status(
        no_lock=bool(getattr(args, "no_lock", False))
    )
    if getattr(args, "json", False):
        print(json_module.dumps(report, sort_keys=True))
        return 0
    _print_active_sessions_report(report)
    return 0


def _cmd_active_sessions_repair(args) -> int:
    if not getattr(args, "stale_only", False):
        print("refusing repair without --stale-only")
        return 2

    from hermes_cli import active_sessions

    report = active_sessions.repair_stale_active_session_leases(
        session_id=getattr(args, "session_id", None),
    )
    print(
        "active sessions repair: "
        f"checked={report['checked']} "
        f"stale={report['stale']} "
        f"repaired={report['repaired']} "
        f"skipped_live={report['skipped_live']}"
    )
    if report.get("repaired_session_ids"):
        print("  repaired_sessions=" + ",".join(report["repaired_session_ids"]))
    if report.get("skipped_live_session_ids"):
        print("  skipped_live_sessions=" + ",".join(report["skipped_live_session_ids"]))
    if getattr(args, "session_id", None) and report["repaired"] == 0 and report["skipped_live"] > 0:
        print("refused live active-session owner; no lease was removed")
        return 1
    return 0


def _cmd_control_plane_status(args) -> int:
    injected = getattr(args, "_control_plane_status", None)
    if injected is not None:
        report = injected
    else:
        from hermes_cli.control_plane import (
            build_control_plane_status,
            probe_websocket_url,
        )

        ws_probe_url = str(getattr(args, "ws_probe_url", "") or "").strip()
        report = build_control_plane_status(
            session_id=getattr(args, "session_id", None),
            ws_probe=(lambda: probe_websocket_url(ws_probe_url)) if ws_probe_url else None,
        )

    if getattr(args, "json", False):
        print(json_module.dumps(report, sort_keys=True))
        return 0

    websocket = report.get("websocket") if isinstance(report, dict) else {}
    if not isinstance(websocket, dict):
        websocket = {}
    print(
        "control-plane: "
        f"status={report.get('status', 'unknown')} "
        f"listener_alive={report.get('listener_alive')} "
        f"ws_probe={websocket.get('probe_status', 'unknown')} "
        f"stale_ws_clients={websocket.get('stale_closed_clients', 0)} "
        f"close_wait_count={report.get('close_wait_count')} "
        f"active_worker_progress={report.get('active_worker_progress')} "
        f"restart_guidance={report.get('restart_guidance', 'unknown')}"
    )
    if report.get("model_policy_violation") is True:
        print(
            "model-policy: "
            "model_policy_violation=True "
            f"required_model={report.get('required_model', '')} "
            f"recommended_action={report.get('model_policy_recommended_action', 'interrupt_and_restore_fixed_model')}"
        )
    for session in report.get("sessions", []) if isinstance(report, dict) else []:
        if not isinstance(session, dict):
            continue
        if session.get("session_id"):
            identity = f"session_id={session.get('session_id')}"
        elif session.get("session_id_fingerprint"):
            identity = f"session_id_fingerprint={session.get('session_id_fingerprint')}"
        else:
            identity = "session_id=unknown"
        parts = [
            identity,
            f"runtime_status={session.get('runtime_status', 'unknown')}",
            f"queued_steer_count={session.get('queued_steer_count', 0)}",
        ]
        if session.get("steer_boundary"):
            parts.append(f"steer_boundary={session['steer_boundary']}")
        if session.get("model_request_status"):
            parts.append(f"model_request_status={session['model_request_status']}")
        if session.get("db_lifecycle_status"):
            parts.append(f"db_lifecycle_status={session['db_lifecycle_status']}")
        if session.get("repair_recommendation"):
            parts.append(f"repair_recommendation={session['repair_recommendation']}")
        if session.get("model_policy_violation") is True:
            parts.append("model_policy_violation=True")
            parts.append(f"required_model={session.get('required_model', '')}")
            parts.append(
                "recommended_action="
                f"{session.get('model_policy_recommended_action', 'interrupt_and_restore_fixed_model')}"
            )
        print("  " + " ".join(parts))
    return 0


def _cmd_control_plane_steer(args) -> int:
    from hermes_cli.control_plane import queue_control_plane_steer

    result = queue_control_plane_steer(
        agent=None,
        session_id=getattr(args, "session_id", None),
        text=getattr(args, "message", ""),
    )
    print(json_module.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "queued" else 1


def _cmd_recovery_prompt(args) -> int:
    from agent.request_watchdog import (
        RECOVERY_RECOMMENDED_ACTION,
        build_bounded_recovery_prompt,
        latest_recoverable_turn_state,
    )

    packet = latest_recoverable_turn_state(session_id=getattr(args, "session", None) or "")
    if packet is None:
        print("No watchdog recovery state found.")
        return 1
    prompt = build_bounded_recovery_prompt(packet)
    if getattr(args, "json", False):
        print(
            json_module.dumps(
                {
                    "prompt": prompt,
                    "recommended_action": RECOVERY_RECOMMENDED_ACTION,
                    "session_id": packet.get("session_id", ""),
                    "status": (packet.get("status") or {}).get("status", "unknown"),
                },
                sort_keys=True,
            )
        )
        return 0
    print(
        "recovery: "
        f"status={(packet.get('status') or {}).get('status', 'unknown')} "
        f"session_id={packet.get('session_id', '')} "
        f"recommended_action={RECOVERY_RECOMMENDED_ACTION}"
    )
    if int(packet.get("queued_steer_count") or 0) > 0:
        print(
            "queued-steer: queued steer cannot land until the active request exits "
            "or reaches a tool boundary"
        )
    print(prompt)
    return 0


def _cmd_recovery_finalization_prompt(args) -> int:
    from hermes_cli.closeout_state import latest_closeout_state
    from hermes_cli.finalization_mode import (
        FINALIZATION_RECOMMENDED_ACTION,
        build_compact_finalization_prompt,
    )

    packet = latest_closeout_state(
        session_id=getattr(args, "session", None),
        task_id=getattr(args, "task_id", None),
    )
    if packet is None:
        print("No closeout state found.")
        return 1
    prompt = str(packet.get("compact_finalization_prompt") or build_compact_finalization_prompt(packet))
    out = getattr(args, "out", None)
    if out:
        path = Path(out)
        if path.parent and str(path.parent) != ".":
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompt, encoding="utf-8")
        print(
            "finalization-prompt: "
            f"recommended_action={FINALIZATION_RECOMMENDED_ACTION} "
            f"wrote={path}"
        )
        return 0
    print(prompt)
    return 0


def _cmd_closeout_status(args) -> int:
    from hermes_cli.closeout_state import latest_closeout_state

    packet = latest_closeout_state(
        session_id=getattr(args, "session", None),
        task_id=getattr(args, "task_id", None),
    )
    if packet is None:
        print("No closeout state found.")
        return 1
    if getattr(args, "json", False):
        print(json_module.dumps(packet, sort_keys=True))
        return 0
    print(
        "closeout: "
        f"status={packet.get('status', 'unknown')} "
        f"session_id={packet.get('session_id', '')} "
        f"latest_session_id={packet.get('latest_session_id') or packet.get('session_id', '')} "
        f"merge_status={packet.get('merge_status', '')} "
        f"ci_status={packet.get('ci_status', '')}"
    )
    reasons = packet.get("closeout_reasons") or []
    if reasons:
        print("  reasons=" + ",".join(str(reason) for reason in reasons))
    return 0


def _cmd_closeout_resume_command(args) -> int:
    from hermes_cli.closeout_state import (
        build_closeout_resume_command,
        latest_closeout_state,
    )

    packet = latest_closeout_state(
        session_id=getattr(args, "session", None),
        task_id=getattr(args, "task_id", None),
    )
    if packet is None:
        print("No closeout state found.")
        return 1
    print(
        build_closeout_resume_command(
            packet,
            hermes_command=getattr(args, "hermes_command", None) or "hermes",
        )
    )
    return 0


def build_parser(subparsers) -> argparse.ArgumentParser:
    runtime_parser = subparsers.add_parser(
        "runtime",
        help="Inspect and repair local Hermes runtime state",
    )
    runtime_parser.set_defaults(func=lambda _args: (runtime_parser.print_help(), 0)[1])
    runtime_sub = runtime_parser.add_subparsers(dest="runtime_command")

    active_parser = runtime_sub.add_parser(
        "active-sessions",
        help="Inspect and repair active-session leases",
    )
    active_parser.set_defaults(func=lambda _args: (active_parser.print_help(), 0)[1])
    active_sub = active_parser.add_subparsers(dest="active_sessions_command")

    status = active_sub.add_parser(
        "status",
        help="Show value-free active-session owner status",
    )
    status.set_defaults(func=_cmd_active_sessions_status)

    diagnose = active_sub.add_parser(
        "diagnose",
        help="Show value-free active-session diagnostics without repairing leases",
    )
    diagnose.add_argument(
        "--no-lock",
        action="store_true",
        help="Read the registry without taking the metadata lock; may be stale",
    )
    diagnose.add_argument("--json", action="store_true", help="Emit JSON")
    diagnose.set_defaults(func=_cmd_active_sessions_diagnose)

    repair = active_sub.add_parser(
        "repair",
        help="Repair stale active-session leases",
    )
    repair.add_argument(
        "--stale-only",
        action="store_true",
        help="Only remove leases whose owning process is proven dead",
    )
    repair.add_argument(
        "--session-id",
        help="Limit repair to one active-session id",
    )
    repair.set_defaults(func=_cmd_active_sessions_repair)

    control_parser = runtime_sub.add_parser(
        "control-plane",
        help="Inspect value-free local control-plane health",
    )
    control_parser.set_defaults(func=lambda _args: (control_parser.print_help(), 0)[1])
    control_sub = control_parser.add_subparsers(dest="control_plane_command")

    control_status = control_sub.add_parser(
        "status",
        help="Show value-free control-plane health and restart guidance",
    )
    control_status.add_argument("--session-id", help="Limit status to one session")
    control_status.add_argument(
        "--ws-probe-url",
        help="Optional WebSocket URL for a fresh lightweight probe",
    )
    control_status.add_argument("--json", action="store_true", help="Emit JSON")
    control_status.set_defaults(func=_cmd_control_plane_status)

    control_steer = control_sub.add_parser(
        "steer",
        help="Queue a value-free steer when a live agent or CLI boundary supports it",
    )
    control_steer.add_argument("--session-id", required=True)
    control_steer.add_argument("--message", required=True)
    control_steer.set_defaults(func=_cmd_control_plane_steer)

    recovery_parser = runtime_sub.add_parser(
        "recovery",
        help="Print bounded recovery actions for high-context request watchdog state",
    )
    recovery_parser.set_defaults(func=lambda _args: (recovery_parser.print_help(), 0)[1])
    recovery_sub = recovery_parser.add_subparsers(dest="recovery_command")

    recovery_prompt = recovery_sub.add_parser(
        "prompt",
        help="Print a bounded recovery prompt for the latest watchdog packet",
    )
    recovery_prompt.add_argument("--session", "--session-id", dest="session", required=True)
    recovery_prompt.add_argument("--json", action="store_true")
    recovery_prompt.set_defaults(func=_cmd_recovery_prompt)

    recovery_finalization = recovery_sub.add_parser(
        "finalization-prompt",
        help="Print a compact finalization prompt from closeout state",
    )
    recovery_finalization.add_argument("--session", "--session-id", dest="session", required=True)
    recovery_finalization.add_argument("--task-id", default=None)
    recovery_finalization.add_argument(
        "--out",
        help="Write the prompt to a file instead of stdout",
    )
    recovery_finalization.set_defaults(func=_cmd_recovery_finalization_prompt)

    closeout_parser = runtime_sub.add_parser(
        "closeout",
        help="Inspect closeout resume state for unfinished runs",
    )
    closeout_parser.set_defaults(func=lambda _args: (closeout_parser.print_help(), 0)[1])
    closeout_sub = closeout_parser.add_subparsers(dest="closeout_command")

    closeout_status = closeout_sub.add_parser(
        "status",
        help="Show latest closeout state for a session",
    )
    closeout_status.add_argument("--session", "--session-id", dest="session", required=True)
    closeout_status.add_argument("--task-id", default=None)
    closeout_status.add_argument("--json", action="store_true")
    closeout_status.set_defaults(func=_cmd_closeout_status)

    closeout_resume = closeout_sub.add_parser(
        "resume-command",
        help="Print a bounded resume command for the latest closeout state",
    )
    closeout_resume.add_argument("--session", "--session-id", dest="session", required=True)
    closeout_resume.add_argument("--task-id", default=None)
    closeout_resume.add_argument("--hermes-command", default="hermes")
    closeout_resume.set_defaults(func=_cmd_closeout_resume_command)
    return runtime_parser

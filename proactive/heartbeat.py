from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from obsidian.audit_log import ObsidianAdapter, append_audit_log
from proactive.progress import (
    mark_emitted,
    now_utc,
    parse_iso,
    read_state,
    scan_progress,
    should_emit,
    write_state,
)
from proactive.standing_orders import StandingOrder, load_standing_orders
from proactive.tool_policy import ToolPolicy, decide_action, load_tool_policy

SILENT = "[SILENT]"


@dataclass(frozen=True)
class HeartbeatConfig:
    obsidian_vault: Path
    standing_orders_path: Path | None = None
    tool_policy_path: Path | None = None
    standing_orders: list[StandingOrder] = field(default_factory=list)
    tool_policy: ToolPolicy | None = None
    proactive_progress_enabled: bool = True
    progress_report_interval_minutes: int = 60
    waiting_input_reminder_minutes: int = 60
    now: datetime | None = None


@dataclass(frozen=True)
class HeartbeatResult:
    output: str
    notification_payload: dict[str, Any] | None
    audit_path: Path
    anomalies: list[str]


def _has_open_checkbox(path: Path) -> bool:
    try:
        return "- [ ]" in path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False


def _scan_obsidian(vault: Path) -> list[str]:
    anomalies: list[str] = []
    watched = [
        ("open task", vault / "System" / "Tasks"),
        ("decision", vault / "System" / "Decision Inbox"),
        ("SOP update", vault / "System" / "SOP Updates"),
    ]
    for label, folder in watched:
        if not folder.exists():
            continue
        for md_file in folder.glob("*.md"):
            if _has_open_checkbox(md_file):
                anomalies.append(f"{label}: {md_file.name}")
    return anomalies


def _scan_cron_failures() -> list[str]:
    try:
        from cron.jobs import list_jobs
    except Exception:
        return []
    anomalies: list[str] = []
    for job in list_jobs(include_disabled=True):
        if job.get("last_status") == "failed" or job.get("last_error"):
            anomalies.append(f"cron failure: {job.get('name') or job.get('id')}")
    return anomalies


def run_heartbeat(config: HeartbeatConfig) -> HeartbeatResult:
    vault = Path(config.obsidian_vault)
    current_time = config.now or now_utc()
    policy = config.tool_policy or load_tool_policy(config.tool_policy_path)
    standing_orders = config.standing_orders or load_standing_orders(config.standing_orders_path)
    adapter = ObsidianAdapter(vault)

    actions = ["read_obsidian", "status_check", "write_audit_log"]
    if config.proactive_progress_enabled:
        actions.append("write_proactive_state")
    blocked = [
        action
        for action in actions
        if decide_action(action, policy).level.value == "DENY"
    ]
    anomalies = _scan_obsidian(vault) + _scan_cron_failures()
    if blocked:
        anomalies.append(f"policy denied: {', '.join(blocked)}")

    progress = scan_progress(vault) if config.proactive_progress_enabled else None
    state = read_state(vault) if config.proactive_progress_enabled else {}
    proactive_notes: list[str] = []
    state_changed = False

    if progress:
        if progress.stuck_or_failed:
            anomalies.extend(
                f"delegated task {item.status}: {item.id} {item.summary}"
                for item in progress.stuck_or_failed
            )
        overdue_waiting_for_kj = [
            item
            for item in progress.waiting_for_kj
            if _is_overdue(item.due_at, current_time)
        ]
        if overdue_waiting_for_kj and should_emit(
            state,
            "last_waiting_input_reminder_at",
            interval_minutes=config.waiting_input_reminder_minutes,
            now=current_time,
        ):
            proactive_notes.append(
                "waiting for KJ input: "
                + "; ".join(
                    f"{item.id} - {item.next_action or item.summary}"
                    for item in overdue_waiting_for_kj
                )
            )
            state = mark_emitted(
                state, "last_waiting_input_reminder_at", now=current_time
            )
            state_changed = True
        elif not progress.waiting_for_kj and progress.active_items and should_emit(
            state,
            "last_progress_report_at",
            interval_minutes=config.progress_report_interval_minutes,
            now=current_time,
        ):
            proactive_notes.append(
                "active progress: "
                + "; ".join(
                    f"{item.kind}:{item.id} {item.status} - {item.summary}"
                    for item in progress.active_items[:5]
                )
            )
            state = mark_emitted(state, "last_progress_report_at", now=current_time)
            state_changed = True

    notified = bool(anomalies or proactive_notes)
    summary = "No proactive anomalies detected."
    if anomalies or proactive_notes:
        parts = []
        if anomalies:
            parts.append("anomalies: " + "; ".join(anomalies))
        if proactive_notes:
            parts.append("progress: " + "; ".join(proactive_notes))
        summary = "Proactive heartbeat found " + " | ".join(parts)

    if state_changed and not blocked:
        write_state(vault, state)

    audit_path = append_audit_log(
        adapter,
        trigger_type="heartbeat",
        decision="notify" if notified else "silent",
        risk_level="medium" if notified else "low",
        actions_taken=actions,
        tools_used=["obsidian_adapter", "cron.jobs"],
        result_summary=summary,
        follow_up="Review notification payload." if notified else "none",
        whether_user_was_notified=notified,
        extra={
            "standing_orders_loaded": len(standing_orders),
            "active_items": len(progress.active_items) if progress else 0,
            "waiting_for_kj": len(progress.waiting_for_kj) if progress else 0,
        },
    )

    if not anomalies and not proactive_notes:
        return HeartbeatResult(output=SILENT, notification_payload=None, audit_path=audit_path, anomalies=[])

    payload = {
        "trigger_type": "heartbeat",
        "risk_level": "medium",
        "summary": summary,
        "anomalies": anomalies,
        "progress": proactive_notes,
        "recommended_next_action": _recommended_next_action(anomalies, proactive_notes),
    }
    return HeartbeatResult(output=summary, notification_payload=payload, audit_path=audit_path, anomalies=anomalies)


def _recommended_next_action(anomalies: list[str], proactive_notes: list[str]) -> str:
    if proactive_notes and any("waiting for KJ input" in note for note in proactive_notes):
        return "Hermes should ask KJ whether the requested information is ready."
    if anomalies:
        return "Hermes should ask KJ for a decision before high-risk action."
    return "Hermes should send a concise progress update and continue monitoring."


def _is_overdue(due_at: str, now: datetime) -> bool:
    due = parse_iso(due_at)
    return due is not None and due <= now.astimezone(due.tzinfo)


def render_notification_payload(payload: dict[str, Any]) -> str:
    """Render a cron-safe human-readable heartbeat notification."""
    lines = ["Proactive Hermes heartbeat"]
    summary = str(payload.get("summary") or "").strip()
    if summary:
        lines.append(summary)
    anomalies = payload.get("anomalies") or []
    progress = payload.get("progress") or []
    if anomalies:
        lines.append("Anomalies:")
        lines.extend(f"- {item}" for item in anomalies)
    if progress:
        lines.append("Progress:")
        lines.extend(f"- {item}" for item in progress)
    next_action = str(payload.get("recommended_next_action") or "").strip()
    if next_action:
        lines.append(f"Next: {next_action}")
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the Proactive Hermes heartbeat once.")
    parser.add_argument("--obsidian-vault", required=True)
    parser.add_argument("--standing-orders")
    parser.add_argument("--tool-policy")
    args = parser.parse_args()
    result = run_heartbeat(
        HeartbeatConfig(
            obsidian_vault=Path(args.obsidian_vault),
            standing_orders_path=Path(args.standing_orders) if args.standing_orders else None,
            tool_policy_path=Path(args.tool_policy) if args.tool_policy else None,
        )
    )
    print(result.output if result.notification_payload is None else render_notification_payload(result.notification_payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

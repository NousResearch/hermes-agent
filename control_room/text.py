"""Control Room plain-text renderer for CLI / gateway (CR-105).

Renders a ``ControlRoomSnapshot`` as compact, terminal-safe text matching the
plan's home contract. No full-screen tricks: usable over SSH and in
non-interactive transports.
"""

from __future__ import annotations

from typing import List

from .contract import (
    AttentionItem,
    AttentionSeverity,
    ControlRoomSnapshot,
    SystemSummary,
)

SECTION_ALIASES = {
    "home": "home",
    "needs-you": "needs-you",
    "needs_you": "needs-you",
    "agents": "agents",
    "tasks": "tasks",
    "messages": "messages",
    "system": "system",
}

VALID_SECTIONS = ("home", "needs-you", "agents", "tasks", "messages", "system")


def normalize_section(name: str) -> str:
    key = SECTION_ALIASES.get(name.strip().lower(), "home")
    return key if key in VALID_SECTIONS else "home"


def _severity_label(sev: AttentionSeverity) -> str:
    return {
        AttentionSeverity.critical: "!!",
        AttentionSeverity.error: "!",
        AttentionSeverity.warning: "~",
        AttentionSeverity.info: "·",
    }.get(sev, "·")


def _attention_line(item: AttentionItem) -> str:
    return f"{_severity_label(item.severity)} {item.title}"


def render_home(snapshot: ControlRoomSnapshot) -> str:
    c = snapshot.counts
    system_label = snapshot.system.state or "unknown"
    lines = [
        f"KENSEI › Control Room (profile: {snapshot.profile})",
        "",
        f"› Needs You     {c.needs_you}",
        f"  Agents        {c.agents_active} active",
        f"  Tasks         {c.tasks_running} running",
        f"  Messages      {c.messages_unread} unread",
        f"  System        {system_label}",
        "",
        "  + New Task      + New Message      + New Agent Run",
        "",
        "↑↓ navigate · Enter open · Esc back · Ctrl+P close   (/control <section>)",
    ]
    return "\n".join(lines)


def render_section(section: str, snapshot: ControlRoomSnapshot) -> str:
    section = normalize_section(section)
    if section == "home":
        return render_home(snapshot)
    if section == "needs-you":
        return _render_needs_you(snapshot)
    if section == "agents":
        return _render_agents(snapshot)
    if section == "tasks":
        return _render_tasks(snapshot)
    if section == "messages":
        return _render_messages(snapshot)
    if section == "system":
        return _render_system(snapshot)
    return render_home(snapshot)


def _render_needs_you(snapshot: ControlRoomSnapshot) -> str:
    lines = [f"Needs You — {snapshot.counts.needs_you} (profile: {snapshot.profile})", ""]
    urgent = [
        a for a in snapshot.attention
        if a.severity in (AttentionSeverity.critical, AttentionSeverity.error)
    ]
    if not urgent:
        lines.append("Nothing needs you right now.")
        return "\n".join(lines)
    for item in urgent:
        lines.append(f"  {_attention_line(item)}")
    lines.append("")
    lines.append("Actions: approve/deny · release/refuse · retry where valid · inspect")
    return "\n".join(lines)


def _render_agents(snapshot: ControlRoomSnapshot) -> str:
    lines = [f"Agents — {snapshot.counts.agents_active} active (profile: {snapshot.profile})", ""]
    if not snapshot.agents:
        lines.append("No active agents.")
    for a in snapshot.agents:
        detail = f" · {a.detail}" if a.detail else ""
        lines.append(f"  [{a.kind}] {a.name} · {a.status}{detail}")
    if snapshot.capabilities.process_control:
        lines.append("")
        lines.append("Actions: inspect · interrupt · pause · kill owned process")
    elif snapshot.agents:
        lines.append("")
        lines.append("Actions unavailable: process/delegation control not wired in this runtime.")
    return "\n".join(lines)


def _render_tasks(snapshot: ControlRoomSnapshot) -> str:
    lines = [f"Tasks — {snapshot.counts.tasks_running} running (profile: {snapshot.profile})", ""]
    if not snapshot.tasks:
        lines.append("No tasks.")
    for t in snapshot.tasks:
        owner = f" · {t.owner}" if t.owner else ""
        lines.append(f"  {t.id} · {t.title} · {t.state}{owner}")
    if snapshot.capabilities.kanban_actions:
        lines.append("")
        lines.append("Actions: inspect · comment · validated state moves (no generic retry)")
    elif snapshot.tasks:
        lines.append("")
        lines.append("Actions unavailable: kanban write route not wired in this runtime.")
    return "\n".join(lines)


def _render_messages(snapshot: ControlRoomSnapshot) -> str:
    lines = [f"Messages — {snapshot.counts.messages_unread} unread (profile: {snapshot.profile})", ""]
    if not snapshot.messages:
        lines.append("No held or queued peer messages.")
    for m in snapshot.messages:
        sender = f" from {m.sender}" if m.sender else ""
        lines.append(f"  [{m.state}] {m.title}{sender}")
    if snapshot.capabilities.peer_messages:
        lines.append("")
        lines.append("Actions: reply · release · refuse · accept/progress/complete request")
    elif snapshot.messages:
        lines.append("")
        lines.append("Actions unavailable: Hermes Peer plugin not wired in this runtime.")
    return "\n".join(lines)


def _render_system(snapshot: ControlRoomSnapshot) -> str:
    sys: SystemSummary = snapshot.system
    lines = [f"System — {sys.state} (profile: {snapshot.profile})", ""]
    if sys.detail:
        lines.append(f"  {sys.detail}")
    if sys.errors:
        lines.append("")
        lines.append("Errors:")
        for err in sys.errors:
            lines.append(f"  - {err}")
    lines.append("")
    lines.append("Actions: inspect · open detailed dashboard/system view (via existing route)")
    return "\n".join(lines)


def attention_status_line(snapshot: ControlRoomSnapshot) -> str:
    """Compact Claude-Code-style status strip for CLI/TUI chrome (CR-301)."""
    c = snapshot.counts
    bits: List[str] = []
    if c.needs_you:
        bits.append(f"● {c.needs_you} need you")
    if c.agents_active:
        bits.append(f"{c.agents_active} agents active")
    if c.messages_unread:
        bits.append(f"{c.messages_unread} unread")
    bits.append("Ctrl+P Control Room")
    return " · ".join(bits)

"""Static, read-only diagnostics command index."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class DiagnosticIndexEntry:
    name: str
    command: str
    area: str
    purpose: str
    checks: tuple[str, ...]
    read_only: bool
    approval_required: bool
    notes: str


DIAGNOSTICS_INDEX: tuple[DiagnosticIndexEntry, ...] = (
    DiagnosticIndexEntry(
        name="OR1 Start Gate",
        command="OR1 Start Gate",
        area="repo",
        purpose="Confirm the requested checkout, branch, HEAD, and worktree state before work starts.",
        checks=("pwd", "repo root", "branch", "HEAD", "git status"),
        read_only=True,
        approval_required=False,
        notes="Manual gate only; this index does not run the checks.",
    ),
    DiagnosticIndexEntry(
        name="reliability doctor",
        command="hermes reliability doctor",
        area="reliability",
        purpose="Inspect Hermes reliability stop-state and health signals.",
        checks=("runtime health", "stop-state signals", "known reliability advisories"),
        read_only=True,
        approval_required=False,
        notes="Read-only diagnostic variant.",
    ),
    DiagnosticIndexEntry(
        name="hermes status",
        command="hermes status",
        area="core",
        purpose="Show high-level Hermes component status.",
        checks=("configuration summary", "component status", "redacted environment state"),
        read_only=True,
        approval_required=False,
        notes="Use --deep only with an explicit approval slice.",
    ),
    DiagnosticIndexEntry(
        name="hermes status deep",
        command="hermes status --deep",
        area="core",
        purpose="Run deeper Hermes status checks.",
        checks=("deep component probes", "extended status checks"),
        read_only=False,
        approval_required=True,
        notes="Marked approval-required because deep checks may probe local services or take longer.",
    ),
    DiagnosticIndexEntry(
        name="hermes doctor",
        command="hermes doctor",
        area="core",
        purpose="Diagnose Hermes configuration and dependency issues.",
        checks=("config validity", "dependency availability", "security advisories"),
        read_only=True,
        approval_required=False,
        notes="Do not use --fix without explicit repair approval.",
    ),
    DiagnosticIndexEntry(
        name="hermes doctor fix",
        command="hermes doctor --fix",
        area="core",
        purpose="Attempt automatic repairs for doctor findings.",
        checks=("repair candidates", "configuration/dependency fixes"),
        read_only=False,
        approval_required=True,
        notes="Mutating repair variant; not part of read-only diagnostics.",
    ),
    DiagnosticIndexEntry(
        name="hermes gateway status",
        command="hermes gateway status",
        area="gateway",
        purpose="Show messaging gateway service status.",
        checks=("service unit state", "gateway process status", "recent gateway state"),
        read_only=True,
        approval_required=False,
        notes="Status only; no restart, install, stop, or repair action.",
    ),
    DiagnosticIndexEntry(
        name="hermes gateway status deep",
        command="hermes gateway status --deep",
        area="gateway",
        purpose="Run deeper gateway status inspection.",
        checks=("service details", "extended gateway status", "platform readiness"),
        read_only=False,
        approval_required=True,
        notes="Marked approval-required because deep status may perform live probes.",
    ),
    DiagnosticIndexEntry(
        name="hermes kanban diagnostics",
        command="hermes kanban diagnostics",
        area="kanban",
        purpose="Inspect kanban task diagnostics and distress signals.",
        checks=("blocked tasks", "repeated failures", "stale/stranded work signals"),
        read_only=True,
        approval_required=False,
        notes="Index entry only; this command is not executed here.",
    ),
    DiagnosticIndexEntry(
        name="hooks doctor",
        command="hermes hooks doctor",
        area="hooks",
        purpose="Inspect configured shell hooks for safety and drift issues.",
        checks=("exec bit", "allowlist state", "mtime drift", "JSON output validity"),
        read_only=False,
        approval_required=True,
        notes="Risky diagnostic: hook doctor can exercise configured hook scripts.",
    ),
)


def available_areas() -> tuple[str, ...]:
    return tuple(sorted({entry.area for entry in DIAGNOSTICS_INDEX}))


def _filter_entries(
    *,
    area: str | None = None,
    read_only: bool = False,
    approval_required: bool = False,
) -> tuple[DiagnosticIndexEntry, ...]:
    if area is not None and area not in available_areas():
        raise SystemExit(f"unknown diagnostics area: {area}")

    entries: Iterable[DiagnosticIndexEntry] = DIAGNOSTICS_INDEX
    if area is not None:
        entries = (entry for entry in entries if entry.area == area)
    if read_only:
        entries = (entry for entry in entries if entry.read_only)
    if approval_required:
        entries = (entry for entry in entries if entry.approval_required)
    return tuple(entries)


def _entry_to_dict(entry: DiagnosticIndexEntry) -> dict[str, object]:
    data = asdict(entry)
    data["checks"] = list(entry.checks)
    return data


def _print_text(entries: tuple[DiagnosticIndexEntry, ...]) -> None:
    print("Diagnostics Index")
    print("=================")
    for entry in entries:
        read_only = "yes" if entry.read_only else "no"
        approval = "yes" if entry.approval_required else "no"
        print()
        print(entry.command)
        print(f"  area: {entry.area}")
        print(f"  purpose: {entry.purpose}")
        print(f"  checks: {', '.join(entry.checks)}")
        print(f"  read-only: {read_only}")
        print(f"  approval-required: {approval}")
        print(f"  notes: {entry.notes}")


def diagnostics_index_command(args) -> None:
    entries = _filter_entries(
        area=getattr(args, "area", None),
        read_only=bool(getattr(args, "read_only", False)),
        approval_required=bool(getattr(args, "approval_required", False)),
    )
    if getattr(args, "json", False):
        print(json.dumps({
            "count": len(entries),
            "diagnostics": [_entry_to_dict(entry) for entry in entries],
        }, indent=2))
        return
    _print_text(entries)

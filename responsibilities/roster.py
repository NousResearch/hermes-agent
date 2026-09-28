"""Responsibility roster; employee contracts on native Hermes."""

from __future__ import annotations
from typing import Any, Mapping


from responsibilities.common import (
    _ROSTER_DESCRIPTION_LENGTH,
    validate_responsibility_name
)

def _roster_description(raw: str) -> str:
    description = raw.strip().strip("'\"")
    if len(description) > _ROSTER_DESCRIPTION_LENGTH:
        return description[: _ROSTER_DESCRIPTION_LENGTH - 3] + "..."
    return description

_RESPONSIBILITY_ROSTER_PREAMBLE = (
    "## Responsibilities\n"
    "These are your responsibilities — durable areas of the organization's "
    "work that you own. Each package is your own standing instruction set "
    "for its area — written by you, for the next run of you, which arrives "
    "with nothing but what the package carries: the assignment, the rules "
    "and workflows already settled, where the work stands. Carry the area "
    "on your own initiative rather than waiting to be asked, and author "
    "the package as proactively as you work it: what you learn reaches the "
    "next run only if you write it in.\n"
    "If the work at hand matches or is even partially relevant to one of "
    "these areas, you MUST open it (read its RESPONSIBILITY.md under "
    "{profile_home}/responsibilities/<name>/) on first touch in a conversation. "
    "Err on the side of opening — it is always better to have context you "
    "don't need than to miss critical steps, pitfalls, or established "
    "workflows. The files its duties name define how the work is done "
    "here: follow them even for work you already know, and update any "
    "found missing steps, wrong commands, or unrecorded pitfalls before "
    "finishing. Work that must continue after this conversation — checking "
    "back, waiting on someone, a recurring rhythm — belongs to a "
    "responsibility too: its schedules run it for you in the background.\n"
    "Scheduled runs and conversations share the package's STATE.md, and it "
    "has one purpose: the handoff to the next run. A line earns its place "
    "only by changing what the next run does — 'Waiting on Jane's signed "
    "contract since 2026-07-21; escalate 2026-07-31' ✓; a log of what runs "
    "did ✗; durable facts ✗ (they belong in the package's files). Rewrite "
    "it whenever your work changes the handoff, and prune in the same "
    "write: the moment a line no longer matters to the next run, it goes "
    "— sharper, never longer, dates absolute. Records too big for it live "
    "in Markdown files under state/, each referenced from STATE.md; "
    "finished ones move whole to archive/. Data files and machine "
    "artifacts belong on the drive, never in the package.\n"
    "When someone corrects or reshapes how an owned area is handled, find "
    "what produced the behavior they corrected — a charter line, a "
    "reference file, a schedule or webhook scope, a linked connection "
    "manual — and edit it at the source before the conversation ends: "
    "most corrections are edits to an existing line, not additions, and a "
    "correction absorbed around still-standing wording is repeated at them "
    "by the next run. A correction that recurs after being filed sat too "
    "low: move it up — into the duty line that forces the read, or a "
    "check a script enforces.\n"
    "Beyond state — creating a package, reshaping a charter, "
    "restructuring — read {guides_root}/responsibility-authoring/guide.md "
    "first."
)

_RESPONSIBILITY_ROSTER_TRAILER = (
    "Only proceed without opening a responsibility if genuinely none are "
    "relevant."
)

def render_responsibility_roster(snapshot: Mapping[str, Any]) -> str:
    """Render the frozen roster wrapped in its activation guidance."""

    raw_entries = snapshot.get("responsibilities", ())
    if not isinstance(raw_entries, (list, tuple)):
        return ""
    entries: list[tuple[str, str, str]] = []
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "")
        description = str(raw.get("description") or "")
        description = _roster_description(description)
        marker = ""
        schedule_errors = raw.get("schedule_errors")
        if isinstance(schedule_errors, Mapping) and schedule_errors:
            broken = ", ".join(sorted(str(key) for key in schedule_errors))
            marker = f" ⚠ broken schedule declarations: {broken}"
        webhook_errors = raw.get("webhook_errors")
        if isinstance(webhook_errors, Mapping) and webhook_errors:
            broken = ", ".join(sorted(str(key) for key in webhook_errors))
            marker += f" ⚠ broken webhook declarations: {broken}"
        stray_files = raw.get("stray_files")
        if isinstance(stray_files, (list, tuple)) and stray_files:
            listed = ", ".join(str(path) for path in stray_files)
            marker += f" ⚠ out-of-contract files: {listed}"
        if validate_responsibility_name(name) is None and description:
            entries.append((name, description, marker))
    if not entries:
        return ""
    # The roster shows the packages as the directory tree they are on disk;
    # responsibilities are managed purely with the generic file tools.
    lines = [
        _RESPONSIBILITY_ROSTER_PREAMBLE,
        "",
        "<available_responsibilities>",
        "{profile_home}/responsibilities/",
    ]
    ordered = sorted(entries)
    for position, (name, description, marker) in enumerate(ordered, start=1):
        branch = "└── " if position == len(ordered) else "├── "
        lines.append(f"{branch}{name}/ — {description}{marker}")
    lines.append("</available_responsibilities>")
    lines.append("")
    lines.append(_RESPONSIBILITY_ROSTER_TRAILER)
    return "\n".join(lines)

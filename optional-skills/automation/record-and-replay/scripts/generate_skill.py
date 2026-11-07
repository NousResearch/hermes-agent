#!/usr/bin/env python3
"""Vision-based skill generation from recording analysis.

Reads the analysis JSON from analyze_recording.py and generates a SKILL.md
with element descriptions (not pixel coordinates), verification criteria,
and error recovery instructions.

If a vision API is configured (Hermes auxiliary vision), uses vision_analyze
to understand screenshots. Otherwise, falls back to text-only analysis from
the AX tree + event log.

Usage:
    python3 generate_skill.py --analysis <analysis.json> [--recording-dir DIR]
                              [--dry-run] [--output-dir DIR] [--name NAME]

Output:
    A SKILL.md file written to the output directory (default:
    ~/.hermes/skills/automation/<name>/SKILL.md)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Vision helper ────────────────────────────────────────────────────────────

def check_vision_available() -> bool:
    """Check if Hermes auxiliary vision is configured."""
    try:
        result = subprocess.run(
            ["hermes", "config"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return "vision" in result.stdout.lower() and "provider" in result.stdout.lower()
    except Exception:
        pass
    return False


def vision_analyze_screenshot(image_path: str, question: str) -> str:
    """Use Hermes vision to analyze a screenshot.

    Falls back to empty string if vision is unavailable.
    """
    if not os.path.exists(image_path):
        return ""

    # Try using hermes chat with vision
    try:
        result = subprocess.run(
            ["hermes", "chat", "-q",
             "Analyze this screenshot: {} {}".format(image_path, question)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    return ""


def analyze_screenshot_with_vision(image_path: str, ax_tree: str, step_num: int) -> dict:
    """Analyze a screenshot to understand the UI state.

    Returns dict with: app_name, elements_visible, key_changes, description
    """
    question = (
        "What application is this? What UI elements are visible? "
        "What action is about to happen or just happened? "
        "Be concise — 2-3 sentences."
    )

    vision_desc = ""
    if os.path.exists(image_path):
        vision_desc = vision_analyze_screenshot(image_path, question)

    # Parse AX tree for element info (fallback/complement)
    elements = []
    if ax_tree:
        for line in ax_tree.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                elements.append(line)

    return {
        "vision_description": vision_desc,
        "ax_elements": elements[:20],  # top 20 elements
        "has_screenshot": os.path.exists(image_path),
    }


# ── SKILL.md generation ─────────────────────────────────────────────────────

def generate_skill_md(analysis: dict, recording_dir: Path, use_vision: bool,
                      skill_name: str) -> str:
    """Generate a complete SKILL.md from the analysis."""
    steps = analysis.get("steps", [])
    patterns = analysis.get("detected_patterns", [])
    suggested_name = analysis.get("suggested_skill_name", skill_name)
    metadata = analysis.get("metadata", {})

    now = datetime.now().strftime("%Y-%m-%d")

    # Determine platform
    platform = metadata.get("platform", "macos")

    # Build step descriptions
    step_sections = []
    total_duration = 0.0

    for i, step in enumerate(steps):
        step_num = i + 1
        app = step.get("app", "Unknown")
        interactions = step.get("interactions", [])
        screenshots = step.get("screenshots", [])
        ax_trees = step.get("ax_trees", [])
        clipboard = step.get("clipboard_events", [])
        window_events = step.get("window_events", [])
        retries = step.get("retries", [])
        signature = step.get("signature", [])
        duration = step.get("duration", 0)
        total_duration += duration

        # Analyze screenshots if vision available
        vision_info = None
        if use_vision and screenshots:
            before_shot = str(recording_dir / screenshots[0]) if screenshots else ""
            ax_content = ""
            if ax_trees:
                ax_path = recording_dir / ax_trees[0]
                if ax_path.exists():
                    ax_content = ax_path.read_text(errors="replace")
            vision_info = analyze_screenshot_with_vision(before_shot, ax_content, step_num)

        # Build step description
        lines = []
        lines.append("### Step {}: {}".format(step_num, _describe_step(signature, interactions)))

        # App context
        if app and app != "Unknown":
            lines.append("- App: {}".format(app))

        # Vision description
        if vision_info and vision_info.get("vision_description"):
            lines.append("- UI state: {}".format(vision_info["vision_description"]))
        elif vision_info and vision_info.get("ax_elements"):
            lines.append("- Visible elements: {}".format(
                ", ".join(vision_info["ax_elements"][:5])))

        # Actions
        lines.append("")
        lines.append("**Actions:**")
        for inter in interactions:
            action = inter.get("action", "")
            pos = inter.get("position")
            char = inter.get("character")
            time_str = "{:.1f}s".format(inter.get("time", 0))
            if pos:
                lines.append("  - [{}] {} at ({}, {})".format(
                    time_str, action, pos[0], pos[1]))
            elif char:
                lines.append("  - [{}] {} → '{}'".format(time_str, action, char))
            else:
                lines.append("  - [{}] {}".format(time_str, action))

        # Clipboard events
        for clip in clipboard:
            preview = clip.get("preview", "")
            lines.append("  - [{:.1f}s] clipboard copy: '{}'".format(
                clip.get("time", 0), preview))

        # Window events
        for we in window_events:
            lines.append("  - [{:.1f}s] {} ({})".format(
                we.get("time", 0), we.get("action", ""), we.get("window", "")))

        # Verification criteria
        lines.append("")
        lines.append("**Verify after step:**")
        if vision_info and vision_info.get("vision_description"):
            lines.append("- {}".format(vision_info["vision_description"]))
        elif screenshots:
            lines.append("- Screenshot {} should show expected state".format(screenshots[-1]))
        else:
            lines.append("- Application should respond to the action")

        # Error recovery
        if retries:
            lines.append("")
            lines.append("**Error recovery:**")
            lines.append("- If the first click misses, the correct target is at ({}, {})".format(
                retries[0]["second_click"][0], retries[0]["second_click"][1]))

        # Replay instructions
        lines.append("")
        lines.append("**Replay:**")
        lines.append("```")
        if "click" in signature:
            lines.append('computer_use(action="capture", mode="som", app="{}")'.format(app))
            lines.append('# Find the target element by description, then:')
            lines.append('computer_use(action="click", element=<N>)')
        if "type" in signature:
            typed_chars = [i.get("character", "") for i in interactions
                           if i.get("action", "").startswith("key:") and i.get("character")]
            if typed_chars:
                lines.append('computer_use(action="type", text="{}")'.format(
                    "".join(typed_chars)))
        if "scroll" in signature:
            lines.append('computer_use(action="scroll", direction="down", amount=3)')
        if "drag" in signature:
            lines.append('computer_use(action="drag", from_element=<N>, to_element=<N>)')
        lines.append("# Verify:")
        lines.append('computer_use(action="capture", mode="som")')
        lines.append("```")

        step_sections.append("\n".join(lines))

    # Pattern notes
    pattern_notes = []
    if patterns:
        pattern_notes.append("### Detected Patterns (Loops)")
        for p in patterns:
            pattern_notes.append("- {}: {} (steps {}–{})".format(
                p["description"], " → ".join(p["pattern"]),
                p["step_range"][0], p["step_range"][1]))
        pattern_notes.append("")
        pattern_notes.append("If this workflow repeats a sub-task, implement the loop body")
        pattern_notes.append("as a function and call it N times instead of repeating steps.")
        pattern_notes.append("")

    # Build full SKILL.md
    skill_md = """---
name: {name}
description: Auto-generated from recording on {date}.
version: 1.0.0
author: generated by record-and-replay
license: MIT
platforms: [{platform}]
metadata:
  hermes:
    tags: [automation, recorded]
    category: automation
    created_by: agent
    source_recording: {recording}
    analysis_method: {method}
---

# {title}

Auto-generated from a recording on {date}. This skill replays a workflow
that was captured by watching the user perform it once.

## When to Use

Replay this workflow when the user asks to {trigger}.

## Prerequisites

- `computer_use` toolset enabled (macOS) or `browser_*` tools (web tasks)
- Platform: {platform}
{app_prereqs}

## Procedure

{steps}

{patterns}

## Verification

After all steps, capture a screenshot and verify:
- The expected final state is reached
- No error dialogs or unexpected popups
- The workflow completed within ~{duration:.0f} seconds

## Pitfalls

- Element indices may change between runs — always re-capture before clicking
- Timing-sensitive UI may need `computer_use(action="wait", seconds=1.0)` between steps
- If the app state is fundamentally different, stop and ask the user
- Window positions may differ — use element descriptions, not pixel coordinates
- Clipboard content from the recording is NOT replayed — provide fresh content
"""

    return skill_md.format(
        name=skill_name,
        date=now,
        platform=platform,
        recording=str(recording_dir),
        method="vision + AX tree" if use_vision else "AX tree + event log (no vision)",
        title=_title_case(skill_name.replace("-", " ")),
        trigger=_describe_trigger(steps),
        app_prereqs=_app_prereqs(steps),
        steps="\n\n".join(step_sections),
        patterns="\n".join(pattern_notes),
        duration=total_duration,
    )


def _describe_step(signature: list, interactions: list) -> str:
    """Generate a human-readable step description from its signature."""
    if not signature:
        return "Wait/observe"
    parts = []
    for s in signature:
        if s == "click":
            parts.append("click")
        elif s == "type":
            parts.append("type")
        elif s == "scroll":
            parts.append("scroll")
        elif s == "drag":
            parts.append("drag")
        elif s == "app_switch":
            parts.append("switch app")
        elif s == "clipboard":
            parts.append("copy to clipboard")
        elif s == "window_resize":
            parts.append("resize window")
        elif s == "window_move":
            parts.append("move window")
        elif s == "window_minimize":
            parts.append("minimize window")
    return " + ".join(parts[:4])


def _describe_trigger(steps: list) -> str:
    """Generate a trigger description from the first step."""
    if not steps:
        return "perform the recorded workflow"
    first = steps[0]
    app = first.get("app", "the application")
    sig = first.get("signature", [])
    if "app_switch" in sig:
        return "switch to {} and perform the workflow".format(app)
    if "click" in sig:
        return "interact with {} as shown in the recording".format(app)
    return "perform the {} workflow".format(app)


def _app_prereqs(steps: list) -> str:
    """Extract unique app prerequisites."""
    apps = set()
    for s in steps:
        app = s.get("app")
        if app and app != "Unknown":
            apps.add(app)
    if not apps:
        return "- No specific app requirements"
    return "\n".join("- {}".format(a) for a in sorted(apps))


def _title_case(s: str) -> str:
    """Convert a slug to title case."""
    return " ".join(w.capitalize() for w in s.split())


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a SKILL.md from recording analysis."
    )
    parser.add_argument("--analysis", type=str, required=True,
                        help="Path to analysis JSON from analyze_recording.py")
    parser.add_argument("--recording-dir", type=str, default=None,
                        help="Recording directory (for screenshot access)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview the generated skill without saving")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ~/.hermes/skills/automation/<name>/)")
    parser.add_argument("--name", type=str, default=None,
                        help="Skill name (default: auto-generated from analysis)")
    parser.add_argument("--no-vision", action="store_true",
                        help="Skip vision analysis, use text-only mode")
    args = parser.parse_args()

    # Load analysis
    analysis_path = Path(args.analysis)
    if not analysis_path.exists():
        print("ERROR: Analysis file not found: {}".format(analysis_path), file=sys.stderr)
        sys.exit(1)

    with open(analysis_path) as f:
        analysis = json.load(f)

    # Determine skill name
    skill_name = args.name or analysis.get("suggested_skill_name", "recorded-workflow")

    # Determine recording dir
    recording_dir = Path(args.recording_dir) if args.recording_dir else Path(analysis.get("recording_dir", "."))

    # Check vision availability
    use_vision = not args.no_vision and not args.dry_run and check_vision_available()
    if use_vision:
        print("Vision provider detected — using vision analysis", file=sys.stderr)
    else:
        print("No vision provider — using text-only analysis (AX tree + event log)", file=sys.stderr)

    # Generate SKILL.md
    skill_md = generate_skill_md(analysis, recording_dir, use_vision, skill_name)

    # Output
    if args.dry_run:
        print(skill_md)
        print("\n--- Dry run — skill not saved ---", file=sys.stderr)
    else:
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
            output_dir = Path(hermes_home) / "skills" / "automation" / skill_name

        output_dir.mkdir(parents=True, exist_ok=True)
        skill_path = output_dir / "SKILL.md"
        with open(skill_path, "w") as f:
            f.write(skill_md)

        print("Skill written to: {}".format(skill_path), file=sys.stderr)
        print("Skill name: {}".format(skill_name), file=sys.stderr)


if __name__ == "__main__":
    main()

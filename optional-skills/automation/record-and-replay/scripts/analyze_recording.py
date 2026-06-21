#!/usr/bin/env python3
"""Analyze a recording and produce a replayable skill summary.

Reads the events.jsonl + screenshots from a recording directory and produces
a structured JSON summary that the vision model can use to generate a SKILL.md.

Usage:
    python3 analyze_recording.py /path/to/recording

Output:
    - Prints a JSON summary to stdout
    - Identifies logical steps (grouped by 2s+ pauses)
    - Pairs screenshots with events
    - Extracts the "story" of what happened
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_events(events_file: Path) -> list[dict]:
    """Load events from JSONL file."""
    events = []
    with open(events_file) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def group_into_steps(events: list[dict], pause_threshold: float = 2.0) -> list[dict]:
    """Group events into logical steps based on pauses.

    A new step starts when:
    - There's a pause > pause_threshold seconds between events
    - The frontmost app changes
    - A snapshot is taken (boundary marker)
    """
    steps = []
    current_step = {
        "step_number": 1,
        "events": [],
        "start_time": 0,
        "screenshots": [],
        "app": None,
    }
    last_time = 0

    for evt in events:
        t = evt.get("timestamp", 0)

        # Check for app switch
        if evt.get("type") == "app_switch":
            if current_step["events"]:
                steps.append(current_step)
            current_step = {
                "step_number": len(steps) + 1,
                "events": [],
                "start_time": t,
                "screenshots": [],
                "app": evt.get("to", "unknown"),
            }
            current_step["events"].append(evt)
            last_time = t
            continue

        # Check for snapshot (marks a boundary)
        if evt.get("type") == "snapshot":
            screenshot = evt.get("screenshot")
            if screenshot:
                current_step["screenshots"].append(screenshot)
            current_step["app"] = evt.get("frontmost_app", {}).get("name", current_step["app"])
            current_step["events"].append(evt)
            last_time = t
            continue

        # Check for pause (new step)
        if t - last_time > pause_threshold and current_step["events"]:
            # Only break on pause if we have actual interaction events
            has_interaction = any(
                e.get("type") in ("mouse_down", "key_down", "scroll", "mouse_dragged")
                for e in current_step["events"]
            )
            if has_interaction:
                steps.append(current_step)
                current_step = {
                    "step_number": len(steps) + 1,
                    "events": [],
                    "start_time": t,
                    "screenshots": current_step["screenshots"][-1:] if current_step["screenshots"] else [],
                    "app": current_step["app"],
                }

        current_step["events"].append(evt)
        last_time = t

    if current_step["events"]:
        steps.append(current_step)

    return steps


def summarize_step(step: dict) -> dict:
    """Produce a human-readable summary of a step."""
    interactions = []
    for evt in step["events"]:
        t = evt.get("timestamp", 0)
        evt_type = evt.get("type", "")

        if evt_type == "mouse_down":
            pos = evt.get("position", [0, 0])
            button = evt.get("button", "left")
            mods = "+".join(evt.get("modifiers", []))
            mod_str = f" [{mods}]" if mods else ""
            interactions.append({
                "time": round(t, 2),
                "action": f"click {button}{mod_str}",
                "position": [round(pos[0]), round(pos[1])],
            })
        elif evt_type == "key_down":
            key = evt.get("key", "unknown")
            char = evt.get("character", "")
            mods = "+".join(evt.get("modifiers", []))
            if mods and key in ("shift", "control", "option", "command", "fn",
                                "caps_lock", "right_shift", "right_option", "right_control"):
                # This is a modifier press itself, skip
                continue
            mod_str = f"{mods}+" if mods else ""
            interactions.append({
                "time": round(t, 2),
                "action": f"key: {mod_str}{key}",
                "character": char if char and char != key else None,
            })
        elif evt_type == "scroll":
            direction = evt.get("direction", "none")
            delta = evt.get("delta_y", 0)
            interactions.append({
                "time": round(t, 2),
                "action": f"scroll {direction} (delta: {delta})",
                "position": [round(evt.get("position", [0, 0])[0]), round(evt.get("position", [0, 0])[1])],
            })
        elif evt_type == "mouse_dragged":
            pos = evt.get("position", [0, 0])
            interactions.append({
                "time": round(t, 2),
                "action": "drag",
                "position": [round(pos[0]), round(pos[1])],
            })
        elif evt_type == "app_switch":
            interactions.append({
                "time": round(t, 2),
                "action": f"app switch: {evt.get('from', '?')} → {evt.get('to', '?')}",
            })

    return {
        "step_number": step["step_number"],
        "app": step["app"],
        "start_time": round(step["start_time"], 2),
        "interaction_count": len(interactions),
        "screenshots": step["screenshots"],
        "interactions": interactions,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_recording.py <recording_dir>", file=sys.stderr)
        sys.exit(1)

    recording_dir = Path(sys.argv[1])
    events_file = recording_dir / "events" / "events.jsonl"
    metadata_file = recording_dir / "metadata.json"

    if not events_file.exists():
        print(f"ERROR: No events file at {events_file}", file=sys.stderr)
        sys.exit(1)

    # Load metadata
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)

    # Load and analyze events
    events = load_events(events_file)
    steps = group_into_steps(events)
    summaries = [summarize_step(s) for s in steps]

    # Build the final output
    output = {
        "recording_dir": str(recording_dir),
        "metadata": metadata,
        "total_steps": len(summaries),
        "steps": summaries,
    }

    # Print to stdout for the agent to read
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

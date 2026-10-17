#!/usr/bin/env python3
"""Self-Learning Mode — agent records its own computer_use actions, iterates
to find the most efficient path, and saves the optimized version as a skill.

The self-learning loop works like this:

    Attempt 1: Agent fumbles through a task (slow, wrong clicks, retries)
               → Recording captured → Analyzed → Metrics computed

    Attempt 2: Agent tries again (faster, learned from attempt 1)
               → Recording captured → Analyzed → Metrics computed

    Attempt 3: Agent optimizes (keyboard shortcuts, fewer captures)
               → Recording captured → Analyzed → Metrics computed

    Compare: self_learn.py ranks all attempts by efficiency score
    Generate: Best attempt → generate_skill.py → SKILL.md
    Verify: Replay the skill to confirm it works

The agent drives the loop using its tools (computer_use, terminal). This
script handles the data processing: analysis, comparison, reporting, and
skill generation.

Usage:
    # 1. Initialize a self-learning session
    python3 self_learn.py --init bambu-slice \\
        --description "Open Bambu Slicer, import 3MF, choose filament, slice, save G-code"

    # 2. After each attempt (agent records via record_workflow.py, then):
    python3 self_learn.py --analyze ~/.hermes/recordings/self-learn/bambu-slice \\
        --attempt 1 --recording ~/.hermes/recordings/bambu-slice-attempt-1

    # 3. After all attempts, compare and pick the best
    python3 self_learn.py --compare ~/.hermes/recordings/self-learn/bambu-slice

    # 4. Generate skill from the best attempt
    python3 self_learn.py --generate ~/.hermes/recordings/self-learn/bambu-slice

    # 5. View the comparison report
    python3 self_learn.py --report ~/.hermes/recordings/self-learn/bambu-slice

Session directory structure:
    ~/.hermes/recordings/self-learn/<task-name>/
    ├── session.json              # Task name, description, created, attempts list
    ├── attempt-1/
    │   ├── analysis.json         # Full analysis from analyze_recording.py
    │   ├── metrics.json          # Computed efficiency metrics
    │   └── recording-path        # Path to the original recording
    ├── attempt-2/
    │   └── ...
    ├── comparison.json           # Ranked attempts + best pick
    ├── report.md                 # Human-readable comparison report
    └── best-skill/
        └── SKILL.md              # Generated skill from best attempt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ── Metric weights (lower score = more efficient) ────────────────────────────

WEIGHTS = {
    "duration": 0.20,      # Wall time (seconds)
    "action_count": 0.30,  # Total interaction events
    "retry_count": 0.30,   # Detected retries (mis-clicks)
    "step_count": 0.20,    # Logical steps detected
}

# Normalize raw values to 0-10 scale for comparison
# These are soft caps — values above the cap are clamped to 10
NORMALIZATION_CAPS = {
    "duration": 120.0,      # 2 minutes = score 10
    "action_count": 50.0,   # 50 actions = score 10
    "retry_count": 10.0,    # 10 retries = score 10
    "step_count": 20.0,     # 20 steps = score 10
}


# ── Session management ───────────────────────────────────────────────────────

def get_session_dir(task_name: str) -> Path:
    """Get the session directory for a task name."""
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return Path(hermes_home) / "recordings" / "self-learn" / task_name


def init_session(task_name: str, description: str = "") -> Path:
    """Initialize a new self-learning session."""
    session_dir = get_session_dir(task_name)
    if session_dir.exists():
        print(f"Session already exists: {session_dir}", file=sys.stderr)
        print("Existing attempts will be preserved.", file=sys.stderr)
    else:
        session_dir.mkdir(parents=True, exist_ok=True)

    session = {
        "task_name": task_name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "attempts": [],
    }

    session_file = session_dir / "session.json"
    if not session_file.exists():
        with open(session_file, "w") as f:
            json.dump(session, f, indent=2)

    print(f"✅ Self-learning session initialized: {task_name}", file=sys.stderr)
    print(f"   Directory: {session_dir}", file=sys.stderr)
    if description:
        print(f"   Task: {description}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"Next steps:", file=sys.stderr)
    print(f"  1. Start recording: python3 record_workflow.py --output-dir <recording-dir>", file=sys.stderr)
    print(f"  2. Perform the task via computer_use", file=sys.stderr)
    print(f"  3. Stop recording: touch <recording-dir>/.stop", file=sys.stderr)
    print(f"  4. Analyze: python3 self_learn.py --analyze {session_dir} --attempt 1 --recording <recording-dir>", file=sys.stderr)
    return session_dir


def load_session(session_dir: Path) -> dict:
    """Load session.json."""
    session_file = session_dir / "session.json"
    if not session_file.exists():
        print(f"ERROR: No session.json in {session_dir}", file=sys.stderr)
        sys.exit(1)
    with open(session_file) as f:
        return json.load(f)


def save_session(session_dir: Path, session: dict):
    """Save session.json."""
    session_file = session_dir / "session.json"
    with open(session_file, "w") as f:
        json.dump(session, f, indent=2)


# ── Attempt analysis ─────────────────────────────────────────────────────────

def analyze_attempt(session_dir: Path, attempt_num: int, recording_dir: str,
                    script_dir: str) -> dict:
    """Analyze a single attempt by running analyze_recording.py on it."""
    attempt_dir = session_dir / f"attempt-{attempt_num}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    # Save the recording path for reference
    (attempt_dir / "recording-path").write_text(recording_dir)

    # Run analyze_recording.py
    analyze_script = os.path.join(script_dir, "analyze_recording.py")
    analysis_file = attempt_dir / "analysis.json"

    print(f"Analyzing attempt {attempt_num}...", file=sys.stderr)
    result = subprocess.run(
        [
            sys.executable, analyze_script,
            recording_dir,
            "--output", str(analysis_file),
        ],
        capture_output=True, text=True, timeout=30,
    )

    if result.returncode != 0:
        print(f"WARNING: analyze_recording.py exited with code {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)

    # Load the analysis
    analysis = {}
    if analysis_file.exists():
        with open(analysis_file) as f:
            analysis = json.load(f)
    else:
        # Fall back to stdout
        try:
            analysis = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            analysis = {"error": "Failed to parse analysis output"}

    # Compute metrics
    metrics = compute_metrics(analysis)

    # Save metrics
    metrics_file = attempt_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Update session
    session = load_session(session_dir)
    attempts = session.get("attempts", [])

    # Replace if already exists
    attempt_entry = {
        "number": attempt_num,
        "recording_dir": recording_dir,
        "analyzed_at": datetime.now().isoformat(),
        "metrics": metrics,
    }

    # Replace existing attempt or append
    found = False
    for i, a in enumerate(attempts):
        if a.get("number") == attempt_num:
            attempts[i] = attempt_entry
            found = True
            break
    if not found:
        attempts.append(attempt_entry)
        attempts.sort(key=lambda a: a.get("number", 0))

    session["attempts"] = attempts
    save_session(session_dir, session)

    print(f"✅ Attempt {attempt_num} analyzed:", file=sys.stderr)
    print(f"   Duration: {metrics['duration_seconds']:.1f}s", file=sys.stderr)
    print(f"   Actions: {metrics['action_count']}", file=sys.stderr)
    print(f"   Steps: {metrics['step_count']}", file=sys.stderr)
    print(f"   Retries: {metrics['retry_count']}", file=sys.stderr)
    print(f"   Efficiency score: {metrics['efficiency_score']:.2f} (lower = better)", file=sys.stderr)

    return metrics


def compute_metrics(analysis: dict) -> dict:
    """Compute efficiency metrics from an analysis JSON.

    Metrics (lower = more efficient for all):
    - duration_seconds: wall time of the recording
    - action_count: total interaction events (clicks + keys + scrolls + drags)
    - step_count: number of logical steps detected
    - retry_count: number of detected retries (mis-clicks)
    - screenshot_count: number of screenshots saved
    - error_count: number of error events
    - efficiency_score: weighted combination (0-10 scale, lower = better)
    """
    metadata = analysis.get("metadata", {})
    steps = analysis.get("steps", [])

    # Duration
    duration = metadata.get("duration_seconds", 0.0)

    # Count interactions across all steps
    action_count = 0
    retry_count = 0
    click_count = 0
    type_count = 0
    scroll_count = 0
    drag_count = 0

    for step in steps:
        interactions = step.get("interactions", [])
        action_count += len(interactions)
        retry_count += step.get("retry_count", 0)

        for inter in interactions:
            action = inter.get("action", "")
            if action.startswith("click"):
                click_count += 1
            elif action.startswith("key"):
                type_count += 1
            elif action.startswith("scroll"):
                scroll_count += 1
            elif action.startswith("drag"):
                drag_count += 1

    # Screenshot count
    screenshot_count = metadata.get("screenshot_count", 0)

    # Error count (from metadata or events)
    error_count = metadata.get("error_count", 0)

    # Step count
    step_count = len(steps)

    # Pattern count (detected loops = potential inefficiency)
    patterns = analysis.get("patterns", [])
    pattern_count = len(patterns)

    # Compute efficiency score (0-10, lower = better)
    normalized = {}
    for key, cap in NORMALIZATION_CAPS.items():
        raw = locals().get(key, 0)
        normalized[key] = min(10.0, (float(raw) / cap) * 10.0) if cap > 0 else 0.0

    efficiency_score = sum(
        normalized.get(key, 0) * weight
        for key, weight in WEIGHTS.items()
    )

    return {
        "duration_seconds": round(duration, 2),
        "action_count": action_count,
        "click_count": click_count,
        "type_count": type_count,
        "scroll_count": scroll_count,
        "drag_count": drag_count,
        "step_count": step_count,
        "retry_count": retry_count,
        "screenshot_count": screenshot_count,
        "error_count": error_count,
        "pattern_count": pattern_count,
        "efficiency_score": round(efficiency_score, 2),
        "normalized": {k: round(v, 2) for k, v in normalized.items()},
    }


# ── Comparison ───────────────────────────────────────────────────────────────

def compare_attempts(session_dir: Path) -> dict:
    """Compare all attempts and pick the best one."""
    session = load_session(session_dir)
    attempts = session.get("attempts", [])

    if not attempts:
        print("ERROR: No attempts to compare. Run --analyze first.", file=sys.stderr)
        sys.exit(1)

    # Sort by efficiency score (lower = better)
    ranked = sorted(attempts, key=lambda a: a.get("metrics", {}).get("efficiency_score", 999.0))

    best = ranked[0]
    best_num = best.get("number", 0)

    # Compute improvements between consecutive attempts
    improvements = []
    sorted_by_num = sorted(attempts, key=lambda a: a.get("number", 0))
    for i in range(1, len(sorted_by_num)):
        prev = sorted_by_num[i - 1].get("metrics", {})
        curr = sorted_by_num[i].get("metrics", {})

        improvement = {
            "from_attempt": sorted_by_num[i - 1].get("number"),
            "to_attempt": sorted_by_num[i].get("number"),
            "duration_delta": round(prev.get("duration_seconds", 0) - curr.get("duration_seconds", 0), 2),
            "action_delta": prev.get("action_count", 0) - curr.get("action_count", 0),
            "retry_delta": prev.get("retry_count", 0) - curr.get("retry_count", 0),
            "score_delta": round(prev.get("efficiency_score", 0) - curr.get("efficiency_score", 0), 2),
        }
        improvements.append(improvement)

    comparison = {
        "task_name": session.get("task_name", "unknown"),
        "total_attempts": len(attempts),
        "ranked": [
            {
                "rank": i + 1,
                "attempt": a.get("number"),
                "score": a.get("metrics", {}).get("efficiency_score", 999.0),
                "duration": a.get("metrics", {}).get("duration_seconds", 0),
                "actions": a.get("metrics", {}).get("action_count", 0),
                "retries": a.get("metrics", {}).get("retry_count", 0),
                "steps": a.get("metrics", {}).get("step_count", 0),
            }
            for i, a in enumerate(ranked)
        ],
        "best_attempt": best_num,
        "best_metrics": best.get("metrics", {}),
        "improvements": improvements,
        "compared_at": datetime.now().isoformat(),
    }

    # Save comparison
    comparison_file = session_dir / "comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)

    # Generate report
    report = generate_report(session_dir, comparison, session)
    report_file = session_dir / "report.md"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"✅ Comparison complete ({len(attempts)} attempts)", file=sys.stderr)
    print(f"   Best: Attempt #{best_num} (score: {best.get('metrics', {}).get('efficiency_score', 0):.2f})", file=sys.stderr)
    print(f"   Report: {report_file}", file=sys.stderr)

    return comparison


def generate_report(session_dir: Path, comparison: dict, session: dict) -> str:
    """Generate a human-readable comparison report."""
    lines = []
    task_name = comparison.get("task_name", "unknown")
    description = session.get("description", "")

    lines.append(f"# Self-Learning Report: {task_name}")
    lines.append("")
    if description:
        lines.append(f"**Task:** {description}")
        lines.append("")
    lines.append(f"**Total attempts:** {comparison['total_attempts']}")
    lines.append(f"**Best attempt:** #{comparison['best_attempt']}")
    lines.append(f"**Generated:** {comparison['compared_at']}")
    lines.append("")

    # Attempts table
    lines.append("## Attempts")
    lines.append("")
    lines.append("| Attempt | Duration | Actions | Clicks | Keys | Steps | Retries | Score |")
    lines.append("|---------|----------|---------|--------|------|-------|---------|-------|")

    for entry in comparison["ranked"]:
        m = entry
        medal = ""
        if entry["rank"] == 1:
            medal = " 🏆"
        lines.append(
            f"| {entry['attempt']}{medal} "
            f"| {entry['duration']:.1f}s "
            f"| {entry['actions']} "
            f"| {m.get('click_count', '?')} "
            f"| {m.get('type_count', '?')} "
            f"| {entry['steps']} "
            f"| {entry['retries']} "
            f"| {entry['score']:.2f} |"
        )

    lines.append("")
    lines.append("*Score is a weighted efficiency metric (lower = better). Weights: "
                 "actions 30%, retries 30%, duration 20%, steps 20%.*")
    lines.append("")

    # Best attempt details
    best_metrics = comparison.get("best_metrics", {})
    lines.append(f"## Best Attempt: #{comparison['best_attempt']}")
    lines.append("")
    lines.append("### Why it won")
    lines.append("")

    # Explain why the best won
    ranked = comparison["ranked"]
    if len(ranked) > 1:
        worst = ranked[-1]
        best = ranked[0]
        lines.append(f"- Efficiency score: **{best['score']:.2f}** vs {worst['score']:.2f} (worst attempt #{worst['attempt']})")
        if best["actions"] < worst["actions"]:
            lines.append(f"- Fewer actions: **{best['actions']}** vs {worst['actions']} "
                        f"({worst['actions'] - best['actions']} fewer)")
        if best["retries"] < worst["retries"]:
            lines.append(f"- Fewer retries: **{best['retries']}** vs {worst['retries']} "
                        f"({worst['retries'] - best['retries']} fewer mis-clicks)")
        if best["duration"] < worst["duration"]:
            saved = worst["duration"] - best["duration"]
            lines.append(f"- Faster: **{best['duration']:.1f}s** vs {worst['duration']:.1f}s "
                        f"({saved:.1f}s saved)")
        if best["steps"] < worst["steps"]:
            lines.append(f"- Fewer steps: **{best['steps']}** vs {worst['steps']} "
                        f"(combined related actions)")
    lines.append("")

    # Improvements across attempts
    improvements = comparison.get("improvements", [])
    if improvements:
        lines.append("### Improvements Across Attempts")
        lines.append("")
        for imp in improvements:
            lines.append(f"**Attempt {imp['from_attempt']} → {imp['to_attempt']}:**")
            if imp["duration_delta"] > 0:
                lines.append(f"- ⏱️  {imp['duration_delta']:.1f}s faster")
            if imp["action_delta"] > 0:
                lines.append(f"- 👆 {imp['action_delta']} fewer actions")
            if imp["retry_delta"] > 0:
                lines.append(f"- 🎯 {imp['retry_delta']} fewer retries")
            if imp["score_delta"] > 0:
                lines.append(f"- 📊 Score improved by {imp['score_delta']:.2f}")
            lines.append("")

    # Recommendation
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"Use **Attempt #{comparison['best_attempt']}** as the basis for the generated skill.")
    lines.append(f"Run `self_learn.py --generate {session_dir}` to create the skill.")
    lines.append("")

    return "\n".join(lines)


# ── Skill generation ─────────────────────────────────────────────────────────

def generate_best_skill(session_dir: Path, script_dir: str, dry_run: bool = False) -> Path:
    """Generate a skill from the best attempt."""
    comparison_file = session_dir / "comparison.json"
    if not comparison_file.exists():
        print("ERROR: No comparison found. Run --compare first.", file=sys.stderr)
        sys.exit(1)

    with open(comparison_file) as f:
        comparison = json.load(f)

    best_num = comparison["best_attempt"]
    best_attempt_dir = session_dir / f"attempt-{best_num}"

    # Get recording path
    recording_path_file = best_attempt_dir / "recording-path"
    if not recording_path_file.exists():
        print(f"ERROR: No recording path for attempt {best_num}", file=sys.stderr)
        sys.exit(1)

    recording_dir = recording_path_file.read_text().strip()
    analysis_file = best_attempt_dir / "analysis.json"

    # Get task name for skill name
    session = load_session(session_dir)
    task_name = session.get("task_name", "learned-task")

    # Run generate_skill.py
    generate_script = os.path.join(script_dir, "generate_skill.py")
    output_dir = session_dir / "best-skill"

    cmd = [
        sys.executable, generate_script,
        "--analysis", str(analysis_file),
        "--recording-dir", recording_dir,
        "--name", task_name,
        "--output-dir", str(output_dir),
    ]
    if dry_run:
        cmd.append("--dry-run")

    print(f"Generating skill from attempt {best_num}...", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"ERROR: generate_skill.py failed (exit {result.returncode})", file=sys.stderr)
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Skill generated from attempt {best_num}", file=sys.stderr)
    if not dry_run:
        skill_path = output_dir / "SKILL.md"
        if skill_path.exists():
            print(f"   Saved to: {skill_path}", file=sys.stderr)
            print(f"", file=sys.stderr)
            print(f"To install the skill:", file=sys.stderr)
            print(f"   cp -r {output_dir} ~/.hermes/skills/automation/{task_name}/", file=sys.stderr)
            return skill_path

    print(result.stdout[:500] if result.stdout else "", file=sys.stderr)
    return output_dir


# ── Report printing ──────────────────────────────────────────────────────────

def print_report(session_dir: Path):
    """Print the comparison report to stdout."""
    report_file = session_dir / "report.md"
    if not report_file.exists():
        print("ERROR: No report found. Run --compare first.", file=sys.stderr)
        sys.exit(1)
    print(report_file.read_text())


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Self-learning mode — agent records its own actions, iterates, and saves the best attempt as a skill"
    )
    parser.add_argument("--init", type=str, metavar="TASK_NAME",
                        help="Initialize a new self-learning session")
    parser.add_argument("--description", type=str, default="",
                        help="Task description (used with --init)")
    parser.add_argument("--analyze", type=str, metavar="SESSION_DIR",
                        help="Analyze an attempt. Requires --attempt and --recording")
    parser.add_argument("--attempt", type=int, metavar="N",
                        help="Attempt number (used with --analyze)")
    parser.add_argument("--recording", type=str, metavar="DIR",
                        help="Path to the recording directory (used with --analyze)")
    parser.add_argument("--compare", type=str, metavar="SESSION_DIR",
                        help="Compare all attempts and pick the best")
    parser.add_argument("--generate", type=str, metavar="SESSION_DIR",
                        help="Generate skill from the best attempt")
    parser.add_argument("--report", type=str, metavar="SESSION_DIR",
                        help="Print the comparison report")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview skill generation without saving")
    args = parser.parse_args()

    # Get script directory (for calling sibling scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.init:
        init_session(args.init, args.description)

    elif args.analyze:
        if not args.attempt or not args.recording:
            print("ERROR: --analyze requires --attempt and --recording", file=sys.stderr)
            sys.exit(1)
        session_dir = Path(args.analyze)
        if not session_dir.exists():
            print(f"ERROR: Session directory not found: {session_dir}", file=sys.stderr)
            sys.exit(1)
        analyze_attempt(session_dir, args.attempt, args.recording, script_dir)

    elif args.compare:
        session_dir = Path(args.compare)
        if not session_dir.exists():
            print(f"ERROR: Session directory not found: {session_dir}", file=sys.stderr)
            sys.exit(1)
        compare_attempts(session_dir)

    elif args.generate:
        session_dir = Path(args.generate)
        if not session_dir.exists():
            print(f"ERROR: Session directory not found: {session_dir}", file=sys.stderr)
            sys.exit(1)
        generate_best_skill(session_dir, script_dir, args.dry_run)

    elif args.report:
        session_dir = Path(args.report)
        if not session_dir.exists():
            print(f"ERROR: Session directory not found: {session_dir}", file=sys.stderr)
            sys.exit(1)
        print_report(session_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

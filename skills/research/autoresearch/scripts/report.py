#!/usr/bin/env python3
"""Autoresearch report generation from state files.

Usage:
    python report.py generate <run_dir>
    python report.py summary <run_dir>
"""
import json, os, sys
from pathlib import Path
from _util import read_json


def _read_text(path):
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return ""


def _results_entries(run_dir):
    content = _read_text(os.path.join(run_dir, "results.log"))
    if not content.strip():
        return []
    return [e.strip() for e in content.split("---\n") if e.strip()]


def generate_report(run_dir):
    config = read_json(os.path.join(run_dir, "config.json"))
    status = read_json(os.path.join(run_dir, "status.json"))
    plan = read_json(os.path.join(run_dir, "plan.json"))
    usage = read_json(os.path.join(run_dir, "usage.json"))
    entries = _results_entries(run_dir)

    merged = [e for e in entries if "Decision: MERGE" in e]
    reverted = [e for e in entries if "Decision: REVERT" in e]

    lines = []
    lines.append(f"# Research Report: {config.get('goal', 'Unknown')}")
    lines.append("")
    lines.append(f"- **Domain**: {config.get('domain', 'N/A')}")
    lines.append(f"- **Scope**: {config.get('scope', 'N/A')}")
    lines.append(f"- **Depth**: {config.get('depth', 'N/A')}")
    lines.append(
        f"- **Experiments**: {status.get('experiments_done', 0)}"
        f" ({status.get('experiments_merged', 0)} merged,"
        f" {status.get('experiments_reverted', 0)} reverted,"
        f" {status.get('experiments_failed', 0)} failed)"
    )

    if usage:
        lines.append(
            f"- **Tokens**: {usage.get('total_tokens', 0):,}"
            f" (input: {usage.get('total_input_tokens', 0):,},"
            f" output: {usage.get('total_output_tokens', 0):,})"
        )
        if usage.get("estimated_cost_usd"):
            lines.append(
                f"- **Estimated cost**: ${usage['estimated_cost_usd']:.4f}"
            )

    lines.append("")

    if entries:
        merge_rate = len(merged) / len(entries) * 100 if entries else 0
        lines.append("## Summary")
        lines.append("")
        lines.append(f"Merge rate: {merge_rate:.0f}% ({len(merged)}/{len(entries)})")
        lines.append("")

    # Experiment log
    if merged:
        lines.append("## Merged Experiments")
        lines.append("")
        for e in merged:
            for line in e.split("\n"):
                if line.startswith("## "):
                    lines.append(f"\n### {line[3:]}")
                elif line.startswith("Reason:"):
                    lines.append(f"  {line}")

    if reverted:
        lines.append("")
        lines.append("## Reverted Experiments")
        lines.append("")
        for e in reverted:
            for line in e.split("\n"):
                if line.startswith("## "):
                    lines.append(f"\n### {line[3:]}")
                elif line.startswith("Reason:"):
                    lines.append(f"  {line}")

    report = "\n".join(lines)
    report_path = os.path.join(run_dir, "report.md")
    Path(report_path).write_text(report)
    print(json.dumps({"status": "generated", "path": report_path}))


def summary(run_dir):
    config = read_json(os.path.join(run_dir, "config.json"))
    status = read_json(os.path.join(run_dir, "status.json"))
    entries = _results_entries(run_dir)

    merged = sum(1 for e in entries if "Decision: MERGE" in e)
    reverted = sum(1 for e in entries if "Decision: REVERT" in e)

    lines = [
        f"Research: {config.get('goal', 'Unknown')}",
        f"Phase: {status.get('phase', 'unknown')}",
        f"Experiments: {status.get('experiments_done', 0)}/{status.get('experiments_total', 0)}"
        f" | Merged: {merged} | Reverted: {reverted} | Failed: {status.get('experiments_failed', 0)}",
    ]

    if entries and merged > 0:
        top_merges = [
            e.split("\n")[0]
            for e in entries
            if "Decision: MERGE" in e
        ][:3]
        for tm in top_merges:
            lines.append(f"  + {tm}")

    print(json.dumps({"summary": "\n".join(lines)}))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    cmd = args[0]
    if cmd == "generate":
        generate_report(args[1])
    elif cmd == "summary":
        summary(args[1])
    else:
        print(f"Unknown: {cmd}")
        sys.exit(1)

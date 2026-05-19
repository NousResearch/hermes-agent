#!/usr/bin/env python3
"""
Compute Self-Heal Schedule

Computes a new schedule based on basic telemetry (simulated for now,
or using simple PR query if gh CLI is available) and safely updates
the .github/self-heal-schedule.yml using marker-based replacement.
"""

import os
import subprocess
import re
import sys
from datetime import datetime

SCHEDULE_FILE = ".github/self-heal-schedule.yml"
MARKER = "# AUTO-UPDATED"

def compute_new_schedule() -> str:
    """
    Computes new schedule cron string based on telemetry.
    For demonstration, we check the number of recent PRs.
    Tiers:
    - High churn (>10 PRs/week): daily at 02:00 -> "0 2 * * *"
    - Standard (<=10 PRs/week): weekly on Sunday at 02:00 -> "0 2 * * 0"
    """
    try:
        # Check if gh CLI is available
        result = subprocess.run(
            ["gh", "pr", "list", "--state", "merged", "--limit", "15"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) > 10:
                print("High churn detected. Setting daily schedule.")
                return "0 2 * * *"
    except FileNotFoundError:
        pass

    print("Standard churn detected. Setting weekly schedule.")
    return "0 2 * * 0"

def update_schedule_file(new_cron: str) -> None:
    """Safely updates the schedule file using marker replacement."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(repo_root, SCHEDULE_FILE)

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if MARKER not in content:
        print(f"Marker '{MARKER}' not found in {file_path}.")
        sys.exit(1)

    # Replace the cron string
    new_content = re.sub(
        r'cron:\s*".*?"',
        f'cron: "{new_cron}"',
        content
    )

    if new_content == content:
        print("Schedule unchanged.")
        sys.exit(0)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Updated {SCHEDULE_FILE} with new schedule: {new_cron}")

def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    new_cron = compute_new_schedule()
    update_schedule_file(new_cron)

if __name__ == "__main__":
    main()

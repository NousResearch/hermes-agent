#!/usr/bin/env python3
"""
Self-Heal Pipeline

Idempotent script to apply allowed repair actions.
After each step, re-runs healthcheck.
If pass+diff -> exits 0 immediately (success).
If pass+no-diff -> next step.
If fail -> next step.
End: exit 1 if no fix.
"""

import subprocess
import sys
import os

SCHEDULE_FILE = ".github/self-heal-schedule.yml"

def run_cmd(cmd: list[str]) -> bool:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed:\n{result.stderr}")
        return False
    return True

def has_diff() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return bool(result.stdout.strip())

def run_healthcheck() -> bool:
    result = subprocess.run(["scripts/healthcheck.sh"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Healthcheck failed:\n{result.stdout}\n{result.stderr}")
        return False
    return True

def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    print("Starting Self-Heal Pipeline...")

    steps = [
        {"name": "Lockfile refresh", "cmd": ["uv", "lock"]},
        {"name": "Ruff lint auto-fix", "cmd": ["uv", "run", "ruff", "check", "--fix", "."]},
        {"name": "Ruff format auto-fix", "cmd": ["uv", "run", "ruff", "format", "."]},
        {"name": "Ty type stub check", "cmd": ["uv", "run", "ty", "check"]},
    ]

    for step in steps:
        print(f"\n--- Executing Step: {step['name']} ---")
        run_cmd(step["cmd"])

        passed = run_healthcheck()
        diff_present = has_diff()

        if passed and diff_present:
            print(f"Step {step['name']} succeeded and generated a diff! Exiting 0.")
            sys.exit(0)
        elif not passed:
            print(f"Step {step['name']} left repo in failing state. Proceeding to next step.")
        else:
            print(f"Step {step['name']} passed but produced no diff. Proceeding to next step.")

    print("\nSelf-Heal Pipeline completed. No viable fixes found.")
    sys.exit(1)

if __name__ == "__main__":
    main()

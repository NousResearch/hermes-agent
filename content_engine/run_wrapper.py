#!/usr/bin/env python3
"""Wrapper to run the workflow script with proper env."""
import os
import subprocess
import sys

WORKDIR = "/home/kensei/repos/KenseiAgent/content_engine"
ENV_FILE = "/home/kensei/.hermes/.env"

env = os.environ.copy()
if os.path.exists(ENV_FILE):
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                env[key] = val
env["PYTHONPATH"] = WORKDIR

proc = subprocess.run(
    ["python3", "run_workflow.py"],
    capture_output=True, text=True, timeout=300,
    env=env, cwd=WORKDIR
)
print(proc.stdout)
if proc.stderr:
    print(f"STDERR:\n{proc.stderr}", file=sys.stderr)
print(f"RETURNCODE: {proc.returncode}", file=sys.stderr)
sys.exit(proc.returncode)
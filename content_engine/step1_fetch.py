#!/usr/bin/env python3
"""Step 1: Run inbox-fetch and print raw output."""
import os
import subprocess
import sys

WORKDIR = "/home/kensei/repos/KenseiAgent/content_engine"
ENV_FILE = "/home/kensei/.hermes/.env"
PYTHON = "python3"
SCRIPT = os.path.join(WORKDIR, "content_engine.py")

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
    [PYTHON, SCRIPT, "inbox-fetch"],
    capture_output=True, text=True, timeout=60,
    env=env, cwd=WORKDIR
)
print(f"RETURNCODE: {proc.returncode}")
print(f"STDOUT:\n{proc.stdout}")
print(f"STDERR:\n{proc.stderr}")
sys.exit(proc.returncode)
#!/usr/bin/env python3
"""Cron wrapper for the hermes-session-log skill (no-agent job, every N minutes).

The Hermes cron scheduler only resolves `script` paths under $HERMES_HOME/scripts/,
so copy this wrapper there:

    cp <skill>/scripts/cron_wrapper.py "$HERMES_HOME/scripts/hermes-session-log-export.py"

Then schedule a no-agent job that runs it (every 5 minutes):

    name:     hermes-session-log
    schedule: 5m
    no_agent: true
    script:   hermes-session-log-export.py

The wrapper is silent on no-op runs (cron watchdog pattern: empty stdout
delivers nothing), and prints a short change report otherwise.
"""
import os
import runpy
import sys
from pathlib import Path


def _candidate_skill_dirs():
    env_dir = os.environ.get("HERMES_SESSION_LOG_SKILL", "")
    if env_dir:
        yield Path(env_dir)
    bases = []
    if os.environ.get("HERMES_HOME"):
        bases.append(Path(os.environ["HERMES_HOME"]))
    bases.append(Path.home() / ".hermes")
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        bases.append(Path(os.environ["LOCALAPPDATA"]) / "hermes")
    for b in bases:
        yield b / "skills" / "hermes-session-log"


def main() -> None:
    script = None
    for d in _candidate_skill_dirs():
        cand = d / "scripts" / "archive_hermes_sessions.py"
        if cand.is_file():
            script = cand
            break
    if script is None:
        print("hermes-session-log: export script not found; set HERMES_SESSION_LOG_SKILL to the skill directory")
        raise SystemExit(1)
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

# Auto-generated/maintained by Hermes. No-agent WorldMonitor PDB situation-report cron.
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = os.environ.get("HERMES_REPO_ROOT", r"C:\Users\downl\Documents\New project\hermes-agent")
TIMEOUT_SECONDS = int(os.environ.get("HERMES_OSINT_TIMEOUT_SECONDS", "1200"))
SLOT = "morning"


def _python_candidates() -> list[str]:
    candidates: list[str] = []
    configured = os.environ.get("HERMES_PYTHON")
    if configured:
        candidates.append(configured)
    root = Path(REPO_ROOT)
    for rel in (r".venv\Scripts\python.exe", r"venv\Scripts\python.exe"):
        p = root / rel
        if p.exists():
            candidates.append(str(p))
    candidates.append(sys.executable)
    return candidates


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT_SECONDS,
    )

args = [
    "-m", "hermes_cli.main",
    "worldmonitor-osint", "situation-report",
    "--slot", SLOT,
    "--cron-stdout",
]

last = None
for pyexe in _python_candidates():
    argv = [pyexe, *args]
    try:
        result = _run(argv)
    except Exception as exc:  # noqa: BLE001
        last = (argv, None, "", str(exc))
        continue
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode == 0:
        print(stdout or "WorldMonitor PDB situation report completed with no stdout.")
        raise SystemExit(0)
    last = (argv, result.returncode, stdout, stderr)

argv, code, stdout, stderr = last or ([], 1, "", "unknown error")
print(f"WorldMonitor PDB cron failed: argv={argv!r} returncode={code}")
if stderr:
    print(stderr)
if stdout:
    print(stdout)
raise SystemExit(code or 1)

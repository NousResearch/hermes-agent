# Auto-generated/maintained by Hermes. Unified OSINT MILSPEC markdown cron.
from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.environ.get("HERMES_REPO_ROOT", r"C:\Users\downl\Documents\New project\hermes-agent")
SLOT = "evening"


def _run(argv):
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("HERMES_OSINT_REPORT_FORMAT", "markdown")
    return subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1200,
    )

args = [
    "-m", "hermes_cli.main",
    "osint-agent", "brief",
    "--slot", SLOT,
    "--topic", "日本の安全保障と世界情勢",
    "--source-mode", "real",
    "--wm-tier", "free",
    "--cron-stdout",
]

candidates = []
configured = os.environ.get("HERMES_PYTHON")
if configured:
    candidates.append([configured, *args])
candidates.append([sys.executable, *args])
candidates.append(["py", "-3", *args])

last = None
for argv in candidates:
    try:
        result = _run(argv)
    except Exception as exc:
        last = (argv, None, "", str(exc))
        continue
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode == 0:
        if stdout:
            print(stdout)
        raise SystemExit(0)
    last = (argv, result.returncode, stdout, stderr)

argv, code, stdout, stderr = last or ([], 1, "", "unknown error")
print(f"OSINT cron failed: argv={argv!r} returncode={code}")
if stderr:
    print(stderr)
if stdout:
    print(stdout)
raise SystemExit(code or 1)

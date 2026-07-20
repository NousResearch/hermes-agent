#!/usr/bin/env python3
"""Standalone ebbinghaus test runner that writes results to a file.

Uses the project venv when available and skips the heavy root conftest
(``--noconftest``) so Windows runs stay responsive.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_tmp" / "ebbinghaus-pytest.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)

venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
python = str(venv_py) if venv_py.exists() else sys.executable

hermes_home = Path(tempfile.mkdtemp(prefix="hermes-ebb-"))
env = os.environ.copy()
env["HERMES_HOME"] = str(hermes_home)
env["PYTHONDONTWRITEBYTECODE"] = "1"

cmds = [
    [python, "-m", "compileall", "-q", "plugins/memory/ebbinghaus"],
    [
        python,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "--noconftest",
        "tests/plugins/test_ebbinghaus_plugin.py",
        "tests/skills/test_ebbinghaus_memory_skill.py",
    ],
]

lines: list[str] = [f"python={python}", f"HERMES_HOME={hermes_home}", ""]
code = 0
for cmd in cmds:
    lines.append("$ " + " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        lines.append(str(exc))
        lines.append("exit=124")
        code = 124
        break
    lines.append(proc.stdout or "")
    lines.append(proc.stderr or "")
    lines.append(f"exit={proc.returncode}")
    lines.append("")
    if proc.returncode != 0:
        code = proc.returncode

OUT.write_text("\n".join(lines), encoding="utf-8")
print(OUT.read_text(encoding="utf-8"))
raise SystemExit(code)

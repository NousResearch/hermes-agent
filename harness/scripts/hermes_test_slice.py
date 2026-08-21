#!/usr/bin/env python3
"""Roda scripts/run_tests.sh em paths explícitos (sem globs).

Saída JSON no stdout. Exit 0 se testes passarem; 1 caso contrário.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


GLOB_CHARS = re.compile(r"[\*\?\[]")


def _find_bash() -> str | None:
    """Prefer Git for Windows / MinGit over WSL system32 bash."""
    candidates = [
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path.home() / "AppData/Local/hermes/git/bin/bash.exe",
        Path.home() / "AppData/Local/hermes/git/usr/bin/bash.exe",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    which = shutil.which("bash")
    if which and "system32" not in Path(which).as_posix().lower():
        return which
    return None


def _bash_path(path: Path) -> str:
    """Windows path Git Bash accepts (forward slashes)."""
    return path.as_posix()


def _find_python_with_pytest(repo: Path) -> str | None:
    """Probe Windows and Unix venv layouts for a python with pytest."""
    candidates = [
        repo / ".venv" / "Scripts" / "python.exe",
        repo / ".venv" / "bin" / "python",
        repo / "venv" / "Scripts" / "python.exe",
        repo / "venv" / "bin" / "python",
        Path.home() / "AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe",
        Path.home() / "AppData/Local/hermes/hermes-agent/venv/bin/python",
    ]
    for py in candidates:
        if not py.is_file():
            continue
        try:
            subprocess.run(
                [str(py), "-c", "import pytest"],
                capture_output=True,
                check=True,
                timeout=30,
            )
            return str(py)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            continue
    return None


def _validate_paths(repo: Path, paths: list[str]) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    errors: list[str] = []
    for raw in paths:
        if GLOB_CHARS.search(raw):
            errors.append(f"glob not allowed: {raw!r}")
            continue
        p = (repo / raw).resolve()
        try:
            p.relative_to(repo.resolve())
        except ValueError:
            errors.append(f"outside repo: {raw!r}")
            continue
        if not p.exists():
            errors.append(f"not found: {raw!r}")
            continue
        ok.append(raw.replace("\\", "/"))
    return ok, errors


def _tail(text: str, max_lines: int = 40) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes test slice runner")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["tests/agent/"],
        help="Test dirs or .py files (no globs)",
    )
    parser.add_argument(
        "--repo",
        default=str(Path(__file__).resolve().parents[2]),
        help="hermes-agent repo root",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Subprocess timeout seconds",
    )
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    run_tests = repo / "scripts" / "run_tests.sh"

    valid_paths, path_errors = _validate_paths(repo, args.paths)
    if path_errors:
        out = {
            "passed": False,
            "paths": args.paths,
            "errors": path_errors,
            "output_summary": "path validation failed",
        }
        print(json.dumps(out, indent=2))
        return 1

    if not run_tests.is_file():
        out = {"passed": False, "output_summary": f"missing {run_tests}"}
        print(json.dumps(out, indent=2))
        return 1

    bash = _find_bash()
    if not bash:
        out = {
            "passed": False,
            "output_summary": "bash not found (install Git for Windows or Hermes MinGit)",
        }
        print(json.dumps(out, indent=2))
        return 1

    env = None
    hermes_python = _find_python_with_pytest(repo)
    if hermes_python:
        import os

        env = os.environ.copy()
        env["HERMES_PYTHON"] = hermes_python

    cmd = [bash, _bash_path(run_tests), *valid_paths]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        out = {
            "passed": False,
            "paths": valid_paths,
            "bash": bash,
            "output_summary": f"timeout after {args.timeout}s",
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
        }
        print(json.dumps(out, indent=2))
        return 1

    combined = (proc.stdout or "") + (proc.stderr or "")
    passed = proc.returncode == 0
    out = {
        "passed": passed,
        "exit_code": proc.returncode,
        "paths": valid_paths,
        "bash": bash,
        "output_summary": _tail(combined, 60),
    }
    print(json.dumps(out, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

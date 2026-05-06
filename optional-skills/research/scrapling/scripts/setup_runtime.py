#!/usr/bin/env python3
"""Create an isolated Scrapling pilot runtime.

This script intentionally installs Scrapling outside the Hermes main venv. Browser
assets are opt-in via --install-browsers because Patchright/Playwright downloads
are large and should not happen during ordinary setup checks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

DEFAULT_PYTHON = os.environ.get("SCRAPLING_PYTHON", "/Users/zhaopufan/.local/bin/python3.11")
DEFAULT_RUNTIME_DIR = Path(os.environ.get("SCRAPLING_RUNTIME_DIR", "~/.hermes/runtimes/scrapling")).expanduser()
DEFAULT_REQUIREMENTS_FILE = Path(__file__).resolve().parents[1] / "requirements.txt"


class RuntimeSetupPlan:
    """Small explicit container; avoids dataclass import edge cases in script tests."""

    def __init__(
        self,
        *,
        python_executable: str,
        runtime_dir: Path,
        requirements_file: Path,
        browser_install_requested: bool,
        commands: list[list[str]],
    ) -> None:
        self.python_executable = python_executable
        self.runtime_dir = runtime_dir
        self.requirements_file = requirements_file
        self.browser_install_requested = browser_install_requested
        self.commands = commands

    def as_dict(self) -> dict:
        return {
            "python_executable": self.python_executable,
            "runtime_dir": str(self.runtime_dir),
            "requirements_file": str(self.requirements_file),
            "browser_install_requested": self.browser_install_requested,
            "commands": self.commands,
        }


def venv_bin_dir(runtime_dir: Path) -> Path:
    return runtime_dir / ("Scripts" if os.name == "nt" else "bin")


def venv_python(runtime_dir: Path) -> Path:
    return venv_bin_dir(runtime_dir) / ("python.exe" if os.name == "nt" else "python")


def venv_scrapling(runtime_dir: Path) -> Path:
    return venv_bin_dir(runtime_dir) / ("scrapling.exe" if os.name == "nt" else "scrapling")


def build_setup_plan(
    *,
    python_executable: str | os.PathLike[str] = DEFAULT_PYTHON,
    runtime_dir: str | os.PathLike[str] = DEFAULT_RUNTIME_DIR,
    requirements_file: str | os.PathLike[str] = DEFAULT_REQUIREMENTS_FILE,
    install_browsers: bool = False,
) -> RuntimeSetupPlan:
    runtime_path = Path(runtime_dir).expanduser()
    requirements_path = Path(requirements_file).expanduser()
    python_path = str(python_executable)
    runtime_python = venv_python(runtime_path)

    commands: list[list[str]] = [
        [python_path, "-m", "venv", str(runtime_path)],
        [str(runtime_python), "-m", "pip", "install", "--upgrade", "pip"],
        [str(runtime_python), "-m", "pip", "install", "-r", str(requirements_path)],
    ]
    if install_browsers:
        commands.append([str(venv_scrapling(runtime_path)), "install"])

    return RuntimeSetupPlan(
        python_executable=python_path,
        runtime_dir=runtime_path,
        requirements_file=requirements_path,
        browser_install_requested=install_browsers,
        commands=commands,
    )


def command_display(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def run_command(command: Sequence[str]) -> dict:
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an isolated Scrapling fallback runtime without touching the Hermes main venv.",
    )
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="Python 3.10-3.13 executable to use for the runtime venv.")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR), help="Runtime directory, default: ~/.hermes/runtimes/scrapling")
    parser.add_argument("--requirements-file", default=str(DEFAULT_REQUIREMENTS_FILE), help="Requirements file to install in the isolated venv.")
    parser.add_argument("--install-browsers", action="store_true", help="Also run 'scrapling install' for browser-backed dynamic/stealth fetchers.")
    parser.add_argument("--dry-run", action="store_true", help="Print the setup plan as JSON without executing commands.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    plan = build_setup_plan(
        python_executable=args.python,
        runtime_dir=args.runtime_dir,
        requirements_file=args.requirements_file,
        install_browsers=args.install_browsers,
    )

    receipt = {
        "backend": "scrapling",
        "action": "setup_runtime",
        "runtime_dir": str(plan.runtime_dir),
        "python_executable": plan.python_executable,
        "browser_install_requested": plan.browser_install_requested,
        "dry_run": args.dry_run,
        "commands": [command_display(command) for command in plan.commands],
        "results": [],
        "errors": [],
    }

    if args.dry_run:
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0

    if not plan.requirements_file.exists():
        receipt["errors"].append({"type": "FileNotFoundError", "message": f"requirements file not found: {plan.requirements_file}"})
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 2

    for command in plan.commands:
        result = run_command(command)
        receipt["results"].append(result)
        if result["returncode"] != 0:
            receipt["errors"].append({"type": "CommandFailed", "message": command_display(command), "returncode": result["returncode"]})
            print(json.dumps(receipt, ensure_ascii=False, indent=2))
            return result["returncode"] or 1

    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

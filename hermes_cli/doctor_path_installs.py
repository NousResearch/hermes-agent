"""``hermes doctor`` check: every ``hermes`` command on PATH, which install each one runs, and its version.

A second ``hermes`` earlier on PATH (an old pip install, a stale managed-environment snapshot, another
checkout) makes the command the user types run different code than the install doctor is diagnosing.
Each distinct command is asked for ``--version``; its ``Install directory:`` line names the install.
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

from hermes_cli.doctor_report import Finding, check_info, check_ok, check_warn, doctor_check

_VERSION_TIMEOUT_S = 20


def _path_commands(name: str = "hermes") -> list[Path]:
    """Every executable *name* on PATH in lookup order, deduplicated by resolved target."""
    seen: set[Path] = set()
    found: list[Path] = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(entry or ".") / name
        if not (candidate.is_file() and os.access(candidate, os.X_OK)):
            continue
        target = candidate.resolve()
        if target not in seen:
            seen.add(target)
            found.append(candidate)
    return found


def _describe(command: Path) -> tuple[str, Path | None]:
    """(version label, install directory) reported by ``<command> --version``; label explains a failure."""
    try:
        proc = subprocess.run(
            [str(command), "--version"], stdin=subprocess.DEVNULL, capture_output=True,
            encoding="utf-8", errors="replace", timeout=_VERSION_TIMEOUT_S, check=False,
        )
    except subprocess.TimeoutExpired:
        return f"(no answer to --version within {_VERSION_TIMEOUT_S} s)", None
    except OSError as exc:
        return f"(could not run: {exc})", None
    lines = proc.stdout.splitlines()
    label = next((line.strip() for line in lines if line.startswith("Hermes Agent")), "")
    install = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("Install directory:")), "")
    if proc.returncode or not label:
        tail = (proc.stderr.strip().splitlines() or [f"exit code {proc.returncode}"])[-1]
        return f"(--version failed: {tail})", Path(install).resolve() if install else None
    return label, Path(install).resolve() if install else None


@doctor_check()
def _check_path_installs(should_fix: bool, f: Finding) -> None:
    """List every ``hermes`` on PATH; warn when the one that wins lookup is not this install."""
    from hermes_cli.doctor import PROJECT_ROOT

    if sys.platform == "win32":
        return
    commands = _path_commands()
    if not commands:
        return check_warn("No hermes command on PATH")
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(commands))) as pool:
        described = list(pool.map(_describe, commands))
    ours = Path(PROJECT_ROOT).resolve()
    for index, (command, (label, install)) in enumerate(zip(commands, described)):
        where = f"→ {install}" if install else ""
        if install == ours:
            check_ok(f"{command}  {label}", where)
        elif index == 0:
            check_warn(f"{command}  {label}", f"{where} (runs first on PATH, but is not this install: {ours})")
            f.manual_issues.append(
                f"`hermes` on PATH runs {install or command}, not {ours}: remove the other install or put "
                f"this install's command earlier on PATH"
            )
        else:
            check_info(f"also on PATH (shadowed): {command}  {label} {where}".rstrip())

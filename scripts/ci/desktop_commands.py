"""Expose already-prepared tools to later CI audit/upload steps; never install."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts.bundles.desktop_prepare import PreparedDesktop
from scripts.ci.setup_toolchain import add_path, file_commands


def export_commands(path: Path) -> None:
    prepared = PreparedDesktop.load(path)
    prepared.validate()
    directories = [str(prepared.python.parent), str(prepared.node.parent)]
    # PBS provides python3 on POSIX. Keep the workflow's `python` command
    # job-local: adding an alias inside the verified tool store invalidates it.
    if os.name != "nt":
        commands = prepared.request.work / "commands"
        if commands.is_symlink():
            raise ValueError("CI commands directory must not be a symlink")
        commands.mkdir(exist_ok=True)
        alias = commands / "python"
        alias.unlink(missing_ok=True)
        alias.symlink_to(prepared.python)
        directories.insert(0, str(commands))
    file_commands("GITHUB_ENV", {
        "HERMES_PYTHON": prepared.python,
        "HERMES_NODE": prepared.node,
        "HERMES_HOME": prepared.request.work / "hermes-home",
        "HERMES_RUNTIME_DIR": prepared.request.cache / "tools",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    add_path(directories)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    export_commands(parser.parse_args().prepared)


if __name__ == "__main__":
    main()

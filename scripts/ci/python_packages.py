"""Prepare isolated CI tools, then export their Python or run it directly.

Requirements precede ``--``; arguments after it run with the selected Python.
Without a command, export the selection for subsequent GitHub Actions steps.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    from pm import ensure_environment
    from scripts.ci.setup_toolchain import add_path, file_commands, python3_alias

    argv = list(sys.argv[1:] if argv is None else argv)
    boundary = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", nargs="+")
    args = parser.parse_args(argv[:boundary])
    command = argv[boundary + 1:]
    if boundary < len(argv) and not command:
        parser.error("-- requires a Python command")
    python = ensure_environment("ci-tools", args.requirements, explicit=True)
    python3_alias(python)
    if command:
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"PYTHONHOME", "VIRTUAL_ENV"}}
        environment.update(HERMES_PYTHON=str(python), VIRTUAL_ENV=str(python.parent.parent))
        environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
        return subprocess.call([str(python), *command], env=environment)
    file_commands("GITHUB_ENV", {"HERMES_PYTHON": python, "VIRTUAL_ENV": python.parent.parent})
    add_path([str(python.parent)])
    print(f"CI tools ready: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Minimal entrypoint for the non-admin Windows Hermes test principal."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("expected one encoded test contract")
    contract = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
    python = Path(contract["python"]).resolve()
    repo = Path(contract["repo"]).resolve()
    if python != Path(sys.executable).resolve() or not repo.is_dir():
        raise SystemExit("restricted test contract does not match this interpreter")
    environment = contract["environment"]
    arguments = contract["arguments"]
    if not isinstance(environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in environment.items()
    ):
        raise SystemExit("invalid restricted environment")
    if not isinstance(arguments, list) or not all(
        isinstance(argument, str) for argument in arguments
    ):
        raise SystemExit("invalid pytest arguments")
    os.environ.clear()
    os.environ.update(environment)
    os.chdir(repo)
    os.execv(str(python), [str(python), "-m", "pytest", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())

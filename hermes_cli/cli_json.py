"""Shared output convention for ``--json`` on CLI list/report commands.

Every document is one JSON object on stdout with ``schema_version`` first (the same convention
as ``hermes status --json``); diagnostics and errors go to stderr with a non-zero exit, so
``hermes <cmd> --json | jq`` never parses prose. Shapes are documented in
website/docs/reference/cli-commands.md; a key is only ever added, and a removal or a meaning
change bumps that document's ``schema_version``.
"""

from __future__ import annotations

import json
import sys
from typing import Any, NoReturn


def print_json_document(document: dict[str, Any], *, schema_version: int = 1) -> None:
    """Print ``{"schema_version": N, **document}`` to stdout."""
    print(json.dumps({"schema_version": schema_version, **document}, indent=2, ensure_ascii=False, default=str))


def json_error(message: str, code: int = 2) -> NoReturn:
    """Report a ``--json`` command failure on stderr and exit non-zero; stdout stays empty."""
    print(f"hermes: error: {message}", file=sys.stderr)
    raise SystemExit(code)

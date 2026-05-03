"""``hermes file-safety`` subcommand — diagnose why a path is read/write blocked.

Wraps :func:`agent.file_safety.explain_write_denial` and
:func:`agent.file_safety.get_read_block_error` so the user can see exactly
which rule matched (denylist exact path / sensitive prefix / safe-root
violation / internal skill cache read-block) instead of just a generic
"denied" from the file tools.
"""

from __future__ import annotations

import json as _json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from agent.file_safety import explain_write_denial, get_read_block_error


def _build_report(raw_path: str) -> Dict[str, Any]:
    resolved = os.path.realpath(os.path.expanduser(raw_path))
    write_reason = explain_write_denial(raw_path)
    read_reason = get_read_block_error(raw_path)
    return {
        "input_path": raw_path,
        "resolved_path": resolved,
        "exists": Path(resolved).exists(),
        "write": {
            "allowed": write_reason is None,
            "reason": write_reason,
        },
        "read": {
            "allowed": read_reason is None,
            "reason": read_reason,
        },
    }


def _print_human(report: Dict[str, Any]) -> None:
    print(f"path:     {report['input_path']}")
    print(f"resolved: {report['resolved_path']}")
    print(f"exists:   {report['exists']}")
    print()

    write = report["write"]
    if write["allowed"]:
        print("write:    OK (allowed)")
    else:
        print(f"write:    BLOCKED  {write['reason']}")

    read = report["read"]
    if read["allowed"]:
        print("read:     OK (allowed)")
    else:
        print(f"read:     BLOCKED  {read['reason']}")


def file_safety_command(args) -> None:
    """Entry point for ``hermes file-safety check <path>``.

    Exits 0 when both read and write are allowed, 1 when either is blocked.
    """
    raw_path = getattr(args, "path", None)
    if not raw_path:
        print("Path is required. Usage: hermes file-safety check <path>", file=sys.stderr)
        raise SystemExit(2)

    report = _build_report(raw_path)
    if getattr(args, "json", False):
        print(_json.dumps(report, indent=2))
    else:
        _print_human(report)

    blocked = (not report["write"]["allowed"]) or (not report["read"]["allowed"])
    raise SystemExit(1 if blocked else 0)

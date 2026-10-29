"""``hermes doctor --json``: the doctor report as one documented JSON document.

The checks run exactly as in the human report; their rows are collected through the
``doctor_report`` primitives while the human rendering is discarded, so stdout carries only
the document. Shape (``schema_version`` 1) is documented in
website/docs/reference/cli-commands.md § ``hermes doctor``. Exit code is the human report's:
1 while any issue remains.
"""

from __future__ import annotations

import contextlib
import io
import platform
from collections import Counter
from typing import Callable

from hermes_cli.doctor_report import Finding, collect_json_rows

_STATUSES = ("ok", "warn", "fail", "info")


def run_doctor_json(run_checks: Callable[[], Finding]) -> int:
    from hermes_cli.cli_json import print_json_document
    from hermes_constants import get_hermes_home

    with collect_json_rows() as rows, contextlib.redirect_stdout(io.StringIO()):
        total = run_checks()
    counts = Counter(row["status"] for row in rows)
    remaining = bool(total.issues or total.manual_issues)
    print_json_document({
        "hermes_home": str(get_hermes_home()),
        "python_version": platform.python_version(),
        "healthy": not remaining,
        "summary": {status: counts.get(status, 0) for status in _STATUSES},
        "checks": rows,
        "issues": list(total.issues),
        "manual_issues": list(total.manual_issues),
        "fixed": total.fixed,
    })
    return int(remaining)

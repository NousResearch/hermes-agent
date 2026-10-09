"""Desktop-owned backend identity predicates.

This module is intentionally stdlib-only so Hermes' pre-import startup path can use the same
ownership logic as the normal runtime without importing CLI startup machinery.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Mapping, Optional


def is_desktop_ssh_backend_argv(argv: Sequence[str]) -> bool:
    """Whether argv identifies Desktop's SSH backend spawn."""
    return "--ssh-session-token-file" in argv


def is_desktop_owned_backend(
    argv: Optional[Sequence[str]] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether this process is a backend spawned and owned by Desktop.

    HERMES_DESKTOP=1 is inherited by terminal/agent children and is therefore insufficient.
    Local Desktop backends carry HERMES_DASHBOARD_SESSION_TOKEN; SSH backends carry the
    per-spawn token-file argv switch instead.
    """
    env = os.environ if environ is None else environ
    if env.get("HERMES_DESKTOP") != "1":
        return False
    if env.get("HERMES_DASHBOARD_SESSION_TOKEN"):
        return True
    args = sys.argv[1:] if argv is None else argv
    return is_desktop_ssh_backend_argv(args)


__all__ = ["is_desktop_owned_backend", "is_desktop_ssh_backend_argv"]

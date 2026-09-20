"""One argv-safe resolver for GitHub CLI invocations.

``HERMES_GH_BIN`` is intentionally a literal executable path, never a shell
fragment.  When unset, existing installations keep resolving ``gh`` from PATH.
"""
from __future__ import annotations

import os
import shutil
from typing import Optional


def resolve_gh_binary() -> Optional[str]:
    """Return the configured GitHub client or the PATH-resolved ``gh`` binary."""
    configured = os.environ.get("HERMES_GH_BIN", "").strip()
    return configured or shutil.which("gh")


def gh_argv(*args: str) -> Optional[list[str]]:
    """Build a literal argv vector, or ``None`` when no client is available."""
    binary = resolve_gh_binary()
    return [binary, *args] if binary else None

"""Shared helpers for discovering the GitHub CLI (``gh``) and extracting tokens.

Both :pymod:`hermes_cli.auth` and :pymod:`hermes_cli.copilot_auth` need to
locate the ``gh`` binary and run ``gh auth token``.  This module centralises
that logic so it is defined in exactly one place.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def gh_cli_candidates() -> list[str]:
    """Return candidate ``gh`` binary paths, including common Homebrew installs."""
    candidates: list[str] = []

    resolved = shutil.which("gh")
    if resolved:
        candidates.append(resolved)

    for candidate in (
        "/opt/homebrew/bin/gh",
        "/usr/local/bin/gh",
        str(Path.home() / ".local" / "bin" / "gh"),
    ):
        if candidate in candidates:
            continue
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            candidates.append(candidate)

    return candidates


def try_gh_cli_token() -> Optional[str]:
    """Return a token from ``gh auth token`` when the GitHub CLI is available."""
    for gh_path in gh_cli_candidates():
        try:
            result = subprocess.run(
                [gh_path, "auth", "token"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.debug("gh CLI token lookup failed (%s): %s", gh_path, exc)
            continue
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None

"""Persistent-backed dashboard session token loader (issue #53972).

Reads/writes the dashboard session token to disk so a server restart
doesn't mint a fresh value and break TUI-Node children + browser SPA
tabs that hold the prior token via subprocess env / injected SPA HTML.

This module is intentionally standalone (no web_server / fastapi
dependencies) so it is unit-testable in isolation and can be imported
without the dashboard's full dependency stack.
"""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ENV_VAR = "HERMES_DASHBOARD_SESSION_TOKEN"


def _get_token_path(home_path: Path) -> Path:
    """Return the canonical token path under ``$HERMES_HOME/state/``."""
    return home_path / "state" / "dashboard_session_token"


def load_or_create(home_path: Optional[Path] = None) -> str:
    """Return the persistent dashboard session token.

    Behavior:

      1. If ``HERMES_DASHBOARD_SESSION_TOKEN`` env var is set, prefer that
         (operator-injected tokens win — pre-existing contract; matches the
         desktop shell's convention).
      2. Else look for ``$HERMES_HOME/state/dashboard_session_token``; if the
         file exists and is non-empty after stripping, return its content.
      3. Else mint a fresh ``secrets.token_urlsafe(32)``, persist it under
         ``$HERMES_HOME/state/dashboard_session_token`` (parent dir created,
         file mode 0o600 when the filesystem supports it), and return it.

    The ``home_path`` argument is the hermes home directory; if None it is
    resolved lazily through ``hermes_cli.config.get_hermes_home`` so this
    module stays independent of plugin startup on import.

    Best-effort: if the file can't be read or written, an in-memory token
    is still returned so server import doesn't crash.
    """
    if (env_token := os.environ.get(ENV_VAR)):
        return env_token

    if home_path is None:
        try:
            from hermes_cli.config import get_hermes_home
            home_path = get_hermes_home()
        except Exception as exc:
            logger.warning(
                "Could not resolve HERMES_HOME (%s); using in-memory token",
                exc,
            )
            return secrets.token_urlsafe(32)

    token_path = _get_token_path(home_path)

    # Try to reuse an existing persisted token.
    try:
        if token_path.exists():
            existing = token_path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
    except OSError:
        # Read failure (perm denied, race after crash, etc.) — fall through.
        pass

    # Generate a fresh one and persist (best-effort).
    token = secrets.token_urlsafe(32)
    try:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write: tmp file then rename.
        tmp_path = token_path.with_suffix(token_path.suffix + ".tmp")
        tmp_path.write_text(token, encoding="utf-8")
        try:
            tmp_path.chmod(0o600)
        except (PermissionError, NotImplementedError, OSError):
            # Filesystems like Windows reject chmod — not fatal.
            pass
        tmp_path.replace(token_path)
    except OSError as exc:
        logger.warning(
            "Could not persist dashboard session token to %s: %s — "
            "token will rotate on next restart until the file can be written",
            token_path, exc,
        )
    return token
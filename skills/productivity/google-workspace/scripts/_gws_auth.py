#!/usr/bin/env python3
"""Shared helpers for detecting the `gws` CLI and its *own* credentials.

Two distinct credential stores are in play for this skill:

* **Hermes-managed OAuth** — ``~/.hermes/google_token.json``. The skill owns
  this token; ``gws_bridge.py`` and ``google_api.py`` point gws at it via
  ``GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE``.
* **gws-native credentials** — whatever the standalone ``gws`` CLI stores on
  its own (e.g. ``~/.config/gws/``) after the user ran ``gws auth login``
  independently of Hermes.

These helpers answer "does gws have its own valid login?" without touching the
Hermes token, so callers can support a **gws-native auth mode**: if Hermes was
never set up but gws is already authenticated, operations that route through
gws can use gws's credentials directly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess


def gws_binary() -> str | None:
    """Path to the gws CLI, honoring the ``HERMES_GWS_BIN`` override."""
    override = os.getenv("HERMES_GWS_BIN")
    if override:
        return override
    return shutil.which("gws")


def _gws_auth_status() -> dict | None:
    """Return parsed ``gws auth status`` JSON using gws's *own* credentials.

    Deliberately does **not** set ``GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE`` —
    we want gws's native login state, not whatever Hermes would point it at.
    Returns ``None`` if gws is missing, errors, or emits non-JSON.
    """
    binary = gws_binary()
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except (ValueError, TypeError):
        return None


def gws_native_authed() -> bool:
    """True if the gws CLI has its own valid, refreshable credentials."""
    data = _gws_auth_status()
    if not data:
        return False
    return bool(data.get("token_valid")) and bool(data.get("has_refresh_token"))


def gws_live_check() -> tuple[bool, str]:
    """Make a real read-only gws API call to verify native creds actually work.

    Mirrors the cheap call ``google_api.gmail_labels`` makes. Returns
    ``(ok, detail)`` where ``detail`` carries the error on failure. Uses gws's
    own credentials (no ``GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE`` override).
    """
    binary = gws_binary()
    if not binary:
        return False, "gws not installed"
    try:
        result = subprocess.run(
            [binary, "gmail", "users", "labels", "list", "--params", json.dumps({"userId": "me"})],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:  # noqa: BLE001 — surface any subprocess failure to caller
        return False, str(e)
    if result.returncode != 0:
        return False, (result.stderr.strip() or result.stdout.strip() or "gws call failed")
    return True, "gws live API call succeeded"

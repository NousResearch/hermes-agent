"""Codex auth readiness helper for Hermes Docs.

Exposes one public function consumed by the plugin_api.py route:

- ``get_codex_status()`` — returns a sanitised status dict indicating whether
  the local openai-codex auth broker is configured and available.

Design rules
------------
- Never expose raw tokens or API keys in the returned dict.
- Never initiate device-code OAuth or any network call.
- Degrade gracefully: if hermes_cli.auth is unavailable the helper returns a
  deterministic "not configured" response rather than raising.
- ``_status_override`` is a test seam; production code leaves it as ``None``.

This module intentionally has no imports from the Hermes application layer at
module-load time so it can be loaded standalone (via importlib) by plugin_api.py.
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test seam
# ---------------------------------------------------------------------------

# When set to a dict, get_codex_status() returns it directly without touching
# hermes_cli.auth.  Shape matches the function's documented return type.
_status_override: Optional[dict] = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROVIDER_ID = "openai-codex"
_CLI_COMMAND = "hermes auth add openai-codex"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_codex_status() -> dict:
    """Return a sanitised Codex auth status dict.

    The returned dict contains:

    ``provider_id``   str   — always ``"openai-codex"``
    ``configured``    bool  — True when valid credentials are present
    ``available``     bool  — alias for ``configured`` (reserved for future
                              liveness checks without breaking callers)
    ``cli_command``   str   — CLI command to authenticate
    ``token_exposed`` bool  — always False; asserts no secret leaks
    ``detail``        str   — human-readable status line
    ``next_action``   str|None — what the user should do; None when ready
    """
    if _status_override is not None:
        return dict(_status_override)

    raw = _read_auth_status()
    logged_in = bool(raw.get("logged_in"))

    if logged_in:
        # Resolve a human-friendly source description without leaking tokens.
        source = raw.get("source") or raw.get("auth_mode") or "hermes-auth-store"
        auth_store = raw.get("auth_store")
        if auth_store:
            detail = f"Authenticated ({source}; store: {auth_store})"
        else:
            detail = f"Authenticated ({source})"
        next_action = None
    else:
        error = raw.get("error", "")
        detail = (
            f"Not configured. {error}".strip().rstrip(".")
            if error
            else "Not configured"
        )
        next_action = f"Run: {_CLI_COMMAND}"

    return {
        "provider_id": _PROVIDER_ID,
        "configured": logged_in,
        "available": logged_in,
        "cli_command": _CLI_COMMAND,
        "token_exposed": False,
        "detail": detail,
        "next_action": next_action,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_auth_status() -> dict:
    """Call hermes_cli.auth.get_codex_auth_status() safely.

    Returns an empty dict (treated as not-configured) when the auth module is
    unavailable — for example in stripped-down test environments.
    """
    try:
        from hermes_cli import auth as hauth  # noqa: PLC0415
        return hauth.get_codex_auth_status()
    except Exception as exc:  # pragma: no cover
        log.debug("codex_auth_helper: could not read codex auth status: %s", exc)
        return {}

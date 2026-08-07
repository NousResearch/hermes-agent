"""Helpers for reporting Blaxel authentication state."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_BLAXEL_CLI_CONFIG = Path.home() / ".blaxel" / "config.yaml"
_CLI_CREDENTIAL_KEYS = ("apiKey", "access_token", "refresh_token")


@dataclass(frozen=True)
class BlaxelAuthStatus:
    ok: bool
    label: str
    workspace: str
    detail_lines: tuple[str, ...]


def _read_cli_config() -> dict[str, Any]:
    """Return the Blaxel CLI config, or an empty mapping when unreadable.

    Never raises: a missing, unparseable, or permission-denied CLI config just
    means "no CLI credentials", which is a normal state.
    """
    try:
        import yaml
    except ImportError:
        return {}
    try:
        raw = _BLAXEL_CLI_CONFIG.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        parsed = yaml.safe_load(raw) or {}
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _cli_workspace_names(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = config.get("workspaces")
    if not isinstance(entries, list):
        return {}
    found: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name"):
            found[str(entry["name"])] = entry
    return found


def _has_cli_credentials(entry: dict[str, Any]) -> bool:
    credentials = entry.get("credentials")
    if not isinstance(credentials, dict):
        return False
    return any(credentials.get(key) for key in _CLI_CREDENTIAL_KEYS)


def resolve_blaxel_workspace() -> str:
    """Return the workspace Blaxel will target, or an empty string.

    ``BL_WORKSPACE`` wins. Otherwise fall back to the workspace the CLI has
    selected, which is what the SDK itself resolves.
    """
    explicit = (os.getenv("BL_WORKSPACE") or "").strip()
    if explicit:
        return explicit
    config = _read_cli_config()
    context = config.get("context")
    if isinstance(context, dict):
        return str(context.get("workspace") or "").strip()
    return ""


def describe_blaxel_auth(workspace: Optional[str] = None) -> BlaxelAuthStatus:
    """Return Blaxel auth status without exposing secret values.

    Two supported paths, mirroring how the Blaxel SDK itself authenticates:

    1. ``BL_API_KEY`` in the environment. This is the documented path for
       deployments and any long-running non-interactive Hermes process.
    2. Credentials already stored by ``bl login`` for the target workspace.
       This covers ordinary local development, where requiring a separate API
       key would reject a working setup.
    """
    resolved = (workspace or resolve_blaxel_workspace()).strip()
    if not resolved:
        return BlaxelAuthStatus(
            ok=False,
            label="no workspace resolved",
            workspace="",
            detail_lines=(
                "Set BL_WORKSPACE, or select one with `bl login <workspace>`.",
            ),
        )

    if os.getenv("BL_API_KEY"):
        return BlaxelAuthStatus(
            ok=True,
            label="BL_API_KEY",
            workspace=resolved,
            detail_lines=(),
        )

    entry = _cli_workspace_names(_read_cli_config()).get(resolved)
    if entry and _has_cli_credentials(entry):
        return BlaxelAuthStatus(
            ok=True,
            label=f"Blaxel CLI credentials for {resolved}",
            workspace=resolved,
            detail_lines=(
                "Set BL_API_KEY for deployments; CLI credentials can expire.",
            ),
        )

    return BlaxelAuthStatus(
        ok=False,
        label="no credentials found",
        workspace=resolved,
        detail_lines=(
            f"Run `bl login {resolved}`, or set BL_API_KEY.",
        ),
    )

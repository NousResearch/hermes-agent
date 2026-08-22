"""Resolve the *installed* codex CLI semver for chatgpt.com backend calls.

The Cloudflare layer in front of ``chatgpt.com/backend-api/codex/*`` allowlists
requests whose ``originator`` is one of ``codex_cli_rs`` / ``codex_vscode`` /
``codex_sdk_ts`` (or starts with ``Codex``) and whose ``User-Agent`` is shaped
like ``codex_cli_rs/MAJOR.MINOR.PATCH``. The same value is sent as the
``client_version`` query parameter on ``/models`` and friends, and mirrored to
the local app-server ``initialize`` handshake so codex's own diagnostics see a
consistent peer.

Per upstream openai/codex (``codex-rs/models-manager/src/lib.rs``), the value
is just the codex CLI's own ``CARGO_PKG_VERSION`` (major.minor.patch; any
prerelease suffix is stripped). To present an *authentic* identity we resolve
it from the actual codex executable Hermes is configured to drive, rather than
guessing a latest-release number:

  1. ``HERMES_CODEX_CLI_VERSION`` env var: operator override.
  2. ``<codex_bin> --version`` on the configured executable, parsed with the
     same ``parse_codex_version`` the app-server startup check already uses.
  3. Hard-coded fallback constant (codex CLI not installed / not on PATH).

Errors are swallowed; resolution always returns a value. The subprocess runs
under a tight timeout and its result is memoized per ``codex_bin`` so this is
safe to invoke on hot paths.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Last-resort fallback used only when the codex CLI cannot be invoked (not
# installed, not on PATH, or ``--version`` fails/parses to nothing). Kept in
# step with ``MIN_CODEX_VERSION`` in ``agent.transports.codex_app_server``.
_FALLBACK_CODEX_CLI_VERSION = "0.136.0"

_VERSION_QUERY_TIMEOUT_SECONDS = 10.0

# Per-executable memo so repeated hot-path calls do not re-spawn ``codex``.
# Keyed by the resolved ``codex_bin`` string; value is the MAJOR.MINOR.PATCH.
_memo: dict[str, str] = {}


def _default_codex_bin() -> str:
    """Return the codex executable name/path Hermes drives.

    Mirrors the ``codex_bin`` default used by ``CodexAppServerSession`` /
    ``check_codex_binary`` ("codex", resolved on PATH). An explicit override
    via ``HERMES_CODEX_BIN`` is honored so a non-PATH install still yields an
    authentic version string.
    """
    return (os.environ.get("HERMES_CODEX_BIN") or "codex").strip() or "codex"


def _query_installed_version(codex_bin: str) -> Optional[str]:
    """Run ``<codex_bin> --version`` and return MAJOR.MINOR.PATCH, or None.

    Reuses ``parse_codex_version`` (the same parser the app-server binary
    check uses) so we accept the real ``codex-cli X.Y.Z`` output shape and
    tolerate trailing metadata. Any failure (missing binary, non-zero exit,
    timeout, unparseable output) returns None so the caller can fall back.
    """
    try:
        from agent.transports.codex_app_server import parse_codex_version

        proc = subprocess.run(
            [codex_bin, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_VERSION_QUERY_TIMEOUT_SECONDS,
            stdin=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            logger.debug(
                "codex_version: %r --version exited %s",
                codex_bin,
                proc.returncode,
            )
            return None
        parsed = parse_codex_version(proc.stdout)
        if parsed is None:
            return None
        return ".".join(str(part) for part in parsed)
    except FileNotFoundError:
        logger.debug("codex_version: %r not found on PATH", codex_bin)
        return None
    except Exception as exc:  # subprocess timeout, import error, etc.
        logger.debug("codex_version: version query failed: %s", exc)
        return None


def get_codex_cli_version(codex_bin: Optional[str] = None) -> str:
    """Return the installed codex CLI semver to advertise on backend calls.

    Always returns a ``MAJOR.MINOR.PATCH`` string. An operator override wins;
    otherwise the version is read from the configured codex executable and
    memoized per binary. When the CLI cannot be invoked or parsed, the
    fallback constant is returned so identity resolution never raises on a
    hot path.
    """
    override = os.environ.get("HERMES_CODEX_CLI_VERSION", "").strip()
    if override:
        # Normalize the override to MAJOR.MINOR.PATCH via the same parser.
        try:
            from agent.transports.codex_app_server import parse_codex_version

            match = parse_codex_version(override)
            if match is not None:
                return ".".join(str(part) for part in match)
        except Exception:
            pass
        return override

    resolved_bin = (codex_bin or _default_codex_bin()).strip() or "codex"
    cached = _memo.get(resolved_bin)
    if cached is not None:
        return cached

    version = _query_installed_version(resolved_bin) or _FALLBACK_CODEX_CLI_VERSION
    _memo[resolved_bin] = version
    return version


__all__ = ["get_codex_cli_version"]

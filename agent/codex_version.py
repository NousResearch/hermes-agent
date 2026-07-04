"""Resolve the current codex CLI semver for chatgpt.com backend calls.

The Cloudflare layer in front of ``chatgpt.com/backend-api/codex/*`` allowlists
requests whose ``originator`` is one of ``codex_cli_rs`` / ``codex_vscode`` /
``codex_sdk_ts`` (or starts with ``Codex``) and whose ``User-Agent`` is shaped
like ``codex_cli_rs/MAJOR.MINOR.PATCH``. The same value is sent as the
``client_version`` query parameter on ``/models`` and friends.

Per upstream openai/codex (``codex-rs/models-manager/src/lib.rs``), the value
is just the codex CLI's own ``CARGO_PKG_VERSION`` (major.minor.patch; any
prerelease suffix is stripped). It is used as the ``models_cache.json`` cache
key, not an API contract version.

We resolve it dynamically so Hermes always advertises a plausible, current
codex CLI version without manual upkeep:

  1. ``HERMES_CODEX_CLI_VERSION`` env var: operator override.
  2. On-disk cache (``~/.cache/hermes/codex_version.json``) if fresh.
  3. GitHub releases API for ``openai/codex``: parse ``tag_name`` for the
     first ``MAJOR.MINOR.PATCH`` substring (tags are shaped ``rust-v0.136.0``).
  4. Hard-coded fallback constant.

Errors are swallowed; resolution always returns a value. Network calls run
under a tight timeout so this is safe to invoke on hot paths.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Last-resort fallback. Bump occasionally; the live GitHub lookup will cover
# the gap between bumps in normal operation. Verified live against npm
# ``@openai/codex@latest`` at the time this constant was last touched.
_FALLBACK_CODEX_CLI_VERSION = "0.136.0"

_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24h
_MEMO_TTL_SECONDS = 60  # in-process memoization
_GITHUB_RELEASES_URL = "https://api.github.com/repos/openai/codex/releases/latest"
_HTTP_TIMEOUT_SECONDS = 3.0

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")

_memo: tuple[float, str] | None = None  # (resolved_at_monotonic, version)


def _cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "hermes" / "codex_version.json"


def _read_cache() -> str | None:
    try:
        path = _cache_path()
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        ts = float(data.get("fetched_at", 0))
        ver = str(data.get("version", "")).strip()
        if not ver or not _VERSION_RE.fullmatch(ver):
            return None
        if time.time() - ts > _CACHE_TTL_SECONDS:
            return None
        return ver
    except Exception as exc:
        logger.debug("codex_version: cache read failed: %s", exc)
        return None


def _write_cache(version: str) -> None:
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": version, "fetched_at": time.time()})
        )
    except Exception as exc:
        logger.debug("codex_version: cache write failed: %s", exc)


def _fetch_github_release() -> str | None:
    """Fetch the latest openai/codex release tag and extract MAJOR.MINOR.PATCH.

    Uses ``urllib`` (stdlib) so this module has no third-party import cost on
    the cold path. GitHub release tags for openai/codex are shaped
    ``rust-v0.136.0``. We match the first ``\\d+\\.\\d+\\.\\d+`` substring,
    which tolerates either ``rust-v`` or ``v`` prefixes (or none).
    """
    try:
        import urllib.request

        req = urllib.request.Request(
            _GITHUB_RELEASES_URL,
            headers={
                "Accept": "application/vnd.github+json",
                # Identify ourselves to GitHub for rate-limit accounting.
                # Anonymous calls are limited to 60/hr/IP, which is fine for
                # this usage (one call per 24h per host).
                "User-Agent": "hermes-codex-version-resolver",
            },
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        tag = str(payload.get("tag_name") or payload.get("name") or "")
        match = _VERSION_RE.search(tag)
        if not match:
            return None
        return ".".join(match.groups())
    except Exception as exc:
        logger.debug("codex_version: github fetch failed: %s", exc)
        return None


def _normalize(version: str) -> str:
    """Coerce a version-like string to ``MAJOR.MINOR.PATCH`` (drops prerelease)."""
    match = _VERSION_RE.search(version)
    return ".".join(match.groups()) if match else _FALLBACK_CODEX_CLI_VERSION


def get_codex_cli_version() -> str:
    """Return the codex CLI semver to advertise on chatgpt.com backend calls.

    Always returns a ``MAJOR.MINOR.PATCH`` string. Network failures, cache
    errors, and missing dependencies are non-fatal; the fallback constant
    is returned instead. Within a single process, the result is memoized for
    ``_MEMO_TTL_SECONDS`` to keep this safe on hot paths.
    """
    global _memo
    now = time.monotonic()
    if _memo is not None and now - _memo[0] < _MEMO_TTL_SECONDS:
        return _memo[1]

    override = os.environ.get("HERMES_CODEX_CLI_VERSION", "").strip()
    if override:
        version = _normalize(override)
        _memo = (now, version)
        return version

    cached = _read_cache()
    if cached:
        _memo = (now, cached)
        return cached

    fetched = _fetch_github_release()
    if fetched:
        _write_cache(fetched)
        _memo = (now, fetched)
        return fetched

    _memo = (now, _FALLBACK_CODEX_CLI_VERSION)
    return _FALLBACK_CODEX_CLI_VERSION


__all__ = ["get_codex_cli_version"]

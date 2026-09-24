"""Small, non-interactive GitHub REST helpers for best-effort callers.

Public endpoints work anonymously, but passive update checks should use a
credential the user has already configured so they do not exhaust the
anonymous GitHub rate limit.  Credentials never appear in URLs or logs.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from typing import Any, Optional
from urllib.error import HTTPError
import urllib.request
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

_UNSET = object()
_gh_cli_token: Optional[str] | object = _UNSET
_gh_cli_token_lock = threading.Lock()


def _env_token() -> Optional[str]:
    """Return an explicitly exported GitHub token in documented precedence."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or None


def _cached_gh_cli_token() -> Optional[str]:
    """Read ``gh auth token`` once, without letting exported tokens mask its login."""
    global _gh_cli_token
    with _gh_cli_token_lock:
        if _gh_cli_token is not _UNSET:
            return _gh_cli_token
        gh = shutil.which("gh")
        if not gh:
            _gh_cli_token = None
            return None
        env = dict(os.environ)
        env.pop("GITHUB_TOKEN", None)
        env.pop("GH_TOKEN", None)
        env["GH_PROMPT_DISABLED"] = "1"
        try:
            result = subprocess.run(
                [gh, "auth", "token"], stdin=subprocess.DEVNULL, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=5, env=env,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.debug("GitHub CLI credential lookup failed: %s", exc)
            _gh_cli_token = None
            return None
        _gh_cli_token = result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None
        return _gh_cli_token


def github_token() -> Optional[str]:
    """Return a GitHub API credential, preferring environment configuration."""
    return _env_token() or _cached_gh_cli_token()


def _request(url: str, *, accept: str, token: Optional[str]) -> bytes:
    headers = {"Accept": accept, "User-Agent": "hermes-cli-update-check"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=10) as response:
        return response.read()


def _log_http_failure(error: HTTPError) -> None:
    if error.code in {403, 429} and (error.headers or {}).get("X-RateLimit-Remaining") == "0":
        logger.debug("GitHub API rate limit reached; the passive update check will retry later")
    else:
        logger.debug("GitHub API request failed with HTTP %s: %s", error.code, error.reason)


def get_text(url: str, *, accept: str = "application/vnd.github+json") -> str:
    """Fetch a GitHub REST resource, retrying anonymously if a token is rejected."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "api.github.com":
        raise ValueError("GitHub API requests must use https://api.github.com")
    token = github_token()
    try:
        return _request(url, accept=accept, token=token).decode("utf-8")
    except HTTPError as error:
        if token and error.code == 401:
            logger.debug("GitHub API credential was rejected; retrying this public request anonymously")
            try:
                return _request(url, accept=accept, token=None).decode("utf-8")
            except HTTPError as retry_error:
                _log_http_failure(retry_error)
                raise
        _log_http_failure(error)
        raise


def get_json(url: str, *, accept: str = "application/vnd.github+json") -> Any:
    """Fetch and decode a JSON GitHub REST response."""
    return json.loads(get_text(url, accept=accept))


def _reset_github_token_cache() -> None:
    """Reset cached CLI credentials for tests."""
    global _gh_cli_token
    with _gh_cli_token_lock:
        _gh_cli_token = _UNSET

"""Credentials and failure copy for the update check's api.github.com calls.

The passive source check asks the REST API for a branch tip. Unauthenticated
that budget is 60 requests/hour keyed on the *client IP*, so a shared exit
(office NAT, VPN, proxy) exhausts it for everyone behind it and the check
reports a 403 that read as "Hermes can't reach the update server".
Authenticating moves the caller onto the token's 5,000/hour budget.

Credential ladder: GITHUB_TOKEN, then GH_TOKEN, then ``gh auth token`` (the
gh CLI's own login — the only rung most desktop users have, since a
GUI-launched app inherits a minimal environment; its answer is cached per
process so the check never spawns gh more than once), then anonymous. A
token GitHub rejects (401) drops that request to anonymous. The token itself
never reaches a log line or an error string.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

GITHUB_TOKEN_ENV_VARS = ("GITHUB_TOKEN", "GH_TOKEN")
_GH_CLI_TIMEOUT_SECONDS = 3
_gh_cli_cache: Optional[str] = None
_gh_cli_probed = False


def github_token_from_env(env=os.environ) -> Optional[str]:
    """First non-blank env token, trimmed. A blank value falls through to the next."""
    for name in GITHUB_TOKEN_ENV_VARS:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return None


def _gh_cli_token() -> Optional[str]:
    global _gh_cli_cache, _gh_cli_probed
    if _gh_cli_probed:
        return _gh_cli_cache
    _gh_cli_probed = True
    from hermes_cli._subprocess_compat import windows_hide_flags
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_GH_CLI_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("gh CLI token lookup unavailable: %s", exc)
        return None
    token = result.stdout.strip() if result.returncode == 0 else ""
    _gh_cli_cache = token or None
    return _gh_cli_cache


def github_token() -> Optional[str]:
    """The credential for this request, or None for anonymous."""
    return github_token_from_env() or _gh_cli_token()


def describe_github_failure(exc: BaseException, authenticated: bool, now: Optional[float] = None) -> str:
    """One line a user can act on instead of a generic "could not resolve" (#105855).

    A 403/429 with ``x-ratelimit-remaining: 0`` is the anonymous per-IP budget spent
    by every client behind one exit — the line names the fix the user actually has
    (a GITHUB_TOKEN) and the real reset time. Any other status is reported as what it is.
    """
    if isinstance(exc, urllib.error.HTTPError):
        status = exc.code
        headers = exc.headers or {}
        remaining = _header_int(headers, "x-ratelimit-remaining")
        if status in (403, 429) and remaining == 0:
            reset = _header_int(headers, "x-ratelimit-reset")
            when = "within an hour"
            if reset is not None:
                minutes = int((reset - (time.time() if now is None else now) + 59) // 60)
                if minutes >= 1:
                    when = "in about a minute" if minutes == 1 else f"in about {minutes} minutes"
            if authenticated:
                return f"GitHub API rate limit reached for your GITHUB_TOKEN (HTTP {status}) — it resets {when}."
            return (f"GitHub API rate limit reached (HTTP {status}): anonymous requests are limited to 60 per hour "
                    f"per network address, shared with everyone behind the same connection. It resets {when}; "
                    "setting GITHUB_TOKEN in the environment lifts the limit.")
        if status >= 500:
            return (f"GitHub is having trouble (HTTP {status} from api.github.com) — check githubstatus.com "
                    "and try again later.")
        return f"api.github.com answered HTTP {status}."
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            return "api.github.com did not answer within 10 seconds."
        return f"Connection to api.github.com failed ({reason}) — check your connection, firewall or proxy."
    if isinstance(exc, TimeoutError):
        return "api.github.com did not answer within 10 seconds."
    return f"api.github.com: {exc}"


def _header_int(headers, name: str) -> Optional[int]:
    try:
        return int(str(headers.get(name, "")).strip())
    except (TypeError, ValueError):
        return None

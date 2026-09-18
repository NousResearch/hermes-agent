"""Non-interactive HTTPS credentials for Hermes's internal git clones (private plugin/MCP/profile repos).

:func:`noninteractive_git_env` deliberately disables credential helpers, askpass and global git
config so a hostile repo cannot make our plumbing prompt or hang. The cost is that a *private*
repo the user can already clone from their shell fails inside ``hermes plugins install`` with
"could not read Username" (or hangs on a GUI askpass until the timeout). This module resolves a
credential up front, from sources the user already owns, and passes it to git as a one-shot
``http.<origin>/.extraheader`` in the environment — never in the URL and never in ``.git/config``,
so nothing is persisted into the installed checkout.

Resolution order for an ``https://`` URL:

1. ``GITHUB_TOKEN`` / ``GH_TOKEN`` (profile-scoped, GitHub hosts only).
2. ``gh auth token`` (GitHub hosts only; the gh CLI's own login).
3. ``git credential fill`` against the user's configured credential helpers (any host: GitLab,
   Bitbucket, self-hosted) with prompting disabled, so a stored credential is returned and a
   missing one fails in ~100 ms instead of asking.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import urllib.parse
from typing import Mapping, Optional

from hermes_cli._subprocess_compat import noninteractive_git_env, windows_hide_flags

logger = logging.getLogger(__name__)

_GITHUB_HOSTS = {"github.com", "gist.github.com"}


def _https_origin(url: str) -> Optional[str]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return f"https://{host}"


def _github_token() -> Optional[str]:
    from agent.secret_scope import get_secret

    token = get_secret("GITHUB_TOKEN") or get_secret("GH_TOKEN")
    if token:
        return token
    gh = shutil.which("gh")
    if not gh:
        return None
    try:
        env = noninteractive_git_env()
        env["GH_PROMPT_DISABLED"] = "1"
        result = subprocess.run(
            [gh, "auth", "token"], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10, stdin=subprocess.DEVNULL, env=env, creationflags=windows_hide_flags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("gh auth token lookup failed: %s", exc)
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _credential_fill(origin: str) -> Optional[tuple[str, str]]:
    """``(username, password)`` from the user's own git credential helpers, never prompting."""
    git = shutil.which("git")
    if not git:
        return None
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    # A GUI askpass (VS Code, ssh-askpass) would block on a dialog nobody sees.
    env.pop("GIT_ASKPASS", None)
    env.pop("SSH_ASKPASS", None)
    parsed = urllib.parse.urlsplit(origin)
    request = f"protocol=https\nhost={parsed.netloc}\n\n"
    try:
        result = subprocess.run(
            [git, "-c", "core.askPass=", "credential", "fill"], input=request, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=15, env=env,
            creationflags=windows_hide_flags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("git credential fill failed for %s: %s", origin, exc)
        return None
    if result.returncode != 0:
        return None
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if fields.get("password"):
        return fields.get("username", ""), fields["password"]
    return None


def resolve_git_basic_auth(url: str) -> Optional[tuple[str, str]]:
    """``(username, password)`` for *url*, or None for non-HTTPS URLs / no stored credential."""
    origin = _https_origin(url)
    if origin is None:
        return None
    if urllib.parse.urlsplit(origin).hostname in _GITHUB_HOSTS:
        token = _github_token()
        if token:
            return "x-access-token", token
    return _credential_fill(origin)


def with_git_auth(env: Mapping[str, str], url: str) -> dict[str, str]:
    """Copy of *env* (a :func:`noninteractive_git_env` result) that authenticates HTTPS requests to
    *url*'s origin via a ``GIT_CONFIG_*`` ``http.<origin>/.extraheader`` entry when a credential is
    available; unchanged otherwise. The header lives only in this process environment."""
    env = dict(env)
    origin = _https_origin(url)
    if origin is None:
        return env
    auth = resolve_git_basic_auth(url)
    if auth is None:
        return env
    encoded = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
    idx = int(env.get("GIT_CONFIG_COUNT", "0") or 0)
    env[f"GIT_CONFIG_KEY_{idx}"] = f"http.{origin}/.extraheader"
    env[f"GIT_CONFIG_VALUE_{idx}"] = f"Authorization: basic {encoded}"
    env["GIT_CONFIG_COUNT"] = str(idx + 1)
    return env

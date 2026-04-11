"""GitHub Copilot authentication utilities.

Implements the OAuth device code flow used by the Copilot CLI and handles
token validation/exchange for the Copilot API.

GitHub OAuth tokens (gho_*, github_pat_*, ghu_*) cannot be used directly
with the Copilot inference API.  They must first be exchanged for a
short-lived Copilot session token via the internal token endpoint
(``https://api.github.com/copilot_internal/v2/token``).  The session
token (``tid=...``) is what the inference API actually accepts.

Token type support (per GitHub docs):
  gho_          OAuth token           ✓  (default via copilot login)
  github_pat_   Fine-grained PAT      ✓  (needs Copilot Requests permission)
  ghu_          GitHub App token      ✓  (via environment variable)
  ghp_          Classic PAT           ✗  NOT SUPPORTED

Credential search order (matching Copilot CLI behaviour):
  1. COPILOT_GITHUB_TOKEN env var
  2. GH_TOKEN env var
  3. GITHUB_TOKEN env var
  4. gh auth token  CLI fallback
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# OAuth device code flow constants (same client ID as opencode/Copilot CLI)
COPILOT_OAUTH_CLIENT_ID = "Ov23li8tweQw6odWQebz"
# Token type prefixes
_CLASSIC_PAT_PREFIX = "ghp_"
_SUPPORTED_PREFIXES = ("gho_", "github_pat_", "ghu_")

# Env var search order (matches Copilot CLI)
COPILOT_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")

# Copilot token exchange endpoint (same as opencode, openclaw, VS Code)
COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"

# Default base URL when proxy-ep is absent from the session token
DEFAULT_COPILOT_API_BASE_URL = "https://api.individual.githubcopilot.com"

# Minimum remaining lifetime (seconds) before we consider a cached token stale
_TOKEN_REFRESH_MARGIN = 300  # 5 minutes

# Polling constants
_DEVICE_CODE_POLL_INTERVAL = 5  # seconds
_DEVICE_CODE_POLL_SAFETY_MARGIN = 3  # seconds


# ─── Session Token Cache ──────────────────────────────────────────────────


@dataclass
class CopilotSessionToken:
    """A Copilot API session token with metadata."""

    token: str  # the tid=... string used as Bearer token
    expires_at: float  # unix timestamp in seconds
    base_url: str  # API base URL derived from proxy-ep
    source: str  # where this came from (cache, fetched, etc.)


def _get_token_cache_path() -> Path:
    """Return the path to the cached Copilot session token."""
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "credentials" / "github-copilot.token.json"
    except ImportError:
        return Path.home() / ".hermes" / "credentials" / "github-copilot.token.json"


def _load_cached_session_token() -> Optional[CopilotSessionToken]:
    """Load a cached Copilot session token if it exists and is still usable."""
    cache_path = _get_token_cache_path()
    if not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        token = data.get("token", "")
        expires_at = data.get("expiresAt", 0)
        if not isinstance(token, str) or not token.strip():
            return None
        # Convert ms to seconds if needed (openclaw stores ms)
        if isinstance(expires_at, (int, float)) and expires_at > 1e11:
            expires_at = expires_at / 1000.0
        if time.time() > expires_at - _TOKEN_REFRESH_MARGIN:
            logger.debug("Cached Copilot session token is expired or near-expiry")
            return None
        base_url = _derive_base_url_from_token(token)
        return CopilotSessionToken(
            token=token,
            expires_at=expires_at,
            base_url=base_url,
            source=f"cache:{cache_path}",
        )
    except Exception as exc:
        logger.debug("Failed to load cached Copilot token: %s", exc)
        return None


def _save_session_token(token: str, expires_at: float) -> None:
    """Persist a Copilot session token to disk."""
    cache_path = _get_token_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token": token,
        "expiresAt": int(expires_at * 1000),  # store as ms for openclaw compat
        "updatedAt": int(time.time() * 1000),
    }
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    # Restrict permissions — token is a secret
    try:
        cache_path.chmod(0o600)
    except OSError:
        pass


def _derive_base_url_from_token(token: str) -> str:
    """Extract the API base URL from the proxy-ep field in a session token.

    Session tokens look like:
      tid=...;proxy-ep=proxy.enterprise.githubcopilot.com;...

    The proxy-ep value is the proxy host.  The API host is derived by
    replacing the ``proxy.`` prefix with ``api.``.  This matches the
    logic in opencode and openclaw.
    """
    match = re.search(r"(?:^|;)\s*proxy-ep=([^;\s]+)", token, re.IGNORECASE)
    if not match:
        return DEFAULT_COPILOT_API_BASE_URL
    proxy_ep = match.group(1).strip()
    if not proxy_ep:
        return DEFAULT_COPILOT_API_BASE_URL
    # Ensure it looks like a URL
    if not proxy_ep.startswith(("http://", "https://")):
        proxy_ep = f"https://{proxy_ep}"
    try:
        from urllib.parse import urlparse

        parsed = urlparse(proxy_ep)
        host = parsed.hostname or ""
    except Exception:
        return DEFAULT_COPILOT_API_BASE_URL
    if not host:
        return DEFAULT_COPILOT_API_BASE_URL
    # proxy.enterprise.githubcopilot.com -> api.enterprise.githubcopilot.com
    api_host = re.sub(r"^proxy\.", "api.", host, flags=re.IGNORECASE)
    return f"https://{api_host}"


# ─── Token Exchange ───────────────────────────────────────────────────────


def exchange_github_token_for_copilot_session(
    github_token: str,
) -> CopilotSessionToken:
    """Exchange a GitHub OAuth/PAT token for a Copilot API session token.

    This is the critical step that opencode, openclaw, and VS Code all
    perform.  The raw GitHub token (gho_*, github_pat_*, ghu_*) is sent
    to ``https://api.github.com/copilot_internal/v2/token`` and a
    short-lived session token (``tid=...``) is returned.

    Raises RuntimeError on failure.
    """
    # Check cache first
    cached = _load_cached_session_token()
    if cached:
        logger.debug(
            "Using cached Copilot session token (expires %.0f)", cached.expires_at
        )
        return cached

    # Exchange
    req = urllib.request.Request(
        COPILOT_TOKEN_URL,
        method="GET",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {github_token}",
            "Editor-Version": "vscode/1.104.1",
            "User-Agent": "HermesAgent/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode()[:500]
        except Exception:
            pass
        raise RuntimeError(
            f"Copilot token exchange failed: HTTP {exc.code} — {body}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Copilot token exchange failed: {exc}") from exc

    token = data.get("token", "")
    expires_at_raw = data.get("expires_at")

    if not isinstance(token, str) or not token.strip():
        raise RuntimeError("Copilot token exchange returned empty token")

    # Parse expires_at (can be int seconds or string)
    if isinstance(expires_at_raw, (int, float)):
        expires_at = float(expires_at_raw)
    elif isinstance(expires_at_raw, str):
        try:
            expires_at = float(expires_at_raw)
        except ValueError:
            expires_at = time.time() + 1800  # fallback: 30 min
    else:
        expires_at = time.time() + 1800

    # Normalize: if < 1e11 it's seconds, otherwise ms
    if expires_at > 1e11:
        expires_at = expires_at / 1000.0

    # Cache
    _save_session_token(token, expires_at)

    base_url = _derive_base_url_from_token(token)
    logger.info(
        "Copilot session token acquired (expires in %.0f min, base_url=%s)",
        (expires_at - time.time()) / 60,
        base_url,
    )

    return CopilotSessionToken(
        token=token,
        expires_at=expires_at,
        base_url=base_url,
        source=f"fetched:{COPILOT_TOKEN_URL}",
    )


def validate_copilot_token(token: str) -> tuple[bool, str]:
    """Validate that a token is usable with the Copilot API.

    Returns (valid, message).
    """
    token = token.strip()
    if not token:
        return False, "Empty token"

    if token.startswith(_CLASSIC_PAT_PREFIX):
        return False, (
            "Classic Personal Access Tokens (ghp_*) are not supported by the "
            "Copilot API. Use one of:\n"
            "  → `copilot login` or `hermes model` to authenticate via OAuth\n"
            "  → A fine-grained PAT (github_pat_*) with Copilot Requests permission\n"
            "  → `gh auth login` with the default device code flow (produces gho_* tokens)"
        )

    return True, "OK"


def resolve_copilot_token() -> tuple[str, str]:
    """Resolve a Copilot API session token ready for inference calls.

    This performs the full resolution chain:
      1. Find a GitHub token (env vars → gh CLI)
      2. Exchange it for a Copilot session token (with caching)

    Returns (session_token, source) where session_token is the ``tid=...``
    string usable as a Bearer token against the Copilot API.

    Raises ValueError if only a classic PAT is available.
    """
    # 1. Check env vars in priority order
    github_token = ""
    github_source = ""
    for env_var in COPILOT_ENV_VARS:
        val = os.getenv(env_var, "").strip()
        if val:
            valid, msg = validate_copilot_token(val)
            if not valid:
                logger.warning("Token from %s is not supported: %s", env_var, msg)
                continue
            github_token = val
            github_source = env_var
            break

    # 2. Fall back to gh auth token
    if not github_token:
        token = _try_gh_cli_token()
        if token:
            valid, msg = validate_copilot_token(token)
            if not valid:
                raise ValueError(
                    f"Token from `gh auth token` is a classic PAT (ghp_*). {msg}"
                )
            github_token = token
            github_source = "gh auth token"

    if not github_token:
        return "", ""

    # 3. Exchange for Copilot session token
    try:
        session = exchange_github_token_for_copilot_session(github_token)
        return session.token, f"copilot-session via {github_source}"
    except RuntimeError as exc:
        logger.error("Copilot token exchange failed: %s", exc)
        # Return empty so the caller can fall back
        return "", ""


def _gh_cli_candidates() -> list[str]:
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


def _try_gh_cli_token() -> Optional[str]:
    """Return a token from ``gh auth token`` when the GitHub CLI is available."""
    for gh_path in _gh_cli_candidates():
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


# ─── OAuth Device Code Flow ────────────────────────────────────────────────


def copilot_device_code_login(
    *,
    host: str = "github.com",
    timeout_seconds: float = 300,
) -> Optional[str]:
    """Run the GitHub OAuth device code flow for Copilot.

    Prints instructions for the user, polls for completion, and returns
    the OAuth access token on success, or None on failure/cancellation.

    This replicates the flow used by opencode and the Copilot CLI.
    """
    import urllib.request
    import urllib.parse

    domain = host.rstrip("/")
    device_code_url = f"https://{domain}/login/device/code"
    access_token_url = f"https://{domain}/login/oauth/access_token"

    # Step 1: Request device code
    data = urllib.parse.urlencode(
        {
            "client_id": COPILOT_OAUTH_CLIENT_ID,
            "scope": "read:user",
        }
    ).encode()

    req = urllib.request.Request(
        device_code_url,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "HermesAgent/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            device_data = json.loads(resp.read().decode())
    except Exception as exc:
        logger.error("Failed to initiate device authorization: %s", exc)
        print(f"  ✗ Failed to start device authorization: {exc}")
        return None

    verification_uri = device_data.get(
        "verification_uri", "https://github.com/login/device"
    )
    user_code = device_data.get("user_code", "")
    device_code = device_data.get("device_code", "")
    interval = max(device_data.get("interval", _DEVICE_CODE_POLL_INTERVAL), 1)

    if not device_code or not user_code:
        print("  ✗ GitHub did not return a device code.")
        return None

    # Step 2: Show instructions
    print()
    print(f"  Open this URL in your browser: {verification_uri}")
    print(f"  Enter this code: {user_code}")
    print()
    print("  Waiting for authorization...", end="", flush=True)

    # Step 3: Poll for completion
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        time.sleep(interval + _DEVICE_CODE_POLL_SAFETY_MARGIN)

        poll_data = urllib.parse.urlencode(
            {
                "client_id": COPILOT_OAUTH_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }
        ).encode()

        poll_req = urllib.request.Request(
            access_token_url,
            data=poll_data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "HermesAgent/1.0",
            },
        )

        try:
            with urllib.request.urlopen(poll_req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
        except Exception:
            print(".", end="", flush=True)
            continue

        if result.get("access_token"):
            print(" ✓")
            return result["access_token"]

        error = result.get("error", "")
        if error == "authorization_pending":
            print(".", end="", flush=True)
            continue
        elif error == "slow_down":
            # RFC 8628: add 5 seconds to polling interval
            server_interval = result.get("interval")
            if isinstance(server_interval, (int, float)) and server_interval > 0:
                interval = int(server_interval)
            else:
                interval += 5
            print(".", end="", flush=True)
            continue
        elif error == "expired_token":
            print()
            print("  ✗ Device code expired. Please try again.")
            return None
        elif error == "access_denied":
            print()
            print("  ✗ Authorization was denied.")
            return None
        elif error:
            print()
            print(f"  ✗ Authorization failed: {error}")
            return None

    print()
    print("  ✗ Timed out waiting for authorization.")
    return None


# ─── Copilot API Headers ───────────────────────────────────────────────────


def copilot_request_headers(
    *,
    is_agent_turn: bool = True,
    is_vision: bool = False,
) -> dict[str, str]:
    """Build the standard headers for Copilot API requests.

    Replicates the header set used by opencode and the Copilot CLI.
    """
    headers: dict[str, str] = {
        "Editor-Version": "vscode/1.104.1",
        "User-Agent": "HermesAgent/1.0",
        "Copilot-Integration-Id": "vscode-chat",
        "Openai-Intent": "conversation-edits",
        "x-initiator": "agent" if is_agent_turn else "user",
    }
    if is_vision:
        headers["Copilot-Vision-Request"] = "true"

    return headers

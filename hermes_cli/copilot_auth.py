"""GitHub Copilot authentication utilities.

Implements the OAuth device code flow used by the Copilot CLI and handles
token validation/exchange for the Copilot API.

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
import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

# OAuth device code flow constants (same client ID as opencode/Copilot CLI)
COPILOT_OAUTH_CLIENT_ID = "Ov23li8tweQw6odWQebz"
COPILOT_DEVICE_CODE_URL = "https://github.com/login/device/code"
COPILOT_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"

# Copilot API constants
COPILOT_TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
COPILOT_API_BASE_URL = "https://api.githubcopilot.com"
COPILOT_TOKEN_REFRESH_SKEW_SECONDS = 5 * 60
_COPILOT_ALLOWED_HOST = "githubcopilot.com"

# Token type prefixes
_CLASSIC_PAT_PREFIX = "ghp_"
_SUPPORTED_PREFIXES = ("gho_", "github_pat_", "ghu_")

# Env var search order (matches Copilot CLI)
COPILOT_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")

# Polling constants
_DEVICE_CODE_POLL_INTERVAL = 5  # seconds
_DEVICE_CODE_POLL_SAFETY_MARGIN = 3  # seconds


def is_classic_pat(token: str) -> bool:
    """Check if a token is a classic PAT (ghp_*), which Copilot doesn't support."""
    return token.strip().startswith(_CLASSIC_PAT_PREFIX)


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
    """Resolve a GitHub token suitable for Copilot API use.

    Returns (token, source) where source describes where the token came from.
    Raises ValueError if only a classic PAT is available.
    """
    # 1. Check env vars in priority order
    for env_var in COPILOT_ENV_VARS:
        val = os.getenv(env_var, "").strip()
        if val:
            valid, msg = validate_copilot_token(val)
            if not valid:
                logger.warning(
                    "Token from %s is not supported: %s", env_var, msg
                )
                continue
            return val, env_var

    # 2. Fall back to gh auth token
    token = _try_gh_cli_token()
    if token:
        valid, msg = validate_copilot_token(token)
        if not valid:
            raise ValueError(
                f"Token from `gh auth token` is a classic PAT (ghp_*). {msg}"
            )
        return token, "gh auth token"

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


def _copilot_token_cache_path() -> Path:
    """Return the on-disk cache path for exchanged Copilot API tokens."""
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    return home / "copilot_token.json"


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _parse_expires_at(value: Any) -> int:
    """Parse GitHub's expires_at field into epoch milliseconds."""
    if isinstance(value, (int, float)) and value > 0:
        numeric = int(value)
        return numeric if numeric > 10_000_000_000 else numeric * 1000
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        try:
            parsed = int(raw)
        except ValueError:
            try:
                parsed_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("Copilot token response missing expires_at") from exc
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
            return int(parsed_dt.timestamp() * 1000)
        return parsed if parsed > 10_000_000_000 else parsed * 1000
    raise ValueError("Copilot token response missing expires_at")


def _normalize_host_candidate(value: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return ""
    parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
    return (parsed.hostname or "").strip().lower()


def _is_allowed_copilot_host(hostname: str) -> bool:
    hostname = (hostname or "").strip().lower()
    return bool(hostname) and (
        hostname == _COPILOT_ALLOWED_HOST
        or hostname.endswith(f".{_COPILOT_ALLOWED_HOST}")
    )


def derive_copilot_api_base_url(token: str) -> str:
    """Best-effort routed Copilot API base URL from the exchanged token payload."""
    trimmed = (token or "").strip()
    if not trimmed:
        return COPILOT_API_BASE_URL

    match = re.search(r"(?:^|;)\s*proxy-ep=([^;\s]+)", trimmed, flags=re.IGNORECASE)
    proxy_endpoint = match.group(1).strip() if match else ""
    hostname = _normalize_host_candidate(proxy_endpoint)
    if not _is_allowed_copilot_host(hostname):
        return COPILOT_API_BASE_URL

    host = re.sub(r"^proxy\.", "api.", hostname, flags=re.IGNORECASE)
    return f"https://{host}" if _is_allowed_copilot_host(host) else COPILOT_API_BASE_URL


def is_copilot_base_url(base_url: Optional[str]) -> bool:
    """Return True for routed GitHub Copilot API hosts."""
    hostname = _normalize_host_candidate(str(base_url or ""))
    return _is_allowed_copilot_host(hostname)


def _load_cached_copilot_api_token(github_token: str) -> Optional[dict[str, Any]]:
    path = _copilot_token_cache_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    cached_token = str(payload.get("token") or "").strip()
    expires_at_ms = payload.get("expires_at_ms")
    fingerprint = str(payload.get("github_token_fingerprint") or "")
    if not cached_token or not isinstance(expires_at_ms, int) or not fingerprint:
        return None
    if fingerprint != _token_fingerprint(github_token):
        return None
    if expires_at_ms - int(time.time() * 1000) <= COPILOT_TOKEN_REFRESH_SKEW_SECONDS * 1000:
        return None

    return {
        "token": cached_token,
        "expires_at_ms": expires_at_ms,
        "base_url": str(payload.get("base_url") or "").strip() or derive_copilot_api_base_url(cached_token),
        "source": f"cache:{path}",
    }


def _save_cached_copilot_api_token(github_token: str, token: str, expires_at_ms: int, base_url: str) -> None:
    path = _copilot_token_cache_path()
    payload = {
        "github_token_fingerprint": _token_fingerprint(github_token),
        "token": token,
        "expires_at_ms": int(expires_at_ms),
        "base_url": base_url,
        "updated_at_ms": int(time.time() * 1000),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        try:
            os.chmod(tmp_path, 0o600)
        except OSError as exc:
            logger.warning("Could not set secure permissions on Copilot token cache %s: %s", tmp_path, exc)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def get_cached_copilot_api_token(github_token: str) -> Optional[dict[str, Any]]:
    """Return a valid cached exchanged Copilot token without making network calls."""
    github_token = (github_token or "").strip()
    if not github_token:
        return None
    return _load_cached_copilot_api_token(github_token)


def exchange_copilot_api_token(github_token: str, *, timeout: float = 15.0) -> dict[str, Any]:
    """Exchange a GitHub OAuth/PAT token for a routed Copilot API token."""
    import urllib.request

    github_token = (github_token or "").strip()
    if not github_token:
        return {
            "token": "",
            "base_url": COPILOT_API_BASE_URL,
            "expires_at_ms": 0,
            "source": "",
        }

    cached = _load_cached_copilot_api_token(github_token)
    if cached:
        return cached

    req = urllib.request.Request(
        COPILOT_TOKEN_EXCHANGE_URL,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {github_token}",
            "User-Agent": "HermesAgent/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode())

    token = str(payload.get("token") or "").strip()
    if not token:
        raise ValueError("Copilot token response missing token")

    expires_at_ms = _parse_expires_at(payload.get("expires_at"))
    endpoints = payload.get("endpoints") or {}
    base_url = str(endpoints.get("api") or "").strip()
    if not is_copilot_base_url(base_url):
        base_url = derive_copilot_api_base_url(token)
    try:
        _save_cached_copilot_api_token(github_token, token, expires_at_ms, base_url)
    except Exception:
        logger.debug("Failed to cache Copilot token", exc_info=True)
    return {
        "token": token,
        "base_url": base_url,
        "expires_at_ms": expires_at_ms,
        "source": COPILOT_TOKEN_EXCHANGE_URL,
    }


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
    data = urllib.parse.urlencode({
        "client_id": COPILOT_OAUTH_CLIENT_ID,
        "scope": "read:user",
    }).encode()

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

    verification_uri = device_data.get("verification_uri", "https://github.com/login/device")
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

        poll_data = urllib.parse.urlencode({
            "client_id": COPILOT_OAUTH_CLIENT_ID,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }).encode()

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
        "Openai-Intent": "conversation-edits",
        "x-initiator": "agent" if is_agent_turn else "user",
    }
    if is_vision:
        headers["Copilot-Vision-Request"] = "true"

    return headers

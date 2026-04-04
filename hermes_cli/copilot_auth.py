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
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# OAuth device code flow constants (same client ID as opencode/Copilot CLI)
COPILOT_OAUTH_CLIENT_ID = "Ov23li8tweQw6odWQebz"
COPILOT_DEVICE_CODE_URL = "https://github.com/login/device/code"
COPILOT_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"

# VSCode OAuth client ID (for copilot-vscode provider)
VSCODE_OAUTH_CLIENT_ID = "Iv1.b507a08c87ecfe98"

# Copilot API constants
COPILOT_TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
COPILOT_API_BASE_URL = "https://api.githubcopilot.com"

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


# ─── VSCode Copilot Token Exchange ────────────────────────────────────────

def exchange_github_token_for_copilot(github_token: str) -> dict:
    """Exchange a GitHub OAuth token for a short-lived Copilot token.

    This implements the VSCode Copilot authentication flow's second step:
    GitHub device code → GitHub token → Copilot token exchange.

    Args:
        github_token: A GitHub OAuth token (gho_*, github_pat_*, or ghu_*)

    Returns:
        dict with:
            - token: The Copilot API token (str)
            - expires_at: Unix timestamp when token expires (int)

    Raises:
        ValueError: If the token exchange fails or returns invalid data
    """
    import urllib.request
    import urllib.error

    if not github_token or not github_token.strip():
        raise ValueError("GitHub token is required for Copilot token exchange")

    req = urllib.request.Request(
        COPILOT_TOKEN_EXCHANGE_URL,
        headers={
            "Authorization": f"token {github_token.strip()}",
            "User-Agent": "GithubCopilot/1.155.0",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        try:
            error_body = exc.read().decode()
        except Exception:
            error_body = ""

        if status_code == 401:
            raise ValueError(
                "GitHub token is invalid or lacks Copilot access. "
                "Ensure you have an active GitHub Copilot subscription and "
                "run `hermes login --provider copilot-vscode` to re-authenticate."
            )
        elif status_code == 403:
            raise ValueError(
                "GitHub token was rejected (403 Forbidden). This may indicate:\n"
                "  → No active Copilot subscription\n"
                "  → Token lacks required permissions\n"
                f"Response: {error_body[:200]}"
            )
        else:
            raise ValueError(
                f"Copilot token exchange failed with HTTP {status_code}.\n"
                f"Response: {error_body[:200]}"
            )
    except urllib.error.URLError as exc:
        raise ValueError(f"Network error during Copilot token exchange: {exc}")
    except Exception as exc:
        raise ValueError(f"Unexpected error during Copilot token exchange: {exc}")

    # Validate response structure
    copilot_token = data.get("token")
    expires_at = data.get("expires_at")

    if not isinstance(copilot_token, str) or not copilot_token.strip():
        raise ValueError(
            "Copilot token exchange response missing 'token' field. "
            f"Response: {json.dumps(data)[:200]}"
        )

    if not isinstance(expires_at, (int, float)) or expires_at <= 0:
        raise ValueError(
            "Copilot token exchange response missing valid 'expires_at' field. "
            f"Response: {json.dumps(data)[:200]}"
        )

    return {
        "token": copilot_token.strip(),
        "expires_at": int(expires_at),
    }


def get_copilot_token_with_refresh(provider_state: dict) -> str:
    """Get a valid Copilot token, refreshing if needed.

    Implements automatic token refresh with a 5-minute expiry buffer to avoid
    mid-request token expiration (matches VSCode behavior).

    Args:
        provider_state: Dict containing:
            - github_token: GitHub OAuth token for refresh (required)
            - copilot_token: Cached Copilot token (optional)
            - copilot_token_expires_at: Expiry timestamp in seconds (optional)

    Returns:
        A valid Copilot API token (updates provider_state in-place if refreshed)

    Raises:
        ValueError: If github_token is missing or token exchange fails
    """
    if not isinstance(provider_state, dict):
        raise ValueError("provider_state must be a dict")

    github_token = provider_state.get("github_token", "").strip()
    if not github_token:
        raise ValueError(
            "GitHub token is required for Copilot token refresh. "
            "Run `hermes login --provider copilot-vscode` to authenticate."
        )

    # Check if cached token is still valid (5-minute buffer)
    copilot_token = provider_state.get("copilot_token", "").strip()
    expires_at = provider_state.get("copilot_token_expires_at")

    if copilot_token and isinstance(expires_at, (int, float)):
        # Token is valid if it won't expire within the next 5 minutes
        buffer_seconds = 300  # 5 minutes
        if float(expires_at) > (time.time() + buffer_seconds):
            logger.debug(
                "Copilot token still valid (expires in %.1f minutes)",
                (float(expires_at) - time.time()) / 60,
            )
            return copilot_token

    # Token missing, expired, or within 5-min buffer - refresh it
    logger.info("Refreshing Copilot token...")
    copilot_data = exchange_github_token_for_copilot(github_token)

    # Update provider state in-place
    provider_state["copilot_token"] = copilot_data["token"]
    provider_state["copilot_token_expires_at"] = copilot_data["expires_at"]

    expires_in_minutes = (copilot_data["expires_at"] - time.time()) / 60
    logger.info(
        "Copilot token refreshed successfully (expires in %.1f minutes)",
        expires_in_minutes,
    )

    return copilot_data["token"]


def copilot_vscode_device_code_login(
    *,
    host: str = "github.com",
    timeout_seconds: float = 300,
) -> Optional[dict]:
    """Run the VSCode OAuth device code flow for GitHub Copilot.

    This implements the two-step VSCode authentication flow:
      1. GitHub device code → GitHub OAuth token (using VSCode's client ID)
      2. GitHub token → Copilot token exchange

    Prints instructions for the user, polls for completion, and returns
    both the GitHub token and Copilot token on success.

    Args:
        host: GitHub host (default: github.com)
        timeout_seconds: How long to wait for user authorization

    Returns:
        dict with keys:
            - github_token: The GitHub OAuth token (for refresh)
            - copilot_token: The Copilot API token
            - copilot_token_expires_at: Unix timestamp when Copilot token expires
        Returns None on failure/cancellation.

    This replicates the flow used by VSCode's Copilot extension.
    """
    import urllib.request
    import urllib.parse

    domain = host.rstrip("/")
    device_code_url = f"https://{domain}/login/device/code"
    access_token_url = f"https://{domain}/login/oauth/access_token"

    # Step 1: Request device code (using VSCode's client ID)
    data = urllib.parse.urlencode({
        "client_id": VSCODE_OAUTH_CLIENT_ID,
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
        logger.error("Failed to initiate VSCode device authorization: %s", exc)
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

    # Step 3: Poll for GitHub OAuth token
    deadline = time.time() + timeout_seconds
    github_token = None

    while time.time() < deadline:
        time.sleep(interval + _DEVICE_CODE_POLL_SAFETY_MARGIN)

        poll_data = urllib.parse.urlencode({
            "client_id": VSCODE_OAUTH_CLIENT_ID,
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
            github_token = result["access_token"]
            print(" ✓")
            break

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

    if not github_token:
        print()
        print("  ✗ Timed out waiting for authorization.")
        return None

    # Step 4: Exchange GitHub token for Copilot token
    print("  Exchanging for Copilot token...", end="", flush=True)
    try:
        copilot_data = exchange_github_token_for_copilot(github_token)
        print(" ✓")
    except ValueError as exc:
        print()
        print(f"  ✗ Copilot token exchange failed: {exc}")
        return None
    except Exception as exc:
        print()
        print(f"  ✗ Unexpected error during token exchange: {exc}")
        logger.exception("Token exchange failed")
        return None

    # Return both tokens
    return {
        "github_token": github_token,
        "copilot_token": copilot_data["token"],
        "copilot_token_expires_at": copilot_data["expires_at"],
    }


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

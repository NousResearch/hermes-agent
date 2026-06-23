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
import sys
import time
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

# Polling constants
_DEVICE_CODE_POLL_INTERVAL = 5  # seconds
_DEVICE_CODE_POLL_SAFETY_MARGIN = 3  # seconds

# ─────────────────────────────────────────────────────────────────────────────
# Single Copilot CLI identity.
#
# Every Copilot-facing request (inference, token exchange, device-OAuth flow)
# presents ONE identity: the `copilot-developer-cli` integration with the real
# `@github/copilot` CLI User-Agent. Previously several call sites each sent a
# different ad-hoc identity (`GitHubCopilotChat/*`, `HermesAgent/1.0`,
# `vscode-chat` + `Editor-*` VS Code Chat headers), which is inconsistent and
# does not match what a CLI agent actually is.
#
# Copilot-Integration-Id is the lever the GitHub backend keys the visible model
# catalog and per-model limits off of (not the User-Agent). A read-only
# `GET /models` sweep across candidate integration ids found that
# `copilot-developer-cli` returns a strict superset of the catalog (it is the
# only id that exposes the newest models with the full reasoning-effort range),
# so it is used uniformly. Override via HERMES_COPILOT_INTEGRATION_ID for an
# account that needs a different integrator.
#
# The integration-id only unlocks the catalog when the request carries a valid
# GitHub Bearer token (resolved by resolve_copilot_token()); without it the
# premium models are not served regardless of the integration-id.
_COPILOT_INTEGRATION_ID_DEFAULT = "copilot-developer-cli"

# Latest @github/copilot CLI version, used for the User-Agent we present. The
# real CLI reports `copilot/<ver> (<platform> <node>) term/<TERM_PROGRAM>`; we
# reproduce that shape from real host values so it is authentic rather than
# fabricated, degrading to the short `copilot/<ver>` form when node is absent.
# Env-overridable for pinning. Fallback is the value shipped at time of writing.
_COPILOT_CLI_VERSION_FALLBACK = "1.0.63"

_copilot_node_version_memo: Optional[str] = None


def _copilot_integration_id() -> str:
    """Return the Copilot-Integration-Id to send (env-overridable)."""
    override = os.getenv("HERMES_COPILOT_INTEGRATION_ID", "").strip()
    return override or _COPILOT_INTEGRATION_ID_DEFAULT


def _copilot_cli_version() -> str:
    """Return the @github/copilot CLI version string for the User-Agent."""
    return os.getenv("HERMES_COPILOT_CLI_VERSION", "").strip() or _COPILOT_CLI_VERSION_FALLBACK


def _copilot_node_version() -> str:
    """Return a ``v``-prefixed Node version for the CLI User-Agent.

    The real ``@github/copilot`` CLI runs on Node and reports
    ``process.version`` in the parenthetical UA segment. We resolve a real node
    version from the host (``node --version``) so the value is authentic; if no
    node is on PATH we return an empty string and the caller falls back to the
    short UA form. Resolution order: ``HERMES_COPILOT_NODE_VERSION`` env
    override, then ``node --version`` (cached in-process), then empty.
    """
    override = os.getenv("HERMES_COPILOT_NODE_VERSION", "").strip()
    if override:
        return override if override.startswith("v") else f"v{override}"

    global _copilot_node_version_memo
    if _copilot_node_version_memo is not None:
        return _copilot_node_version_memo

    ver = ""
    node_path = shutil.which("node")
    if node_path:
        try:
            out = subprocess.run(
                [node_path, "--version"],
                capture_output=True, text=True, timeout=2,
            )
            cand = (out.stdout or "").strip()
            if cand.startswith("v"):
                ver = cand
        except Exception as exc:  # pragma: no cover - host-dependent
            logger.debug("node --version probe failed: %s", exc)

    _copilot_node_version_memo = ver
    return ver


def _copilot_user_agent() -> str:
    """Build the @github/copilot CLI User-Agent presented on every request.

    Reproduces the real CLI builder ``copilot/<ver> (<platform> <node>)
    term/<TERM_PROGRAM>``. ``TERM_PROGRAM`` is read from the environment, else a
    valid ``vscode`` default (never the literal ``unknown`` the raw CLI emits
    when it cannot resolve a terminal). When no node runtime is available the
    value degrades to the short ``copilot/<ver>`` form.
    """
    version = _copilot_cli_version()
    node = _copilot_node_version()
    if not node:
        return f"copilot/{version}"
    platform = "linux"
    if sys.platform == "darwin":
        platform = "darwin"
    elif sys.platform.startswith("win"):
        platform = "win32"
    term = os.getenv("HERMES_COPILOT_TERM_PROGRAM", "").strip() or os.getenv("TERM_PROGRAM", "").strip() or "vscode"
    return f"copilot/{version} ({platform} {node}) term/{term}"


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
    """Return a token from ``gh auth token`` when the GitHub CLI is available.

    When COPILOT_GH_HOST is set, passes ``--hostname`` so gh returns the
    correct host's token.  Also strips GITHUB_TOKEN / GH_TOKEN from the
    subprocess environment so ``gh`` reads from its own credential store
    (hosts.yml) instead of just echoing the env var back.
    """
    hostname = os.getenv("COPILOT_GH_HOST", "").strip()

    # Build a clean env so gh doesn't short-circuit on GITHUB_TOKEN / GH_TOKEN
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in {"GITHUB_TOKEN", "GH_TOKEN"}}

    for gh_path in _gh_cli_candidates():
        cmd = [gh_path, "auth", "token"]
        if hostname:
            cmd += ["--hostname", hostname]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                env=clean_env,
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
            "User-Agent": _copilot_user_agent(),
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
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
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
                "User-Agent": _copilot_user_agent(),
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


# ─── Copilot Token Exchange ────────────────────────────────────────────────

# Module-level cache for exchanged Copilot API tokens.
# Maps raw_token_fingerprint -> (api_token, expires_at_epoch).
_jwt_cache: dict[str, tuple[str, float]] = {}
_JWT_REFRESH_MARGIN_SECONDS = 120  # refresh 2 min before expiry

# Token exchange endpoint. We present our single Copilot CLI identity (the
# `copilot-developer-cli` integration + `_copilot_user_agent()`), the same one
# used on the inference path, so there is exactly one identity across every
# Copilot-facing request.
_TOKEN_EXCHANGE_URL="https:...oken"


def _token_fingerprint(raw_token: str) -> str:
    """Short fingerprint of a raw token for cache keying (avoids storing full token)."""
    import hashlib
    return hashlib.sha256(raw_token.encode()).hexdigest()[:16]


def exchange_copilot_token(raw_token: str, *, timeout: float = 10.0) -> tuple[str, float]:
    """Exchange a raw GitHub token for a short-lived Copilot API token.

    Calls ``GET https://api.github.com/copilot_internal/v2/token`` with
    the raw GitHub token and returns ``(api_token, expires_at)``.

    The returned token is a semicolon-separated string (not a standard JWT)
    used as ``Authorization: Bearer <token>`` for Copilot API requests.

    Results are cached in-process and reused until close to expiry.
    Raises ``ValueError`` on failure.
    """
    import urllib.request

    fp = _token_fingerprint(raw_token)

    # Check cache first
    cached = _jwt_cache.get(fp)
    if cached:
        api_token, expires_at = cached
        if time.time() < expires_at - _JWT_REFRESH_MARGIN_SECONDS:
            return api_token, expires_at

    req = urllib.request.Request(
        _TOKEN_EXCHANGE_URL,
        method="GET",
        headers={
            "Authorization": f"token {raw_token}",
            "User-Agent": _copilot_user_agent(),
            "Accept": "application/json",
            "Copilot-Integration-Id": _copilot_integration_id(),
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        raise ValueError(f"Copilot token exchange failed: {exc}") from exc

    api_token = data.get("token", "")
    expires_at = data.get("expires_at", 0)
    if not api_token:
        raise ValueError("Copilot token exchange returned empty token")

    # Convert expires_at to float if needed
    expires_at = float(expires_at) if expires_at else time.time() + 1800

    _jwt_cache[fp] = (api_token, expires_at)
    logger.debug(
        "Copilot token exchanged, expires_at=%s",
        expires_at,
    )
    return api_token, expires_at


def get_copilot_api_token(raw_token: str) -> str:
    """Exchange a raw GitHub token for a Copilot API token, with fallback.

    Convenience wrapper: returns the exchanged token on success, or the
    raw token unchanged if the exchange fails (e.g. network error, unsupported
    account type). This preserves existing behaviour for accounts that don't
    need exchange while enabling access to internal-only models for those that do.
    """
    if not raw_token:
        return raw_token
    try:
        api_token, _ = exchange_copilot_token(raw_token)
        return api_token
    except Exception as exc:
        logger.debug("Copilot token exchange failed, using raw token: %s", exc)
        return raw_token


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
        "User-Agent": _copilot_user_agent(),
        "Copilot-Integration-Id": _copilot_integration_id(),
        "Openai-Intent": "conversation-edits",
        "x-initiator": "agent" if is_agent_turn else "user",
    }
    if is_vision:
        headers["Copilot-Vision-Request"] = "true"

    return headers

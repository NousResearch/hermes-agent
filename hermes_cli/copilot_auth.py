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

from hermes_cli._subprocess_compat import IS_WINDOWS, windows_hide_flags

logger = logging.getLogger(__name__)

# OAuth device code flow constants — VS Code's GitHub App client ID.
# The previous opencode OAuth App ID (Ov23li8tweQw6odWQebz) produces gho_*
# tokens that cannot be exchanged for Copilot API JWTs (404 on
# /copilot_internal/v2/token). VS Code's App ID produces ghu_* tokens
# that support exchange, which is required to access internal-only models
# (e.g. claude-opus-4.6-1m) and enterprise endpoints.
# Tested on Individual and Enterprise accounts.
COPILOT_OAUTH_CLIENT_ID = "Iv1.b507a08c87ecfe98"
# Token type prefixes
_CLASSIC_PAT_PREFIX = "ghp_"
_SUPPORTED_PREFIXES = ("gho_", "github_pat_", "ghu_")

# Env var search order (matches Copilot CLI)
COPILOT_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")

# Polling constants
_DEVICE_CODE_POLL_INTERVAL = 5  # seconds
_DEVICE_CODE_POLL_SAFETY_MARGIN = 3  # seconds


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

    _popen_kwargs = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {}
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
                **_popen_kwargs,
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


# ─── Copilot Token Exchange ────────────────────────────────────────────────

# Module-level cache for exchanged Copilot API tokens.
# Maps raw_token_fingerprint -> (api_token, expires_at_epoch, base_url).
_jwt_cache: dict[str, tuple[str, float, Optional[str]]] = {}
_JWT_REFRESH_MARGIN_SECONDS = 120  # refresh 2 min before expiry

# Token exchange endpoint and headers (matching VS Code / Copilot CLI)
_TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
_EDITOR_VERSION = "vscode/1.104.1"
_EXCHANGE_USER_AGENT = "GitHubCopilotChat/0.26.7"

# Transient-failure hardening for the token exchange. Gateway startup often
# races network readiness (launchd relaunch, DHCP/VPN settling); a single-shot
# exchange that fails there silently degrades to the RAW GitHub token, which the
# Copilot server routes to the "copilot-language-server" integrator whose model
# allowlist omits enterprise-only models (e.g. claude-opus-4.8) → HTTP 400 on
# every turn until the next restart. Retry a few times, and persist the last
# good exchanged JWT to disk so a restart during a blip reuses the still-valid
# ~30-min token instead of degrading.
_EXCHANGE_MAX_ATTEMPTS = 3
_EXCHANGE_BACKOFF_BASE_SECONDS = 1.5  # sleeps ~1.5s, ~3.0s between attempts
_JWT_DISK_FILENAME = ".copilot_jwt.json"
_JWT_DISK_MAX_BYTES = 1_048_576  # 1 MiB cap on the persisted JWT store read


def _token_fingerprint(raw_token: str) -> str:
    """Short fingerprint of a raw token for cache keying (avoids storing full token)."""
    import hashlib
    return hashlib.sha256(raw_token.encode()).hexdigest()[:16]


def evict_cached_exchanged_token(raw_token: str) -> None:
    """Drop any cached exchanged JWT for ``raw_token`` (in-process + on-disk).

    Used by the runtime stale-credential recovery path: when a live request
    starts failing with a Copilot ``model_not_available_for_integrator`` /
    ``model_not_supported`` 400, the cached exchanged token (or a degraded raw
    fallback that was cached in its place) is stale. Evicting both cache tiers
    forces the next ``exchange_copilot_token`` call to hit the network and mint
    a fresh token instead of returning the poisoned cache entry.
    """
    if not raw_token:
        return
    fp = _token_fingerprint(raw_token)
    _jwt_cache.pop(fp, None)
    path = _jwt_disk_path()
    if not path or not path.exists():
        return
    try:
        store = json.loads(path.read_text())
        if isinstance(store, dict) and fp in store:
            del store[fp]
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(store))
            try:
                os.chmod(tmp, 0o600)
            except Exception:
                pass
            os.replace(tmp, path)
    except Exception as exc:
        logger.debug("Failed to evict cached Copilot JWT: %s", exc)


def _jwt_disk_path() -> Optional[Path]:
    """Path to the on-disk exchanged-JWT cache (profile-aware), or None."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()) / _JWT_DISK_FILENAME
    except Exception:
        return None


def _load_jwt_from_disk(fp: str) -> Optional[tuple[str, float, Optional[str]]]:
    """Load a persisted exchanged JWT for ``fp`` → (api_token, expires_at, base_url)."""
    path = _jwt_disk_path()
    if not path or not path.exists():
        return None
    try:
        # Bound the read: this file is a small JSON map of fingerprint → token.
        # A well-formed store is a few KB; cap at 1 MiB so a corrupt/oversized
        # file can't balloon memory (mirrors the auth JSON read bound). A file
        # over the cap is treated as unusable — the caller re-exchanges.
        if path.stat().st_size > _JWT_DISK_MAX_BYTES:
            logger.debug("Persisted Copilot JWT store exceeds %d bytes; ignoring", _JWT_DISK_MAX_BYTES)
            return None
        store = json.loads(path.read_text())
        entry = store.get(fp) if isinstance(store, dict) else None
        if not isinstance(entry, dict):
            return None
        api_token = entry.get("api_token", "")
        expires_at = float(entry.get("expires_at", 0) or 0)
        base_url = entry.get("base_url")
        if api_token and expires_at:
            return api_token, expires_at, base_url
    except Exception as exc:
        logger.debug("Failed to load persisted Copilot JWT: %s", exc)
    return None


def _save_jwt_to_disk(
    fp: str, api_token: str, expires_at: float, base_url: Optional[str]
) -> None:
    """Persist an exchanged JWT (0o600), pruning expired entries."""
    path = _jwt_disk_path()
    if not path:
        return
    try:
        store: dict = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text())
                if isinstance(loaded, dict):
                    store = loaded
            except Exception:
                store = {}
        now = time.time()
        store = {
            k: v
            for k, v in store.items()
            if isinstance(v, dict) and float(v.get("expires_at", 0) or 0) > now
        }
        store[fp] = {
            "api_token": api_token,
            "expires_at": expires_at,
            "base_url": base_url,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(store))
        try:
            os.chmod(tmp, 0o600)
        except Exception:
            pass
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
    except Exception as exc:
        logger.debug("Failed to persist Copilot JWT: %s", exc)


def exchange_copilot_token(raw_token: str, *, timeout: float = 10.0) -> tuple[str, float, Optional[str]]:
    """Exchange a raw GitHub token for a short-lived Copilot API token.

    Calls ``GET https://api.github.com/copilot_internal/v2/token`` with
    the raw GitHub token and returns ``(api_token, expires_at, base_url)``.

    The returned token is a semicolon-separated string (not a standard JWT)
    used as ``Authorization: Bearer <token>`` for Copilot API requests.
    ``base_url`` is the account-specific API host: the authoritative
    ``endpoints.api`` advertised by the exchange (enterprise/proxied
    accounts), falling back to a host derived from the token's ``proxy-ep``
    field. Individual accounts have neither, so ``base_url`` is None.

    Results are cached in-process and reused until close to expiry.
    Raises ``ValueError`` on failure.
    """
    import urllib.request

    fp = _token_fingerprint(raw_token)

    # Check in-process cache first
    cached = _jwt_cache.get(fp)
    if cached:
        api_token, expires_at, base_url = cached
        if time.time() < expires_at - _JWT_REFRESH_MARGIN_SECONDS:
            return api_token, expires_at, base_url

    # Then the on-disk cache: a fresh process (e.g. gateway restart) has an
    # empty in-process cache but may have a still-valid persisted JWT. Reusing
    # it avoids a network round-trip at startup — precisely when the network is
    # most likely to be flaky and the single-shot exchange would degrade to the
    # raw token.
    disk_cached = _load_jwt_from_disk(fp)
    if disk_cached:
        api_token, expires_at, base_url = disk_cached
        if time.time() < expires_at - _JWT_REFRESH_MARGIN_SECONDS:
            _jwt_cache[fp] = (api_token, expires_at, base_url)
            return api_token, expires_at, base_url

    req = urllib.request.Request(
        _TOKEN_EXCHANGE_URL,
        method="GET",
        headers={
            "Authorization": f"token {raw_token}",
            "User-Agent": _EXCHANGE_USER_AGENT,
            "Accept": "application/json",
            "Editor-Version": _EDITOR_VERSION,
        },
    )

    # Retry with backoff. Startup network races (launchd relaunch, VPN/DHCP
    # settling) make the first attempt flaky; without this the sole failure
    # silently degrades to the raw token for the whole process lifetime.
    data = None
    last_exc: Optional[Exception] = None
    for attempt in range(_EXCHANGE_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            break
        except Exception as exc:  # noqa: BLE001 — retry all, re-raise below
            last_exc = exc
            if attempt < _EXCHANGE_MAX_ATTEMPTS - 1:
                sleep_s = _EXCHANGE_BACKOFF_BASE_SECONDS * (attempt + 1)
                logger.debug(
                    "Copilot token exchange attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt + 1, _EXCHANGE_MAX_ATTEMPTS, exc, sleep_s,
                )
                time.sleep(sleep_s)
    if data is None:
        raise ValueError(
            f"Copilot token exchange failed after {_EXCHANGE_MAX_ATTEMPTS} attempts: {last_exc}"
        ) from last_exc

    api_token = data.get("token", "")
    expires_at = data.get("expires_at", 0)
    if not api_token:
        raise ValueError("Copilot token exchange returned empty token")

    # Convert expires_at to float if needed
    expires_at = float(expires_at) if expires_at else time.time() + 1800

    # Resolve the account-specific API base URL. GitHub advertises the
    # authoritative endpoint under ``endpoints.api`` in the exchange response
    # (it differs for Copilot Enterprise / proxied accounts). When the
    # response omits it, fall back to deriving the host from the ``proxy-ep``
    # field embedded in the exchanged token. Individual accounts have neither,
    # so ``base_url`` stays None and callers use the registry default.
    base_url: Optional[str] = None
    endpoints = data.get("endpoints")
    if isinstance(endpoints, dict):
        api_endpoint = str(endpoints.get("api") or "").strip().rstrip("/")
        if api_endpoint:
            base_url = api_endpoint
    if not base_url:
        base_url = _derive_base_url_from_proxy_ep(api_token)

    _jwt_cache[fp] = (api_token, expires_at, base_url)
    _save_jwt_to_disk(fp, api_token, expires_at, base_url)
    logger.debug(
        "Copilot token exchanged, expires_at=%s, base_url=%s",
        expires_at,
        base_url,
    )
    return api_token, expires_at, base_url


def _derive_base_url_from_proxy_ep(token: str) -> Optional[str]:
    """Derive the Copilot API base URL from a proxy-ep field in the token.

    The exchanged Copilot token is a semicolon-separated string like
    ``tid=xxx;exp=xxx;proxy-ep=proxy.enterprise.githubcopilot.com;...``.
    This extracts ``proxy-ep`` and converts it to an API base URL by
    replacing the leading ``proxy.`` with ``api.``.

    Returns ``https://{api_hostname}`` or None if proxy-ep is absent.
    """
    import re
    m = re.search(r'(?:^|;)\s*proxy-ep=([^;\s]+)', token)
    if not m:
        return None

    proxy_ep = m.group(1)
    # Strip scheme if present
    for prefix in ("https://", "http://"):
        if proxy_ep.startswith(prefix):
            proxy_ep = proxy_ep[len(prefix):]
            break
    proxy_ep = proxy_ep.rstrip("/")

    # Replace leading "proxy." with "api."
    if proxy_ep.startswith("proxy."):
        api_host = "api." + proxy_ep[len("proxy."):]
    else:
        api_host = proxy_ep

    return f"https://{api_host}"


def get_copilot_api_token(raw_token: str) -> tuple[str, Optional[str]]:
    """Exchange a raw GitHub token for a Copilot API token, with fallback.

    Convenience wrapper: returns ``(api_token, base_url)`` on success, or
    ``(raw_token, None)`` if the exchange fails (e.g. network error, unsupported
    account type). This preserves existing behaviour for accounts that don't
    need exchange while enabling access to internal-only models for those that do.

    ``base_url`` is the account-specific API endpoint advertised by the
    exchange (``endpoints.api``, with a ``proxy-ep`` fallback), or None for
    individual accounts.
    """
    if not raw_token:
        return raw_token, None
    try:
        api_token, _, base_url = exchange_copilot_token(raw_token)
        return api_token, base_url
    except Exception as exc:
        logger.debug("Copilot token exchange failed, using raw token: %s", exc)
        return raw_token, None


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

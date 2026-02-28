"""
Multi-provider authentication system for Hermes Agent.

Supports OAuth device code flows (Nous Portal, future: OpenAI Codex) and
traditional API key providers (OpenRouter, custom endpoints). Auth state
is persisted in ~/.hermes/auth.json with cross-process file locking.

Architecture:
- ProviderConfig registry defines known OAuth providers
- Auth store (auth.json) holds per-provider credential state
- resolve_provider() picks the active provider via priority chain
- resolve_*_runtime_credentials() handles token refresh and key minting
- login_command() / logout_command() are the CLI entry points
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import base64
import subprocess
import time
import webbrowser
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

from hermes_cli.config import get_hermes_home, get_config_path
from hermes_constants import OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)

try:
    import fcntl
except Exception:
    fcntl = None

# =============================================================================
# Constants
# =============================================================================

AUTH_STORE_VERSION = 1
AUTH_LOCK_TIMEOUT_SECONDS = 15.0

# Nous Portal defaults
DEFAULT_NOUS_PORTAL_URL = "https://portal.nousresearch.com"
DEFAULT_NOUS_INFERENCE_URL = "https://inference-api.nousresearch.com/v1"
DEFAULT_NOUS_CLIENT_ID = "hermes-cli"
DEFAULT_NOUS_SCOPE = "inference:mint_agent_key"
DEFAULT_AGENT_KEY_MIN_TTL_SECONDS = 30 * 60  # 30 minutes
ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120       # refresh 2 min before expiry
DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS = 1     # poll at most every 1s
DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120


# =============================================================================
# Provider Registry
# =============================================================================

@dataclass
class ProviderConfig:
    """Describes a known OAuth provider."""
    id: str
    name: str
    auth_type: str  # "oauth_device_code" or "api_key"
    portal_base_url: str = ""
    inference_base_url: str = ""
    client_id: str = ""
    scope: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    "nous": ProviderConfig(
        id="nous",
        name="Nous Portal",
        auth_type="oauth_device_code",
        portal_base_url=DEFAULT_NOUS_PORTAL_URL,
        inference_base_url=DEFAULT_NOUS_INFERENCE_URL,
        client_id=DEFAULT_NOUS_CLIENT_ID,
        scope=DEFAULT_NOUS_SCOPE,
    ),
    "openai-codex": ProviderConfig(
        id="openai-codex",
        name="OpenAI Codex",
        auth_type="oauth_external",
        inference_base_url=DEFAULT_CODEX_BASE_URL,
    ),
}


# =============================================================================
# Error Types
# =============================================================================

class AuthError(RuntimeError):
    """Structured auth error with UX mapping hints."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        code: Optional[str] = None,
        relogin_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.relogin_required = relogin_required


def format_auth_error(error: Exception) -> str:
    """Map auth failures to concise user-facing guidance."""
    if not isinstance(error, AuthError):
        return str(error)

    if error.relogin_required:
        return f"{error} Run `hermes login` to re-authenticate."

    if error.code == "subscription_required":
        return (
            "No active paid subscription found on Nous Portal. "
            "Please purchase/activate a subscription, then retry."
        )

    if error.code == "insufficient_credits":
        return (
            "Subscription credits are exhausted. "
            "Top up/renew credits in Nous Portal, then retry."
        )

    if error.code == "temporarily_unavailable":
        return f"{error} Please retry in a few seconds."

    return str(error)


# =============================================================================
# Auth Store — persistence layer for ~/.hermes/auth.json
# =============================================================================

def _auth_file_path() -> Path:
    return get_hermes_home() / "auth.json"


def _auth_lock_path() -> Path:
    return _auth_file_path().with_suffix(".lock")


@contextmanager
def _auth_store_lock(timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS):
    """Cross-process advisory lock for auth.json reads+writes."""
    lock_path = _auth_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+") as lock_file:
        if fcntl is None:
            yield
            return

        deadline = time.time() + max(1.0, timeout_seconds)
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= deadline:
                    raise TimeoutError("Timed out waiting for auth store lock")
                time.sleep(0.05)

        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_auth_store(auth_file: Optional[Path] = None) -> Dict[str, Any]:
    auth_file = auth_file or _auth_file_path()
    if not auth_file.exists():
        return {"version": AUTH_STORE_VERSION, "providers": {}}

    try:
        raw = json.loads(auth_file.read_text())
    except Exception:
        return {"version": AUTH_STORE_VERSION, "providers": {}}

    if isinstance(raw, dict) and isinstance(raw.get("providers"), dict):
        return raw

    # Migrate from PR's "systems" format if present
    if isinstance(raw, dict) and isinstance(raw.get("systems"), dict):
        systems = raw["systems"]
        providers = {}
        if "nous_portal" in systems:
            providers["nous"] = systems["nous_portal"]
        return {"version": AUTH_STORE_VERSION, "providers": providers,
                "active_provider": "nous" if providers else None}

    return {"version": AUTH_STORE_VERSION, "providers": {}}


def _save_auth_store(auth_store: Dict[str, Any]) -> Path:
    auth_file = _auth_file_path()
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    auth_store["version"] = AUTH_STORE_VERSION
    auth_store["updated_at"] = datetime.now(timezone.utc).isoformat()
    auth_file.write_text(json.dumps(auth_store, indent=2) + "\n")
    # Restrict file permissions to owner only
    try:
        auth_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return auth_file


def _load_provider_state(auth_store: Dict[str, Any], provider_id: str) -> Optional[Dict[str, Any]]:
    providers = auth_store.get("providers")
    if not isinstance(providers, dict):
        return None
    state = providers.get(provider_id)
    return dict(state) if isinstance(state, dict) else None


def _save_provider_state(auth_store: Dict[str, Any], provider_id: str, state: Dict[str, Any]) -> None:
    providers = auth_store.setdefault("providers", {})
    if not isinstance(providers, dict):
        auth_store["providers"] = {}
        providers = auth_store["providers"]
    providers[provider_id] = state
    auth_store["active_provider"] = provider_id


def get_provider_auth_state(provider_id: str) -> Optional[Dict[str, Any]]:
    """Return persisted auth state for a provider, or None."""
    auth_store = _load_auth_store()
    return _load_provider_state(auth_store, provider_id)


def get_active_provider() -> Optional[str]:
    """Return the currently active provider ID from auth store."""
    auth_store = _load_auth_store()
    return auth_store.get("active_provider")


def clear_provider_auth(provider_id: Optional[str] = None) -> bool:
    """
    Clear auth state for a provider. Used by `hermes logout`.
    If provider_id is None, clears the active provider.
    Returns True if something was cleared.
    """
    with _auth_store_lock():
        auth_store = _load_auth_store()
        target = provider_id or auth_store.get("active_provider")
        if not target:
            return False

        providers = auth_store.get("providers", {})
        if target not in providers:
            return False

        del providers[target]
        if auth_store.get("active_provider") == target:
            auth_store["active_provider"] = None
        _save_auth_store(auth_store)
    return True


def deactivate_provider() -> None:
    """
    Clear active_provider in auth.json without deleting credentials.
    Used when the user switches to a non-OAuth provider (OpenRouter, custom)
    so auto-resolution doesn't keep picking the OAuth provider.
    """
    with _auth_store_lock():
        auth_store = _load_auth_store()
        auth_store["active_provider"] = None
        _save_auth_store(auth_store)


# =============================================================================
# Provider Resolution — picks which provider to use
# =============================================================================

def resolve_provider(
    requested: Optional[str] = None,
    *,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> str:
    """
    Determine which inference provider to use.

    Priority (when requested="auto" or None):
    1. active_provider in auth.json with valid credentials
    2. Explicit CLI api_key/base_url -> "openrouter"
    3. OPENAI_API_KEY or OPENROUTER_API_KEY env vars -> "openrouter"
    4. Fallback: "openrouter"
    """
    normalized = (requested or "auto").strip().lower()

    if normalized in {"openrouter", "custom"}:
        return "openrouter"
    if normalized in PROVIDER_REGISTRY:
        return normalized
    if normalized == "chutes":
        return "chutes"
    if normalized == "nvidia":
<<<<<<< HEAD
        return "nvidia" 
=======
        return "nvidia"
>>>>>>> fee6b78 (feat: add NVIDIA NIM as a first-class inference provider)
    if normalized != "auto":
        raise AuthError(
            f"Unknown provider '{normalized}'.",
            code="invalid_provider",
        )

    # Explicit one-off CLI creds always mean openrouter/custom
    if explicit_api_key or explicit_base_url:
        return "openrouter"

    # Check auth store for an active OAuth provider
    try:
        auth_store = _load_auth_store()
        active = auth_store.get("active_provider")
        if active and active in PROVIDER_REGISTRY:
            status = get_auth_status(active)
            if status.get("logged_in"):
                return active
    except Exception as e:
        logger.debug("Could not detect active auth provider: %s", e)

    if os.getenv("CHUTES_API_KEY"):
        return "chutes"
    if os.getenv("NVIDIA_API_KEY"):
        return "nvidia"
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"

    return "openrouter"


# =============================================================================
# Timestamp / TTL helpers
# =============================================================================

def _parse_iso_timestamp(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _is_expiring(expires_at_iso: Any, skew_seconds: int) -> bool:
    expires_epoch = _parse_iso_timestamp(expires_at_iso)
    if expires_epoch is None:
        return True
    return expires_epoch <= (time.time() + skew_seconds)


def _coerce_ttl_seconds(expires_in: Any) -> int:
    try:
        ttl = int(expires_in)
    except Exception:
        ttl = 0
    return max(0, ttl)


def _optional_base_url(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().rstrip("/")
    return cleaned if cleaned else None


def _decode_jwt_claims(token: Any) -> Dict[str, Any]:
    if not isinstance(token, str) or token.count(".") != 2:
        return {}
    payload = token.split(".")[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode("utf-8"))
        claims = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def _codex_access_token_is_expiring(access_token: Any, skew_seconds: int) -> bool:
    claims = _decode_jwt_claims(access_token)
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        return False
    return float(exp) <= (time.time() + max(0, int(skew_seconds)))


# =============================================================================
# SSH / remote session detection
# =============================================================================

def _is_remote_session() -> bool:
    """Detect if running in an SSH session where webbrowser.open() won't work."""
    return bool(os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY"))


# =============================================================================
# OpenAI Codex auth file helpers
# =============================================================================

def resolve_codex_home_path() -> Path:
    """Resolve CODEX_HOME, defaulting to ~/.codex."""
    codex_home = os.getenv("CODEX_HOME", "").strip()
    if not codex_home:
        codex_home = str(Path.home() / ".codex")
    return Path(codex_home).expanduser()


def _codex_auth_file_path() -> Path:
    return resolve_codex_home_path() / "auth.json"


def _codex_auth_lock_path(auth_path: Path) -> Path:
    return auth_path.with_suffix(auth_path.suffix + ".lock")


@contextmanager
def _codex_auth_file_lock(
    auth_path: Path,
    timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS,
):
    lock_path = _codex_auth_lock_path(auth_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+") as lock_file:
        if fcntl is None:
            yield
            return

        deadline = time.time() + max(1.0, timeout_seconds)
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Timed out waiting for Codex auth lock: {lock_path}")
                time.sleep(0.05)

        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def read_codex_auth_file() -> Dict[str, Any]:
    """Read and validate Codex auth.json shape."""
    codex_home = resolve_codex_home_path()
    if not codex_home.exists():
        raise AuthError(
            f"Codex home directory not found at {codex_home}.",
            provider="openai-codex",
            code="codex_home_missing",
            relogin_required=True,
        )

    auth_path = codex_home / "auth.json"
    if not auth_path.exists():
        raise AuthError(
            f"Codex auth file not found at {auth_path}.",
            provider="openai-codex",
            code="codex_auth_missing",
            relogin_required=True,
        )

    try:
        payload = json.loads(auth_path.read_text())
    except Exception as exc:
        raise AuthError(
            f"Failed to parse Codex auth file at {auth_path}.",
            provider="openai-codex",
            code="codex_auth_invalid_json",
            relogin_required=True,
        ) from exc

    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        raise AuthError(
            "Codex auth file is missing a valid 'tokens' object.",
            provider="openai-codex",
            code="codex_auth_invalid_shape",
            relogin_required=True,
        )

    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise AuthError(
            "Codex auth file is missing tokens.access_token.",
            provider="openai-codex",
            code="codex_auth_missing_access_token",
            relogin_required=True,
        )
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise AuthError(
            "Codex auth file is missing tokens.refresh_token.",
            provider="openai-codex",
            code="codex_auth_missing_refresh_token",
            relogin_required=True,
        )

    return {
        "payload": payload,
        "tokens": tokens,
        "auth_path": auth_path,
        "codex_home": codex_home,
    }


def _persist_codex_auth_payload(
    auth_path: Path,
    payload: Dict[str, Any],
    *,
    lock_held: bool = False,
) -> None:
    auth_path.parent.mkdir(parents=True, exist_ok=True)

    def _write() -> None:
        serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        tmp_path = auth_path.parent / f".{auth_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            with tmp_path.open("w", encoding="utf-8") as tmp_file:
                tmp_file.write(serialized)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, auth_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        try:
            auth_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    if lock_held:
        _write()
        return

    with _codex_auth_file_lock(auth_path):
        _write()


def _refresh_codex_auth_tokens(
    *,
    payload: Dict[str, Any],
    auth_path: Path,
    timeout_seconds: float,
    lock_held: bool = False,
) -> Dict[str, Any]:
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        raise AuthError(
            "Codex auth file is missing a valid 'tokens' object.",
            provider="openai-codex",
            code="codex_auth_invalid_shape",
            relogin_required=True,
        )

    refresh_token = tokens.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise AuthError(
            "Codex auth file is missing tokens.refresh_token.",
            provider="openai-codex",
            code="codex_auth_missing_refresh_token",
            relogin_required=True,
        )

    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}) as client:
        response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        )

    if response.status_code != 200:
        code = "codex_refresh_failed"
        message = f"Codex token refresh failed with status {response.status_code}."
        relogin_required = False
        try:
            err = response.json()
            if isinstance(err, dict):
                err_code = err.get("error")
                if isinstance(err_code, str) and err_code.strip():
                    code = err_code.strip()
                err_desc = err.get("error_description") or err.get("message")
                if isinstance(err_desc, str) and err_desc.strip():
                    message = f"Codex token refresh failed: {err_desc.strip()}"
        except Exception:
            pass
        if code in {"invalid_grant", "invalid_token", "invalid_request"}:
            relogin_required = True
        raise AuthError(
            message,
            provider="openai-codex",
            code=code,
            relogin_required=relogin_required,
        )

    try:
        refresh_payload = response.json()
    except Exception as exc:
        raise AuthError(
            "Codex token refresh returned invalid JSON.",
            provider="openai-codex",
            code="codex_refresh_invalid_json",
            relogin_required=True,
        ) from exc

    access_token = refresh_payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise AuthError(
            "Codex token refresh response was missing access_token.",
            provider="openai-codex",
            code="codex_refresh_missing_access_token",
            relogin_required=True,
        )

    updated_tokens = dict(tokens)
    updated_tokens["access_token"] = access_token.strip()
    next_refresh = refresh_payload.get("refresh_token")
    if isinstance(next_refresh, str) and next_refresh.strip():
        updated_tokens["refresh_token"] = next_refresh.strip()
    payload["tokens"] = updated_tokens
    payload["last_refresh"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _persist_codex_auth_payload(auth_path, payload, lock_held=lock_held)
    return updated_tokens


def resolve_codex_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    """Resolve runtime credentials from Codex CLI auth state."""
    data = read_codex_auth_file()
    payload = data["payload"]
    tokens = dict(data["tokens"])
    auth_path = data["auth_path"]
    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_timeout_seconds = float(os.getenv("HERMES_CODEX_REFRESH_TIMEOUT_SECONDS", "20"))

    should_refresh = bool(force_refresh)
    if (not should_refresh) and refresh_if_expiring:
        should_refresh = _codex_access_token_is_expiring(access_token, refresh_skew_seconds)
    if should_refresh:
        lock_timeout = max(float(AUTH_LOCK_TIMEOUT_SECONDS), refresh_timeout_seconds + 5.0)
        with _codex_auth_file_lock(auth_path, timeout_seconds=lock_timeout):
            data = read_codex_auth_file()
            payload = data["payload"]
            tokens = dict(data["tokens"])
            access_token = str(tokens.get("access_token", "") or "").strip()

            should_refresh = bool(force_refresh)
            if (not should_refresh) and refresh_if_expiring:
                should_refresh = _codex_access_token_is_expiring(access_token, refresh_skew_seconds)

            if should_refresh:
                tokens = _refresh_codex_auth_tokens(
                    payload=payload,
                    auth_path=auth_path,
                    timeout_seconds=refresh_timeout_seconds,
                    lock_held=True,
                )
                access_token = str(tokens.get("access_token", "") or "").strip()

    base_url = (
        os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/")
        or DEFAULT_CODEX_BASE_URL
    )

    return {
        "provider": "openai-codex",
        "base_url": base_url,
        "api_key": access_token,
        "source": "codex-auth-json",
        "last_refresh": payload.get("last_refresh"),
        "auth_mode": payload.get("auth_mode"),
        "auth_file": str(auth_path),
        "codex_home": str(data["codex_home"]),
    }


# =============================================================================
# TLS verification helper
# =============================================================================

def _resolve_verify(
    *,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    auth_state: Optional[Dict[str, Any]] = None,
) -> bool | str:
    tls_state = auth_state.get("tls") if isinstance(auth_state, dict) else {}
    tls_state = tls_state if isinstance(tls_state, dict) else {}

    effective_insecure = (
        bool(insecure) if insecure is not None
        else bool(tls_state.get("insecure", False))
    )
    effective_ca = (
        ca_bundle
        or tls_state.get("ca_bundle")
        or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
    )

    if effective_insecure:
        return False
    if effective_ca:
        return str(effective_ca)
    return True


# =============================================================================
# OAuth Device Code Flow — generic, parameterized by provider
# =============================================================================

def _request_device_code(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    scope: Optional[str],
) -> Dict[str, Any]:
    """POST to the device code endpoint. Returns device_code, user_code, etc."""
    response = client.post(
        f"{portal_base_url}/api/oauth/device/code",
        data={
            "client_id": client_id,
            **({"scope": scope} if scope else {}),
        },
    )
    response.raise_for_status()
    data = response.json()

    required_fields = [
        "device_code", "user_code", "verification_uri",
        "verification_uri_complete", "expires_in", "interval",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Device code response missing fields: {', '.join(missing)}")
    return data


def _poll_for_token(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    device_code: str,
    expires_in: int,
    poll_interval: int,
) -> Dict[str, Any]:
    """Poll the token endpoint until the user approves or the code expires."""
    deadline = time.time() + max(1, expires_in)
    current_interval = max(1, min(poll_interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS))

    while time.time() < deadline:
        response = client.post(
            f"{portal_base_url}/api/oauth/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": client_id,
                "device_code": device_code,
            },
        )

        if response.status_code == 200:
            payload = response.json()
            if "access_token" not in payload:
                raise ValueError("Token response did not include access_token")
            return payload

        try:
            error_payload = response.json()
        except Exception:
            response.raise_for_status()
            raise RuntimeError("Token endpoint returned a non-JSON error response")

        error_code = error_payload.get("error", "")
        if error_code == "authorization_pending":
            time.sleep(current_interval)
            continue
        if error_code == "slow_down":
            current_interval = min(current_interval + 1, 30)
            time.sleep(current_interval)
            continue

        description = error_payload.get("error_description") or "Unknown authentication error"
        raise RuntimeError(f"{error_code}: {description}")

    raise TimeoutError("Timed out waiting for device authorization")


# =============================================================================
# Nous Portal — token refresh, agent key minting, model discovery
# =============================================================================

def _refresh_access_token(
    *,
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    refresh_token: str,
) -> Dict[str, Any]:
    response = client.post(
        f"{portal_base_url}/api/oauth/token",
        data={
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": refresh_token,
        },
    )

    if response.status_code == 200:
        payload = response.json()
        if "access_token" not in payload:
            raise AuthError("Refresh response missing access_token",
                            provider="nous", code="invalid_token", relogin_required=True)
        return payload

    try:
        error_payload = response.json()
    except Exception as exc:
        raise AuthError("Refresh token exchange failed",
                        provider="nous", relogin_required=True) from exc

    code = str(error_payload.get("error", "invalid_grant"))
    description = str(error_payload.get("error_description") or "Refresh token exchange failed")
    relogin = code in {"invalid_grant", "invalid_token"}
    raise AuthError(description, provider="nous", code=code, relogin_required=relogin)


def _mint_agent_key(
    *,
    client: httpx.Client,
    portal_base_url: str,
    access_token: str,
    min_ttl_seconds: int,
) -> Dict[str, Any]:
    """Mint (or reuse) a short-lived inference API key."""
    response = client.post(
        f"{portal_base_url}/api/oauth/agent-key",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"min_ttl_seconds": max(60, int(min_ttl_seconds))},
    )

    if response.status_code == 200:
        payload = response.json()
        if "api_key" not in payload:
            raise AuthError("Mint response missing api_key",
                            provider="nous", code="server_error")
        return payload

    try:
        error_payload = response.json()
    except Exception as exc:
        raise AuthError("Agent key mint request failed",
                        provider="nous", code="server_error") from exc

    code = str(error_payload.get("error", "server_error"))
    description = str(error_payload.get("error_description") or "Agent key mint request failed")
    relogin = code in {"invalid_token", "invalid_grant"}
    raise AuthError(description, provider="nous", code=code, relogin_required=relogin)


def fetch_nous_models(
    *,
    inference_base_url: str,
    api_key: str,
    timeout_seconds: float = 15.0,
    verify: bool | str = True,
) -> List[str]:
    """Fetch available model IDs from the Nous inference API."""
    timeout = httpx.Timeout(timeout_seconds)
    with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
        response = client.get(
            f"{inference_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )

    if response.status_code != 200:
        description = f"/models request failed with status {response.status_code}"
        try:
            err = response.json()
            description = str(err.get("error_description") or err.get("error") or description)
        except Exception as e:
            logger.debug("Could not parse error response JSON: %s", e)
        raise AuthError(description, provider="nous", code="models_fetch_failed")

    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        return []

    model_ids: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            mid = model_id.strip()
            # Skip Hermes models — they're not reliable for agentic tool-calling
            if "hermes" in mid.lower():
                continue
            model_ids.append(mid)

    return list(dict.fromkeys(model_ids))


def _agent_key_is_usable(state: Dict[str, Any], min_ttl_seconds: int) -> bool:
    key = state.get("agent_key")
    if not isinstance(key, str) or not key.strip():
        return False
    return not _is_expiring(state.get("agent_key_expires_at"), min_ttl_seconds)


def resolve_nous_runtime_credentials(
    *,
    min_key_ttl_seconds: int = DEFAULT_AGENT_KEY_MIN_TTL_SECONDS,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_mint: bool = False,
) -> Dict[str, Any]:
    """
    Resolve Nous inference credentials for runtime use.

    Ensures access_token is valid (refreshes if needed) and a short-lived
    inference key is present with minimum TTL (mints/reuses as needed).
    Concurrent processes coordinate through the auth store file lock.

    Returns dict with: provider, base_url, api_key, key_id, expires_at,
    expires_in, source ("cache" or "portal").
    """
    min_key_ttl_seconds = max(60, int(min_key_ttl_seconds))

    with _auth_store_lock():
        auth_store = _load_auth_store()
        state = _load_provider_state(auth_store, "nous")

        if not state:
            raise AuthError("Hermes is not logged into Nous Portal.",
                            provider="nous", relogin_required=True)

        portal_base_url = (
            _optional_base_url(state.get("portal_base_url"))
            or os.getenv("HERMES_PORTAL_BASE_URL")
            or os.getenv("NOUS_PORTAL_BASE_URL")
            or DEFAULT_NOUS_PORTAL_URL
        ).rstrip("/")
        inference_base_url = (
            _optional_base_url(state.get("inference_base_url"))
            or os.getenv("NOUS_INFERENCE_BASE_URL")
            or DEFAULT_NOUS_INFERENCE_URL
        ).rstrip("/")
        client_id = str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID)

        verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
        timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)

        with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
            access_token = state.get("access_token")
            refresh_token = state.get("refresh_token")

            if not isinstance(access_token, str) or not access_token:
                raise AuthError("No access token found for Nous Portal login.",
                                provider="nous", relogin_required=True)

            # Step 1: refresh access token if expiring
            if _is_expiring(state.get("expires_at"), ACCESS_TOKEN_REFRESH_SKEW_SECONDS):
                if not isinstance(refresh_token, str) or not refresh_token:
                    raise AuthError("Session expired and no refresh token is available.",
                                    provider="nous", relogin_required=True)

                refreshed = _refresh_access_token(
                    client=client, portal_base_url=portal_base_url,
                    client_id=client_id, refresh_token=refresh_token,
                )
                now = datetime.now(timezone.utc)
                access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
                state["access_token"] = refreshed["access_token"]
                state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
                state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
                state["scope"] = refreshed.get("scope") or state.get("scope")
                refreshed_url = _optional_base_url(refreshed.get("inference_base_url"))
                if refreshed_url:
                    inference_base_url = refreshed_url
                state["obtained_at"] = now.isoformat()
                state["expires_in"] = access_ttl
                state["expires_at"] = datetime.fromtimestamp(
                    now.timestamp() + access_ttl, tz=timezone.utc
                ).isoformat()
                access_token = state["access_token"]

            # Step 2: mint agent key if missing/expiring
            used_cached_key = False
            mint_payload: Optional[Dict[str, Any]] = None

            if not force_mint and _agent_key_is_usable(state, min_key_ttl_seconds):
                used_cached_key = True
            else:
                try:
                    mint_payload = _mint_agent_key(
                        client=client, portal_base_url=portal_base_url,
                        access_token=access_token, min_ttl_seconds=min_key_ttl_seconds,
                    )
                except AuthError as exc:
                    # Retry path: access token may be stale server-side despite local checks
                    if exc.code in {"invalid_token", "invalid_grant"} and isinstance(refresh_token, str) and refresh_token:
                        refreshed = _refresh_access_token(
                            client=client, portal_base_url=portal_base_url,
                            client_id=client_id, refresh_token=refresh_token,
                        )
                        now = datetime.now(timezone.utc)
                        access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
                        state["access_token"] = refreshed["access_token"]
                        state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
                        state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
                        state["scope"] = refreshed.get("scope") or state.get("scope")
                        refreshed_url = _optional_base_url(refreshed.get("inference_base_url"))
                        if refreshed_url:
                            inference_base_url = refreshed_url
                        state["obtained_at"] = now.isoformat()
                        state["expires_in"] = access_ttl
                        state["expires_at"] = datetime.fromtimestamp(
                            now.timestamp() + access_ttl, tz=timezone.utc
                        ).isoformat()
                        access_token = state["access_token"]

                        mint_payload = _mint_agent_key(
                            client=client, portal_base_url=portal_base_url,
                            access_token=access_token, min_ttl_seconds=min_key_ttl_seconds,
                        )
                    else:
                        raise

            if mint_payload is not None:
                now = datetime.now(timezone.utc)
                state["agent_key"] = mint_payload.get("api_key")
                state["agent_key_id"] = mint_payload.get("key_id")
                state["agent_key_expires_at"] = mint_payload.get("expires_at")
                state["agent_key_expires_in"] = mint_payload.get("expires_in")
                state["agent_key_reused"] = bool(mint_payload.get("reused", False))
                state["agent_key_obtained_at"] = now.isoformat()
                minted_url = _optional_base_url(mint_payload.get("inference_base_url"))
                if minted_url:
                    inference_base_url = minted_url

            # Persist routing and TLS metadata for non-interactive refresh/mint
            state["portal_base_url"] = portal_base_url
            state["inference_base_url"] = inference_base_url
            state["client_id"] = client_id
            state["tls"] = {
                "insecure": verify is False,
                "ca_bundle": verify if isinstance(verify, str) else None,
            }

        _save_provider_state(auth_store, "nous", state)
        _save_auth_store(auth_store)

    api_key = state.get("agent_key")
    if not isinstance(api_key, str) or not api_key:
        raise AuthError("Failed to resolve a Nous inference API key",
                        provider="nous", code="server_error")

    expires_at = state.get("agent_key_expires_at")
    expires_epoch = _parse_iso_timestamp(expires_at)
    expires_in = (
        max(0, int(expires_epoch - time.time()))
        if expires_epoch is not None
        else _coerce_ttl_seconds(state.get("agent_key_expires_in"))
    )

    return {
        "provider": "nous",
        "base_url": inference_base_url,
        "api_key": api_key,
        "key_id": state.get("agent_key_id"),
        "expires_at": expires_at,
        "expires_in": expires_in,
        "source": "cache" if used_cached_key else "portal",
    }


# =============================================================================
# Status helpers
# =============================================================================

def get_nous_auth_status() -> Dict[str, Any]:
    """Status snapshot for `hermes status` output."""
    state = get_provider_auth_state("nous")
    if not state:
        return {
            "logged_in": False,
            "portal_base_url": None,
            "inference_base_url": None,
            "access_expires_at": None,
            "agent_key_expires_at": None,
            "has_refresh_token": False,
        }
    return {
        "logged_in": bool(state.get("access_token")),
        "portal_base_url": state.get("portal_base_url"),
        "inference_base_url": state.get("inference_base_url"),
        "access_expires_at": state.get("expires_at"),
        "agent_key_expires_at": state.get("agent_key_expires_at"),
        "has_refresh_token": bool(state.get("refresh_token")),
    }


def get_codex_auth_status() -> Dict[str, Any]:
    """Status snapshot for Codex auth."""
    state = get_provider_auth_state("openai-codex") or {}
    auth_file = state.get("auth_file") or str(_codex_auth_file_path())
    codex_home = state.get("codex_home") or str(resolve_codex_home_path())
    try:
        creds = resolve_codex_runtime_credentials()
        return {
            "logged_in": True,
            "auth_file": creds.get("auth_file"),
            "codex_home": creds.get("codex_home"),
            "last_refresh": creds.get("last_refresh"),
            "auth_mode": creds.get("auth_mode"),
            "source": creds.get("source"),
        }
    except AuthError as exc:
        return {
            "logged_in": False,
            "auth_file": auth_file,
            "codex_home": codex_home,
            "error": str(exc),
        }


def get_auth_status(provider_id: Optional[str] = None) -> Dict[str, Any]:
    """Generic auth status dispatcher."""
    target = provider_id or get_active_provider()
    if target == "nous":
        return get_nous_auth_status()
    if target == "openai-codex":
        return get_codex_auth_status()
    return {"logged_in": False}


# =============================================================================
# CLI Commands — login / logout
# =============================================================================

def _update_config_for_provider(provider_id: str, inference_base_url: str) -> Path:
    """Update config.yaml and auth.json to reflect the active provider."""
    # Set active_provider in auth.json so auto-resolution picks this provider
    with _auth_store_lock():
        auth_store = _load_auth_store()
        auth_store["active_provider"] = provider_id
        _save_auth_store(auth_store)

    # Update config.yaml model section
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = yaml.safe_load(config_path.read_text()) or {}
            if isinstance(loaded, dict):
                config = loaded
        except Exception:
            config = {}

    current_model = config.get("model")
    if isinstance(current_model, dict):
        model_cfg = dict(current_model)
    elif isinstance(current_model, str) and current_model.strip():
        model_cfg = {"default": current_model.strip()}
    else:
        model_cfg = {}

    model_cfg["provider"] = provider_id
    model_cfg["base_url"] = inference_base_url.rstrip("/")
    config["model"] = model_cfg

    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def _reset_config_provider() -> Path:
    """Reset config.yaml provider back to auto after logout."""
    config_path = get_config_path()
    if not config_path.exists():
        return config_path

    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return config_path

    if not isinstance(config, dict):
        return config_path

    model = config.get("model")
    if isinstance(model, dict):
        model["provider"] = "auto"
        if "base_url" in model:
            model["base_url"] = OPENROUTER_BASE_URL
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def _prompt_model_selection(model_ids: List[str], current_model: str = "") -> Optional[str]:
    """Interactive model selection. Puts current_model first with a marker. Returns chosen model ID or None."""
    # Reorder: current model first, then the rest (deduplicated)
    ordered = []
    if current_model and current_model in model_ids:
        ordered.append(current_model)
    for mid in model_ids:
        if mid not in ordered:
            ordered.append(mid)

    # Build display labels with marker on current
    def _label(mid):
        if mid == current_model:
            return f"{mid}  ← currently in use"
        return mid

    # Default cursor on the current model (index 0 if it was reordered to top)
    default_idx = 0

    # Try arrow-key menu first, fall back to number input
    try:
        from simple_term_menu import TerminalMenu
        choices = [f"  {_label(mid)}" for mid in ordered]
        choices.append("  Enter custom model name")
        choices.append("  Skip (keep current)")
        menu = TerminalMenu(
            choices,
            cursor_index=default_idx,
            menu_cursor="-> ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("fg_green",),
            cycle_cursor=True,
            clear_screen=False,
            title="Select default model:",
        )
        idx = menu.show()
        if idx is None:
            return None
        print()
        if idx < len(ordered):
            return ordered[idx]
        elif idx == len(ordered):
            custom = input("Enter model name: ").strip()
            return custom if custom else None
        return None
    except (ImportError, NotImplementedError):
        pass

    # Fallback: numbered list
    print("Select default model:")
    for i, mid in enumerate(ordered, 1):
        print(f"  {i}. {_label(mid)}")
    n = len(ordered)
    print(f"  {n + 1}. Enter custom model name")
    print(f"  {n + 2}. Skip (keep current)")
    print()

    while True:
        try:
            choice = input(f"Choice [1-{n + 2}] (default: skip): ").strip()
            if not choice:
                return None
            idx = int(choice)
            if 1 <= idx <= n:
                return ordered[idx - 1]
            elif idx == n + 1:
                custom = input("Enter model name: ").strip()
                return custom if custom else None
            elif idx == n + 2:
                return None
            print(f"Please enter 1-{n + 2}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            return None


def _save_model_choice(model_id: str) -> None:
    """Save the selected model to config.yaml and .env."""
    from hermes_cli.config import save_config, load_config, save_env_value

    config = load_config()
    # Handle both string and dict model formats
    if isinstance(config.get("model"), dict):
        config["model"]["default"] = model_id
    else:
        config["model"] = model_id
    save_config(config)
    save_env_value("LLM_MODEL", model_id)


def login_command(args) -> None:
    """Run OAuth device code login for the selected provider."""
    provider_id = getattr(args, "provider", None) or "nous"

    if provider_id not in PROVIDER_REGISTRY:
        print(f"Unknown provider: {provider_id}")
        print(f"Available: {', '.join(PROVIDER_REGISTRY.keys())}")
        raise SystemExit(1)

    pconfig = PROVIDER_REGISTRY[provider_id]

    if provider_id == "nous":
        _login_nous(args, pconfig)
    elif provider_id == "openai-codex":
        _login_openai_codex(args, pconfig)
    else:
        print(f"Login for provider '{provider_id}' is not yet implemented.")
        raise SystemExit(1)


def _login_openai_codex(args, pconfig: ProviderConfig) -> None:
    """OpenAI Codex login flow using Codex CLI auth state."""
    codex_path = shutil.which("codex")
    if not codex_path:
        print("Codex CLI was not found in PATH.")
        print("Install Codex CLI, then retry `hermes login --provider openai-codex`.")
        raise SystemExit(1)

    print(f"Starting Hermes login via {pconfig.name}...")
    print(f"Using Codex CLI: {codex_path}")
    print(f"Codex home: {resolve_codex_home_path()}")

    creds: Dict[str, Any]
    try:
        creds = resolve_codex_runtime_credentials()
    except AuthError:
        print("No usable Codex auth found. Running `codex login`...")
        try:
            subprocess.run(["codex", "login"], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Codex login failed with exit code {exc.returncode}.")
            raise SystemExit(1)
        except KeyboardInterrupt:
            print("\nLogin cancelled.")
            raise SystemExit(130)
        try:
            creds = resolve_codex_runtime_credentials()
        except AuthError as exc:
            print(format_auth_error(exc))
            raise SystemExit(1)

    auth_state = {
        "auth_file": creds.get("auth_file"),
        "codex_home": creds.get("codex_home"),
        "last_refresh": creds.get("last_refresh"),
        "auth_mode": creds.get("auth_mode"),
        "source": creds.get("source"),
    }

    with _auth_store_lock():
        auth_store = _load_auth_store()
        _save_provider_state(auth_store, "openai-codex", auth_state)
        saved_to = _save_auth_store(auth_store)

    config_path = _update_config_for_provider("openai-codex", creds["base_url"])
    print()
    print("Login successful!")
    print(f"  Auth state: {saved_to}")
    print(f"  Config updated: {config_path} (model.provider=openai-codex)")


def _login_nous(args, pconfig: ProviderConfig) -> None:
    """Nous Portal device authorization flow."""
    portal_base_url = (
        getattr(args, "portal_url", None)
        or os.getenv("HERMES_PORTAL_BASE_URL")
        or os.getenv("NOUS_PORTAL_BASE_URL")
        or pconfig.portal_base_url
    ).rstrip("/")
    requested_inference_url = (
        getattr(args, "inference_url", None)
        or os.getenv("NOUS_INFERENCE_BASE_URL")
        or pconfig.inference_base_url
    ).rstrip("/")
    client_id = getattr(args, "client_id", None) or pconfig.client_id
    scope = getattr(args, "scope", None) or pconfig.scope
    open_browser = not getattr(args, "no_browser", False)
    timeout_seconds = getattr(args, "timeout", None) or 15.0
    timeout = httpx.Timeout(timeout_seconds)

    insecure = bool(getattr(args, "insecure", False))
    ca_bundle = (
        getattr(args, "ca_bundle", None)
        or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
    )
    verify: bool | str = False if insecure else (ca_bundle if ca_bundle else True)

    # Skip browser open in SSH sessions
    if _is_remote_session():
        open_browser = False

    print(f"Starting Hermes login via {pconfig.name}...")
    print(f"Portal: {portal_base_url}")
    if insecure:
        print("TLS verification: disabled (--insecure)")
    elif ca_bundle:
        print(f"TLS verification: custom CA bundle ({ca_bundle})")

    try:
        with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
            device_data = _request_device_code(
                client=client, portal_base_url=portal_base_url,
                client_id=client_id, scope=scope,
            )

            verification_url = str(device_data["verification_uri_complete"])
            user_code = str(device_data["user_code"])
            expires_in = int(device_data["expires_in"])
            interval = int(device_data["interval"])

            print()
            print("To continue:")
            print(f"  1. Open: {verification_url}")
            print(f"  2. If prompted, enter code: {user_code}")

            if open_browser:
                opened = webbrowser.open(verification_url)
                if opened:
                    print("  (Opened browser for verification)")
                else:
                    print("  Could not open browser automatically — use the URL above.")

            effective_interval = max(1, min(interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS))
            print(f"Waiting for approval (polling every {effective_interval}s)...")

            token_data = _poll_for_token(
                client=client, portal_base_url=portal_base_url,
                client_id=client_id, device_code=str(device_data["device_code"]),
                expires_in=expires_in, poll_interval=interval,
            )

        # Process token response
        now = datetime.now(timezone.utc)
        token_expires_in = _coerce_ttl_seconds(token_data.get("expires_in", 0))
        expires_at = now.timestamp() + token_expires_in
        inference_base_url = (
            _optional_base_url(token_data.get("inference_base_url"))
            or requested_inference_url
        )
        if inference_base_url != requested_inference_url:
            print(f"Using portal-provided inference URL: {inference_base_url}")

        auth_state = {
            "portal_base_url": portal_base_url,
            "inference_base_url": inference_base_url,
            "client_id": client_id,
            "scope": token_data.get("scope") or scope,
            "token_type": token_data.get("token_type", "Bearer"),
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "obtained_at": now.isoformat(),
            "expires_at": datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
            "expires_in": token_expires_in,
            "tls": {
                "insecure": verify is False,
                "ca_bundle": verify if isinstance(verify, str) else None,
            },
            "agent_key": None,
            "agent_key_id": None,
            "agent_key_expires_at": None,
            "agent_key_expires_in": None,
            "agent_key_reused": None,
            "agent_key_obtained_at": None,
        }

        # Save auth state
        with _auth_store_lock():
            auth_store = _load_auth_store()
            _save_provider_state(auth_store, "nous", auth_state)
            saved_to = _save_auth_store(auth_store)

        config_path = _update_config_for_provider("nous", inference_base_url)
        print()
        print("Login successful!")
        print(f"  Auth state: {saved_to}")
        print(f"  Config updated: {config_path} (model.provider=nous)")

        # Mint an initial agent key and list available models
        try:
            runtime_creds = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=5 * 60,
                timeout_seconds=timeout_seconds,
                insecure=insecure, ca_bundle=ca_bundle,
            )
            runtime_key = runtime_creds.get("api_key")
            runtime_base_url = runtime_creds.get("base_url") or inference_base_url
            if not isinstance(runtime_key, str) or not runtime_key:
                raise AuthError("No runtime API key available to fetch models",
                                provider="nous", code="invalid_token")

            model_ids = fetch_nous_models(
                inference_base_url=runtime_base_url,
                api_key=runtime_key,
                timeout_seconds=timeout_seconds,
                verify=verify,
            )

            print()
            if model_ids:
                selected_model = _prompt_model_selection(model_ids)
                if selected_model:
                    _save_model_choice(selected_model)
                    print(f"Default model set to: {selected_model}")
            else:
                print("No models were returned by the inference API.")
        except Exception as exc:
            message = format_auth_error(exc) if isinstance(exc, AuthError) else str(exc)
            print()
            print(f"Login succeeded, but could not fetch available models. Reason: {message}")

    except KeyboardInterrupt:
        print("\nLogin cancelled.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Login failed: {exc}")
        raise SystemExit(1)


def logout_command(args) -> None:
    """Clear auth state for a provider."""
    provider_id = getattr(args, "provider", None)

    if provider_id and provider_id not in PROVIDER_REGISTRY:
        print(f"Unknown provider: {provider_id}")
        raise SystemExit(1)

    active = get_active_provider()
    target = provider_id or active

    if not target:
        print("No provider is currently logged in.")
        return

    provider_name = PROVIDER_REGISTRY[target].name if target in PROVIDER_REGISTRY else target

    if clear_provider_auth(target):
        _reset_config_provider()
        print(f"Logged out of {provider_name}.")
        if os.getenv("OPENROUTER_API_KEY"):
            print("Hermes will use OpenRouter for inference.")
        else:
            print("Run `hermes login` or configure an API key to use Hermes.")
    else:
        print(f"No auth state found for {provider_name}.")

"""
Device authorization login flow and runtime auth helpers for Hermes CLI.
"""

from __future__ import annotations

import json
import os
import time
import webbrowser
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List

import httpx
import yaml

from hermes_cli.config import get_hermes_home

try:
    import fcntl  # POSIX file locking (macOS/Linux)
except Exception:  # pragma: no cover - Windows fallback
    fcntl = None

DEFAULT_PORTAL_BASE_URL = "https://portal.nousresearch.com"
DEFAULT_INFERENCE_BASE_URL = "https://inference-api.nousresearch.com/v1"
DEFAULT_CLIENT_ID = "hermes-cli"
DEFAULT_SCOPE = "inference:mint_agent_key"
DEFAULT_AGENT_KEY_MIN_TTL_SECONDS = 30 * 60
ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS = 1
AUTH_STORE_VERSION = 1
NOUS_PORTAL_AUTH_KEY = "nous_portal"
AUTH_LOCK_TIMEOUT_SECONDS = 15.0


class NousAuthError(RuntimeError):
    """Structured auth error for CLI UX mapping."""

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        relogin_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.relogin_required = relogin_required


def format_nous_auth_error(error: Exception) -> str:
    """Map low-level auth failures to concise user-facing guidance."""
    if not isinstance(error, NousAuthError):
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


def _resolve_portal_base_url(explicit_url: Optional[str]) -> str:
    base_url = (
        explicit_url
        or os.getenv("HERMES_PORTAL_BASE_URL")
        or os.getenv("NOUS_PORTAL_BASE_URL")
        or DEFAULT_PORTAL_BASE_URL
    )
    return base_url.rstrip("/")


def _resolve_inference_base_url(explicit_url: Optional[str] = None) -> str:
    base_url = explicit_url or os.getenv("NOUS_INFERENCE_BASE_URL") or DEFAULT_INFERENCE_BASE_URL
    return base_url.rstrip("/")


def _optional_base_url(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().rstrip("/")
    return cleaned if cleaned else None


def _auth_file_path() -> Path:
    return get_hermes_home() / "config" / "auth.json"


def _auth_lock_path() -> Path:
    return _auth_file_path().with_suffix(".lock")


@contextmanager
def _auth_store_lock(timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS):
    """Cross-process lock for auth.json reads+writes and mint/refresh operations."""
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


def _load_auth_store(auth_file: Path) -> Dict[str, Any]:
    if not auth_file.exists():
        return {
            "version": AUTH_STORE_VERSION,
            "systems": {},
        }

    try:
        raw = json.loads(auth_file.read_text())
    except Exception:
        return {
            "version": AUTH_STORE_VERSION,
            "systems": {},
        }

    if isinstance(raw, dict) and isinstance(raw.get("systems"), dict):
        return raw

    return {
        "version": AUTH_STORE_VERSION,
        "systems": {},
    }


def _save_auth_store(auth_store: Dict[str, Any]) -> Path:
    auth_file = _auth_file_path()
    auth_file.parent.mkdir(parents=True, exist_ok=True)
    auth_store["version"] = AUTH_STORE_VERSION
    auth_store["updated_at"] = datetime.now(timezone.utc).isoformat()
    auth_file.write_text(json.dumps(auth_store, indent=2) + "\n")
    return auth_file


def _load_nous_state(auth_store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    systems = auth_store.get("systems")
    if not isinstance(systems, dict):
        return None

    state = systems.get(NOUS_PORTAL_AUTH_KEY)
    return state if isinstance(state, dict) else None


def _save_nous_state(auth_store: Dict[str, Any], payload: Dict[str, Any]) -> None:
    systems = auth_store.setdefault("systems", {})
    if not isinstance(systems, dict):
        auth_store["systems"] = {}
        systems = auth_store["systems"]
    systems[NOUS_PORTAL_AUTH_KEY] = payload


def get_nous_portal_auth_state() -> Optional[Dict[str, Any]]:
    """Return persisted Nous auth state if present."""
    auth_store = _load_auth_store(_auth_file_path())
    state = _load_nous_state(auth_store)
    if not state:
        return None
    return dict(state)


def _save_auth_state(payload: Dict[str, Any]) -> Path:
    with _auth_store_lock():
        auth_store = _load_auth_store(_auth_file_path())
        _save_nous_state(auth_store, payload)
        return _save_auth_store(auth_store)


def _update_cli_model_config_for_nous(inference_base_url: str) -> Path:
    """
    Set CLI defaults to use Nous provider after successful portal login.

    Preserves existing model.default when config currently stores model as a string.
    """
    config_path = get_hermes_home() / "config.yaml"
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

    model_cfg["provider"] = "nous"
    model_cfg["base_url"] = inference_base_url.rstrip("/")
    config["model"] = model_cfg

    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


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


def _resolve_verify(
    *,
    insecure: Optional[bool],
    ca_bundle: Optional[str],
    auth_state: Optional[Dict[str, Any]],
) -> bool | str:
    tls_state = auth_state.get("tls") if isinstance(auth_state, dict) else {}
    tls_state = tls_state if isinstance(tls_state, dict) else {}

    effective_insecure = (
        bool(insecure)
        if insecure is not None
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


def _request_device_code(
    client: httpx.Client,
    portal_base_url: str,
    client_id: str,
    scope: Optional[str],
) -> Dict[str, Any]:
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
        "device_code",
        "user_code",
        "verification_uri",
        "verification_uri_complete",
        "expires_in",
        "interval",
    ]
    missing = [field for field in required_fields if field not in data]
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
    deadline = time.time() + max(1, expires_in)
    # Cap the client polling cadence to keep post-approval latency low.
    # If the server needs slower polling it can respond with slow_down.
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
            raise NousAuthError("Refresh response missing access_token", code="invalid_token", relogin_required=True)
        return payload

    try:
        error_payload = response.json()
    except Exception as exc:
        raise NousAuthError("Refresh token exchange failed", relogin_required=True) from exc

    code = str(error_payload.get("error", "invalid_grant"))
    description = str(error_payload.get("error_description") or "Refresh token exchange failed")
    relogin = code in {"invalid_grant", "invalid_token"}
    raise NousAuthError(description, code=code, relogin_required=relogin)


def _mint_agent_key(
    *,
    client: httpx.Client,
    portal_base_url: str,
    access_token: str,
    min_ttl_seconds: int,
) -> Dict[str, Any]:
    response = client.post(
        f"{portal_base_url}/api/oauth/agent-key",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"min_ttl_seconds": max(60, int(min_ttl_seconds))},
    )

    if response.status_code == 200:
        payload = response.json()
        if "api_key" not in payload:
            raise NousAuthError("Mint response missing api_key", code="server_error")
        return payload

    try:
        error_payload = response.json()
    except Exception as exc:
        raise NousAuthError("Agent key mint request failed", code="server_error") from exc

    code = str(error_payload.get("error", "server_error"))
    description = str(error_payload.get("error_description") or "Agent key mint request failed")

    relogin = code in {"invalid_token", "invalid_grant"}
    raise NousAuthError(description, code=code, relogin_required=relogin)


def _fetch_available_models(
    *,
    client: httpx.Client,
    inference_base_url: str,
    api_key: str,
) -> List[str]:
    response = client.get(
        f"{inference_base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    if response.status_code != 200:
        description = f"/models request failed with status {response.status_code}"
        try:
            error_payload = response.json()
            description = str(error_payload.get("error_description") or error_payload.get("error") or description)
        except Exception:
            pass
        raise NousAuthError(description, code="models_fetch_failed")

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
            model_ids.append(model_id.strip())

    # Keep stable order from API while removing duplicates.
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

    Ensures:
    - access_token is valid (refreshes if needed)
    - short-lived inference key is present and has minimum TTL (mints/reuses)
    - concurrent processes coordinate through auth store lock
    """
    min_key_ttl_seconds = max(60, int(min_key_ttl_seconds))

    with _auth_store_lock():
        auth_file = _auth_file_path()
        auth_store = _load_auth_store(auth_file)
        state = _load_nous_state(auth_store)

        if not state:
            raise NousAuthError("Hermes is not logged into Nous Portal.", relogin_required=True)

        portal_base_url = _resolve_portal_base_url(state.get("portal_base_url"))
        inference_base_url = _resolve_inference_base_url(state.get("inference_base_url"))
        client_id = str(state.get("client_id") or DEFAULT_CLIENT_ID)

        verify = _resolve_verify(insecure=insecure, ca_bundle=ca_bundle, auth_state=state)
        timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)

        with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
            access_token = state.get("access_token")
            refresh_token = state.get("refresh_token")

            if not isinstance(access_token, str) or not access_token:
                raise NousAuthError("No access token found for Nous Portal login.", relogin_required=True)

            if _is_expiring(state.get("expires_at"), ACCESS_TOKEN_REFRESH_SKEW_SECONDS):
                if not isinstance(refresh_token, str) or not refresh_token:
                    raise NousAuthError("Session expired and no refresh token is available.", relogin_required=True)

                refreshed = _refresh_access_token(
                    client=client,
                    portal_base_url=portal_base_url,
                    client_id=client_id,
                    refresh_token=refresh_token,
                )

                now = datetime.now(timezone.utc)
                access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
                state["access_token"] = refreshed["access_token"]
                state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
                state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
                state["scope"] = refreshed.get("scope") or state.get("scope")
                refreshed_inference_url = _optional_base_url(refreshed.get("inference_base_url"))
                if refreshed_inference_url:
                    inference_base_url = refreshed_inference_url
                state["obtained_at"] = now.isoformat()
                state["expires_in"] = access_ttl
                state["expires_at"] = datetime.fromtimestamp(now.timestamp() + access_ttl, tz=timezone.utc).isoformat()
                access_token = state["access_token"]

            used_cached_key = False
            mint_payload: Optional[Dict[str, Any]] = None

            if not force_mint and _agent_key_is_usable(state, min_key_ttl_seconds):
                used_cached_key = True
            else:
                try:
                    mint_payload = _mint_agent_key(
                        client=client,
                        portal_base_url=portal_base_url,
                        access_token=access_token,
                        min_ttl_seconds=min_key_ttl_seconds,
                    )
                except NousAuthError as exc:
                    # One retry path: token may be stale on server side despite local expiry check.
                    if exc.code in {"invalid_token", "invalid_grant"} and isinstance(refresh_token, str) and refresh_token:
                        refreshed = _refresh_access_token(
                            client=client,
                            portal_base_url=portal_base_url,
                            client_id=client_id,
                            refresh_token=refresh_token,
                        )
                        now = datetime.now(timezone.utc)
                        access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
                        state["access_token"] = refreshed["access_token"]
                        state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
                        state["token_type"] = refreshed.get("token_type") or state.get("token_type") or "Bearer"
                        state["scope"] = refreshed.get("scope") or state.get("scope")
                        refreshed_inference_url = _optional_base_url(
                            refreshed.get("inference_base_url")
                        )
                        if refreshed_inference_url:
                            inference_base_url = refreshed_inference_url
                        state["obtained_at"] = now.isoformat()
                        state["expires_in"] = access_ttl
                        state["expires_at"] = datetime.fromtimestamp(now.timestamp() + access_ttl, tz=timezone.utc).isoformat()
                        access_token = state["access_token"]

                        mint_payload = _mint_agent_key(
                            client=client,
                            portal_base_url=portal_base_url,
                            access_token=access_token,
                            min_ttl_seconds=min_key_ttl_seconds,
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
                minted_inference_url = _optional_base_url(mint_payload.get("inference_base_url"))
                if minted_inference_url:
                    inference_base_url = minted_inference_url

            # Persist TLS and routing metadata for future non-interactive refresh/mint calls
            state["portal_base_url"] = portal_base_url
            state["inference_base_url"] = inference_base_url
            state["client_id"] = client_id
            state["tls"] = {
                "insecure": verify is False,
                "ca_bundle": verify if isinstance(verify, str) else None,
            }

        _save_nous_state(auth_store, state)
        _save_auth_store(auth_store)

    api_key = state.get("agent_key")
    if not isinstance(api_key, str) or not api_key:
        raise NousAuthError("Failed to resolve a Nous inference API key", code="server_error")

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


def get_nous_auth_status() -> Dict[str, Any]:
    """Small status snapshot for `hermes status` output."""
    state = get_nous_portal_auth_state()
    if not state:
        return {
            "logged_in": False,
            "portal_base_url": None,
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


def login_command(args) -> None:
    portal_base_url = _resolve_portal_base_url(getattr(args, "portal_url", None))
    requested_inference_base_url = _resolve_inference_base_url(getattr(args, "inference_url", None))
    client_id = getattr(args, "client_id", None) or DEFAULT_CLIENT_ID
    scope = getattr(args, "scope", None) or DEFAULT_SCOPE
    open_browser = not getattr(args, "no_browser", False)

    timeout_seconds = getattr(args, "timeout", None)
    timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)

    insecure = bool(getattr(args, "insecure", False))
    ca_bundle = getattr(args, "ca_bundle", None) or os.getenv("HERMES_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    verify: bool | str = False if insecure else (ca_bundle if ca_bundle else True)

    print("Starting Hermes login via device authorization flow...")
    print(f"Portal: {portal_base_url}")
    if insecure:
        print("TLS verification: disabled (--insecure)")
    elif ca_bundle:
        print(f"TLS verification: custom CA bundle ({ca_bundle})")

    try:
        with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as client:
            device_data = _request_device_code(
                client=client,
                portal_base_url=portal_base_url,
                client_id=client_id,
                scope=scope,
            )

            verification_uri_complete = str(device_data["verification_uri_complete"])
            user_code = str(device_data["user_code"])
            expires_in = int(device_data["expires_in"])
            interval = int(device_data["interval"])

            print()
            print("To continue:")
            print(f"1. Open: {verification_uri_complete}")
            print(f"2. If prompted, enter code: {user_code}")

            if open_browser:
                opened = webbrowser.open(verification_uri_complete)
                if opened:
                    print("Opened browser for verification.")
                else:
                    print("Could not automatically open browser; use the URL above.")

            effective_poll_interval = max(
                1,
                min(interval, DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS),
            )
            print(f"Waiting for approval (polling every {effective_poll_interval}s)...")

            token_data = _poll_for_token(
                client=client,
                portal_base_url=portal_base_url,
                client_id=client_id,
                device_code=str(device_data["device_code"]),
                expires_in=expires_in,
                poll_interval=interval,
            )

        now = datetime.now(timezone.utc)
        token_expires_in = _coerce_ttl_seconds(token_data.get("expires_in", 0))
        expires_at = now.timestamp() + token_expires_in
        inference_base_url = (
            _optional_base_url(token_data.get("inference_base_url"))
            or requested_inference_base_url
        )
        if inference_base_url != requested_inference_base_url:
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
            # Clear cached key material from older sessions on fresh login
            "agent_key": None,
            "agent_key_id": None,
            "agent_key_expires_at": None,
            "agent_key_expires_in": None,
            "agent_key_reused": None,
            "agent_key_obtained_at": None,
        }

        saved_to = _save_auth_state(auth_state)
        config_path = _update_cli_model_config_for_nous(inference_base_url)
        print("Login successful.")
        print(f"Saved auth state to: {saved_to} (systems.{NOUS_PORTAL_AUTH_KEY})")
        print(
            "Updated CLI config to prefer Nous provider: "
            f"{config_path} (model.provider=nous, model.base_url={inference_base_url})"
        )

        try:
            runtime_creds = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=5 * 60,
                timeout_seconds=timeout_seconds if timeout_seconds else 15.0,
                insecure=insecure,
                ca_bundle=ca_bundle,
            )
            runtime_key = runtime_creds.get("api_key")
            runtime_base_url = runtime_creds.get("base_url") or inference_base_url
            if not isinstance(runtime_key, str) or not runtime_key:
                raise NousAuthError("No runtime API key available to fetch models", code="invalid_token")

            with httpx.Client(timeout=timeout, headers={"Accept": "application/json"}, verify=verify) as model_client:
                model_ids = _fetch_available_models(
                    client=model_client,
                    inference_base_url=runtime_base_url,
                    api_key=runtime_key,
                )

            print()
            if model_ids:
                print(f"Available models ({len(model_ids)}):")
                for model_id in model_ids:
                    print(f"  - {model_id}")
            else:
                print("No models were returned by the inference API.")
        except Exception as exc:
            message = format_nous_auth_error(exc) if isinstance(exc, NousAuthError) else str(exc)
            print()
            print(
                "Login succeeded, but could not fetch available models. "
                f"Reason: {message}"
            )

    except KeyboardInterrupt:
        print("Login cancelled.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Login failed: {exc}")
        raise SystemExit(1)

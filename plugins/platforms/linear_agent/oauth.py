"""OAuth helpers for the Linear Agent platform plugin.

The runtime pieces in this module intentionally keep credentials out of logs.
For Linear Agent/app attribution, prefer Linear's ``client_credentials`` grant:
Hermes stores OAuth client credentials in the active Hermes profile `.env`,
caches the app-actor access tokens in the profile `auth.json`, and
obtains a replacement token before expiry or after a 401.

The module can also be executed directly to either mint a client-credentials
token or perform the older browser-based local callback flow:

    python -m plugins.platforms.linear_agent.oauth --profile <profile> \
        --client-id ... --client-secret ... --client-credentials
"""

from __future__ import annotations

import argparse
import asyncio
import weakref
import json
import logging
import os
import secrets
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime dependency checked elsewhere
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

LINEAR_AUTHORIZE_URL = "https://linear.app/oauth/authorize"
LINEAR_TOKEN_URL = "https://api.linear.app/oauth/token"
DEFAULT_REDIRECT_URI = "http://localhost:8765/oauth/linear/callback"
DEFAULT_SCOPES = "read,write,app:assignable,app:mentionable,customer:read,customer:write,initiative:read,initiative:write"
DEFAULT_ACTOR = "app"
AUTH_PROVIDER_ID = "linear_agent"
REFRESH_SKEW_SECONDS = 300

TokenUpdateCallback = Callable[[dict[str, Any]], None]


class LinearOAuthError(RuntimeError):
    """Raised when Linear OAuth setup or refresh fails."""


@dataclass
class LinearOAuthConfig:
    """Linear OAuth token configuration.

    ``client_credentials`` is the preferred Linear Agent/app path: it mints an
    app-actor access token from the OAuth client ID/secret and does not involve
    a browser flow or a refresh token. ``refresh_token`` remains supported as a
    fallback for older authorization-code installations.
    """

    client_id: str = ""
    client_secret: str = ""
    refresh_token: str = ""
    access_token: str = ""
    expires_at: float = 0.0
    token_url: str = LINEAR_TOKEN_URL
    oauth_scopes: str = DEFAULT_SCOPES
    refresh_skew_seconds: int = REFRESH_SKEW_SECONDS
    persist_callback: Optional[TokenUpdateCallback] = None
    session_factory: Optional[Callable[..., Any]] = None

    @property
    def can_client_credentials(self) -> bool:
        return bool(self.client_id and self.client_secret)

    @property
    def can_refresh(self) -> bool:
        return bool(self.client_id and self.refresh_token and self.client_secret)

    @property
    def configured(self) -> bool:
        return bool(self.access_token or self.can_client_credentials or self.can_refresh)


class LinearOAuthTokenManager:
    """Manage Linear OAuth access-token refreshes for the GraphQL client."""

    def __init__(self, config: LinearOAuthConfig) -> None:
        self.config = config
        # Per-loop: one manager serves the adapter loop AND the sync→async
        # bridge loops tool handlers run on; an asyncio.Lock is bound to the
        # loop that created it, so sharing one across loops raises. Worst
        # case cross-loop is a duplicate mint (harmless), never a crash.
        self._refresh_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
            weakref.WeakKeyDictionary()
        )

    @property
    def configured(self) -> bool:
        return self.config.configured

    @property
    def access_token(self) -> str:
        return self.config.access_token

    def _lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        lock = self._refresh_locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            self._refresh_locks[loop] = lock
        return lock

    def _expires_soon(self) -> bool:
        if not self.config.expires_at:
            return False
        return time.time() >= (self.config.expires_at - self.config.refresh_skew_seconds)

    async def get_access_token(self) -> str:
        """Return a usable access token, minting/reissuing one first when needed."""
        if self.config.access_token and not self._expires_soon():
            return self.config.access_token
        if self.config.can_client_credentials or self.config.can_refresh:
            await self.refresh_access_token()
        if self.config.access_token:
            return self.config.access_token
        raise LinearOAuthError(
            "Linear OAuth is not configured: set LINEAR_AGENT_ACCESS_TOKEN, "
            "or set LINEAR_AGENT_CLIENT_ID and LINEAR_AGENT_CLIENT_SECRET for "
            "client-credentials app tokens"
        )

    async def refresh_access_token(self) -> dict[str, Any]:
        """Issue a new access token and persist it if configured.

        Prefer Linear's client_credentials grant for app-actor tokens. If the
        workspace/app has not enabled client credentials but an older refresh
        token is present, fall back to the refresh_token grant.
        """
        if not (self.config.can_client_credentials or self.config.can_refresh):
            raise LinearOAuthError(
                "Cannot issue Linear OAuth token without client_id/client_secret "
                "or refresh-token credentials"
            )
        async with self._lock():
            if self.config.access_token and not self._expires_soon():
                return {
                    "access_token": self.config.access_token,
                    "refresh_token": self.config.refresh_token,
                    "expires_at": self.config.expires_at,
                }
            token_data: dict[str, Any] | None = None
            client_credentials_error: LinearOAuthError | None = None
            if self.config.can_client_credentials:
                try:
                    token_data = await _async_client_credentials_token(
                        token_url=self.config.token_url,
                        client_id=self.config.client_id,
                        client_secret=self.config.client_secret,
                        scope=self.config.oauth_scopes,
                        session_factory=self.config.session_factory,
                    )
                except LinearOAuthError as exc:
                    client_credentials_error = exc
                    if not self.config.can_refresh:
                        raise
            if token_data is None and self.config.can_refresh:
                try:
                    token_data = await _async_refresh_token(
                        token_url=self.config.token_url,
                        client_id=self.config.client_id,
                        client_secret=self.config.client_secret,
                        refresh_token=self.config.refresh_token,
                        session_factory=self.config.session_factory,
                    )
                except LinearOAuthError as exc:
                    if "invalid_grant" in str(exc).lower():
                        # The refresh token was revoked or expired. Discard it so
                        # later attempts don't keep retrying a dead credential,
                        # and tell the operator how to recover.
                        self.config.refresh_token = ""
                        raise LinearOAuthError(
                            "Linear refresh token was revoked or expired; re-run the "
                            "Linear OAuth flow (see plugins/platforms/linear_agent/README.md) "
                            "or configure LINEAR_AGENT_CLIENT_ID/LINEAR_AGENT_CLIENT_SECRET "
                            "for client-credentials tokens"
                        ) from exc
                    if client_credentials_error is not None:
                        raise client_credentials_error
                    raise
            if token_data is None:
                raise LinearOAuthError("Linear OAuth token request did not produce a response")
            self._apply_token_response(token_data)
            return token_data

    def force_refresh_after_auth_error(self) -> None:
        """Mark the cached token stale so the next refresh call actually refreshes."""
        self.config.expires_at = 0.0
        self.config.access_token = ""

    def _apply_token_response(self, token_data: dict[str, Any]) -> None:
        access_token = str(token_data.get("access_token") or "").strip()
        refresh_token = str(token_data.get("refresh_token") or "").strip()
        if not access_token:
            raise LinearOAuthError("Linear OAuth refresh response did not include access_token")
        if refresh_token:
            self.config.refresh_token = refresh_token
        self.config.access_token = access_token
        expires_in = _safe_int(token_data.get("expires_in"), 0)
        if expires_in > 0:
            self.config.expires_at = time.time() + expires_in
        token_data["expires_at"] = self.config.expires_at
        if self.config.persist_callback:
            self.config.persist_callback(token_data)


async def _async_token_request(
    token_url: str,
    payload: dict[str, str],
    label: str,
    session_factory: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """POST an OAuth token-grant form and return the JSON response."""
    if not AIOHTTP_AVAILABLE or aiohttp is None:
        raise LinearOAuthError("aiohttp is not installed")
    session_cls = session_factory or aiohttp.ClientSession
    async with session_cls() as session:
        async with session.post(token_url, data=payload) as response:
            text = await response.text()
            if response.status >= 300:
                raise LinearOAuthError(
                    f"Linear OAuth {label} failed with HTTP {response.status}: {text[:200]}"
                )
            try:
                return await response.json()
            except Exception as exc:  # noqa: BLE001
                raise LinearOAuthError(f"Linear OAuth {label} response was not JSON") from exc


async def _async_refresh_token(
    *,
    token_url: str,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    session_factory: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    return await _async_token_request(
        token_url,
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        "refresh",
        session_factory,
    )


async def _async_client_credentials_token(
    *,
    token_url: str,
    client_id: str,
    client_secret: str,
    scope: str = DEFAULT_SCOPES,
    session_factory: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    return await _async_token_request(
        token_url,
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope or DEFAULT_SCOPES,
        },
        "client_credentials",
        session_factory,
    )


def exchange_code_for_token(
    *,
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    token_url: str = LINEAR_TOKEN_URL,
) -> dict[str, Any]:
    """Exchange an OAuth authorization code for tokens using Linear's token endpoint."""
    payload = urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "client_secret": client_secret,
        }
    ).encode("utf-8")
    request = Request(
        token_url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - Linear OAuth endpoint
        body = response.read().decode("utf-8")
    data = json.loads(body)
    if not data.get("access_token"):
        raise LinearOAuthError("Linear OAuth token response did not include access_token")
    return data


def issue_client_credentials_token(
    *,
    client_id: str,
    client_secret: str,
    env_path: Path,
    auth_path: Path | None = None,
    scope: str = DEFAULT_SCOPES,
    token_url: str = LINEAR_TOKEN_URL,
) -> dict[str, Any]:
    """Mint and persist a Linear app-actor token using client_credentials."""
    payload = urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope or DEFAULT_SCOPES,
        }
    ).encode("utf-8")
    request = Request(
        token_url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - Linear OAuth endpoint
        body = response.read().decode("utf-8")
    token_data = json.loads(body)
    access_token = str(token_data.get("access_token") or "").strip()
    if not access_token:
        raise LinearOAuthError("Linear client_credentials response did not include access_token")
    expires_in = _safe_int(token_data.get("expires_in"), 0)
    expires_at = int(time.time() + expires_in) if expires_in else 0
    updates = {
        "LINEAR_AGENT_CLIENT_ID": client_id,
        "LINEAR_AGENT_CLIENT_SECRET": client_secret,
        "LINEAR_AGENT_OAUTH_SCOPES": scope or DEFAULT_SCOPES,
    }
    update_env_file(env_path, updates)
    remove_env_keys(
        env_path,
        {"LINEAR_AGENT_ACCESS_TOKEN", "LINEAR_AGENT_REFRESH_TOKEN", "LINEAR_AGENT_TOKEN_EXPIRES_AT"},
    )
    token_data["expires_at"] = expires_at
    auth_path = auth_path or _auth_path_for_env_path(env_path)
    persist_auth_token(
        auth_path,
        token_data,
        client_id=client_id,
        token_url=token_url,
        scope=scope or DEFAULT_SCOPES,
    )
    return {
        "env_path": str(env_path),
        "auth_path": str(auth_path),
        "expires_at": expires_at,
        "scope": token_data.get("scope") or scope,
    }


def build_authorization_url(
    *,
    client_id: str,
    redirect_uri: str,
    scope: str = DEFAULT_SCOPES,
    state: str,
    actor: str = DEFAULT_ACTOR,
    prompt_consent: bool = False,
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
    }
    if actor:
        params["actor"] = actor
    if prompt_consent:
        params["prompt"] = "consent"
    return f"{LINEAR_AUTHORIZE_URL}?{urlencode(params)}"


def persist_auth_token(
    auth_path: Path,
    token_data: dict[str, Any],
    *,
    client_id: str = "",
    token_url: str = LINEAR_TOKEN_URL,
    scope: str = DEFAULT_SCOPES,
) -> None:
    """Persist Linear Agent OAuth token state to profile auth.json."""
    access_token = str(token_data.get("access_token") or "").strip()
    if not access_token:
        return
    expires_at = _safe_int(token_data.get("expires_at"), 0)
    state: dict[str, Any] = {
        "auth_type": "oauth",
        "grant_type": "client_credentials",
        "access_token": access_token,
        "token_url": token_url,
        "scope": token_data.get("scope") or scope,
        "updated_at": int(time.time()),
    }
    if client_id:
        state["client_id"] = client_id
    refresh_token = str(token_data.get("refresh_token") or "").strip()
    if refresh_token:
        state["refresh_token"] = refresh_token
        state["grant_type"] = "refresh_token"
    if expires_at:
        state["expires_at"] = expires_at
    _write_auth_state(auth_path, state)


def read_auth_token(auth_path: Path) -> dict[str, Any]:
    """Read Linear Agent OAuth token state from profile auth.json."""
    try:
        from hermes_cli.auth import _auth_store_lock, _load_auth_store

        with _auth_store_lock():
            store = _load_auth_store(auth_path.expanduser())
    except Exception:
        return {}
    providers = store.get("providers")
    if not isinstance(providers, dict):
        return {}
    state = providers.get(AUTH_PROVIDER_ID)
    return dict(state) if isinstance(state, dict) else {}


def build_auth_token_update_callback(
    auth_path: Path,
    *,
    client_id: str = "",
    token_url: str = LINEAR_TOKEN_URL,
    scope: str = DEFAULT_SCOPES,
) -> TokenUpdateCallback:
    """Return a callback that persists rotated Linear tokens to auth.json."""

    def _persist(token_data: dict[str, Any]) -> None:
        persist_auth_token(
            auth_path,
            token_data,
            client_id=client_id,
            token_url=token_url,
            scope=scope,
        )

    return _persist


def _write_auth_state(auth_path: Path, state: dict[str, Any]) -> None:
    auth_path = auth_path.expanduser()
    try:
        from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store, _store_provider_state

        with _auth_store_lock():
            store = _load_auth_store(auth_path)
            _store_provider_state(store, AUTH_PROVIDER_ID, state, set_active=False)
            _save_auth_store(store, target_path=auth_path)
    except Exception as exc:  # pragma: no cover - auth store failures surface as OAuth setup errors
        raise LinearOAuthError(f"Failed to persist Linear Agent token to auth.json: {exc}") from exc


def _auth_path_for_env_path(env_path: Path) -> Path:
    return env_path.expanduser().with_name("auth.json")


def update_env_file(path: Path, updates: dict[str, str]) -> None:
    """Create or update keys in a dotenv file without printing secret values."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            out.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in remaining:
            out.append(f"{key}={_dotenv_quote(remaining.pop(key))}")
        else:
            out.append(line)
    for key, value in remaining.items():
        out.append(f"{key}={_dotenv_quote(value)}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def remove_env_keys(path: Path, keys: set[str]) -> None:
    """Remove keys from a dotenv file without exposing their values."""
    path = path.expanduser()
    if not path.exists() or not keys:
        return
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in keys:
                continue
        out.append(line)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def read_env_file(path: Path) -> dict[str, str]:
    """Read simple dotenv key/value pairs without executing the file as shell."""
    path = path.expanduser()
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = _dotenv_unquote(value.strip())
    return values


def _dotenv_quote(value: str) -> str:
    value = str(value)
    if value == "" or any(ch.isspace() for ch in value) or any(ch in value for ch in ['#', '"', "'"]):
        return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return value


def _dotenv_unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1].replace('\\"', '"').replace('\\\\', '\\')
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _env_path_for_profile(profile: str | None) -> Path:
    if profile:
        return Path.home() / ".hermes" / "profiles" / profile / ".env"
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / ".env"
    except Exception:  # pragma: no cover - fallback for direct standalone use
        return Path.home() / ".hermes" / ".env"


def run_local_oauth_flow(
    *,
    client_id: str,
    client_secret: str,
    env_path: Path,
    auth_path: Path | None = None,
    redirect_uri: str = DEFAULT_REDIRECT_URI,
    scope: str = DEFAULT_SCOPES,
    actor: str = DEFAULT_ACTOR,
    prompt_consent: bool = False,
    open_browser: bool = True,
) -> dict[str, Any]:
    """Run a localhost callback OAuth flow and persist Linear Agent tokens."""
    parsed = urlparse(redirect_uri)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"}:
        raise LinearOAuthError("The setup helper only handles localhost http redirect URIs")
    port = parsed.port or 80
    path = parsed.path or "/"
    state = secrets.token_urlsafe(32)
    result: dict[str, str] = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib API
            return

        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            request_url = urlparse(self.path)
            if request_url.path != path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")
                return
            query = parse_qs(request_url.query)
            returned_state = (query.get("state") or [""])[0]
            code = (query.get("code") or [""])[0]
            error = (query.get("error") or [""])[0]
            if error:
                result["error"] = error
            elif returned_state != state:
                result["error"] = "OAuth state mismatch"
            elif not code:
                result["error"] = "No OAuth code returned"
            else:
                result["code"] = code
            self.send_response(200 if "code" in result else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            if "code" in result:
                self.wfile.write(b"Linear authorization complete. You can close this tab and return to Hermes.\n")
            else:
                self.wfile.write(f"Linear authorization failed: {result.get('error', 'unknown error')}\n".encode())

    server = ThreadingHTTPServer((parsed.hostname or "localhost", port), CallbackHandler)
    url = build_authorization_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        state=state,
        actor=actor,
        prompt_consent=prompt_consent,
    )
    print("Open this Linear authorization URL if your browser did not open automatically:")
    print(url)
    if open_browser:
        webbrowser.open(url)
    server.handle_request()
    server.server_close()
    if result.get("error"):
        raise LinearOAuthError(result["error"])
    token_data = exchange_code_for_token(
        code=result["code"],
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
    )
    expires_in = _safe_int(token_data.get("expires_in"), 0)
    expires_at = int(time.time() + expires_in) if expires_in else 0
    updates = {
        "LINEAR_AGENT_CLIENT_ID": client_id,
        "LINEAR_AGENT_CLIENT_SECRET": client_secret,
        "LINEAR_AGENT_REDIRECT_URI": redirect_uri,
        "LINEAR_AGENT_OAUTH_SCOPES": scope,
        "LINEAR_AGENT_OAUTH_ACTOR": actor,
    }
    update_env_file(env_path, {k: v for k, v in updates.items() if v})
    remove_env_keys(env_path, {"LINEAR_AGENT_ACCESS_TOKEN", "LINEAR_AGENT_REFRESH_TOKEN", "LINEAR_AGENT_TOKEN_EXPIRES_AT"})
    token_data["expires_at"] = expires_at
    auth_path = auth_path or _auth_path_for_env_path(env_path)
    persist_auth_token(
        auth_path,
        token_data,
        client_id=client_id,
        token_url=LINEAR_TOKEN_URL,
        scope=scope,
    )
    return {
        "env_path": str(env_path),
        "auth_path": str(auth_path),
        "expires_at": expires_at,
        "scope": token_data.get("scope"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Authorize Linear Agent OAuth for a Hermes profile")
    parser.add_argument("--profile", help="Hermes profile name; writes ~/.hermes/profiles/<profile>/.env")
    parser.add_argument("--env-path", type=Path, help="Explicit .env path to update")
    parser.add_argument("--auth-path", type=Path, help="Explicit auth.json path to update")
    parser.add_argument("--client-id", default=os.getenv("LINEAR_AGENT_CLIENT_ID", ""))
    parser.add_argument("--client-secret", default=os.getenv("LINEAR_AGENT_CLIENT_SECRET", ""))
    parser.add_argument("--redirect-uri", default=os.getenv("LINEAR_AGENT_REDIRECT_URI", ""))
    parser.add_argument("--scope", default=os.getenv("LINEAR_AGENT_OAUTH_SCOPES", ""))
    parser.add_argument("--actor", default=os.getenv("LINEAR_AGENT_OAUTH_ACTOR", ""))
    parser.add_argument("--prompt-consent", action="store_true")
    parser.add_argument("--client-credentials", action="store_true", help="Mint a Linear app-actor token with grant_type=client_credentials instead of opening a browser")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL but do not open a browser")
    args = parser.parse_args(argv)

    env_path = args.env_path or _env_path_for_profile(args.profile)
    auth_path = args.auth_path or _auth_path_for_env_path(env_path)
    env_values = read_env_file(env_path)
    client_id = args.client_id or env_values.get("LINEAR_AGENT_CLIENT_ID", "")
    client_secret = args.client_secret or env_values.get("LINEAR_AGENT_CLIENT_SECRET", "")
    scope = args.scope or env_values.get("LINEAR_AGENT_OAUTH_SCOPES", DEFAULT_SCOPES)
    redirect_uri = args.redirect_uri or env_values.get("LINEAR_AGENT_REDIRECT_URI", DEFAULT_REDIRECT_URI)
    actor = args.actor or env_values.get("LINEAR_AGENT_OAUTH_ACTOR", DEFAULT_ACTOR)
    if not client_id or not client_secret:
        parser.error(
            "--client-id and --client-secret are required, or set "
            "LINEAR_AGENT_CLIENT_ID/SECRET in the environment or profile .env"
        )
    if args.client_credentials:
        result = issue_client_credentials_token(
            client_id=client_id,
            client_secret=client_secret,
            env_path=env_path,
            auth_path=auth_path,
            scope=scope,
        )
        print(f"Saved Linear client-credentials access token to {result['auth_path']}")
        print("Hermes can also reissue this token automatically at runtime from the client ID/secret.")
    else:
        result = run_local_oauth_flow(
            client_id=client_id,
            client_secret=client_secret,
            env_path=env_path,
            auth_path=auth_path,
            redirect_uri=redirect_uri,
            scope=scope,
            actor=actor,
            prompt_consent=args.prompt_consent,
            open_browser=not args.no_browser,
        )
        print(f"Saved Linear OAuth tokens to {result['auth_path']}")
    print("Restart the Hermes gateway for the profile so the new credentials are loaded.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

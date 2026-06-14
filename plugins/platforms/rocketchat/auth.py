"""Auth helpers for the Rocket.Chat Hermes platform plugin."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import tempfile
import threading
import time
import webbrowser
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse

from utils import atomic_replace, is_truthy_value

logger = logging.getLogger(__name__)

try:
    from hermes_constants import get_hermes_home
except (ImportError, ModuleNotFoundError):
    def get_hermes_home() -> Path:
        raw = os.environ.get("HERMES_HOME", "").strip()
        if raw:
            return Path(raw)
        return Path.home() / ".hermes"


DEFAULT_OAUTH_SCOPE = "openid profile email"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8633/rocketchat/callback"
DEFAULT_ARTIFACT_FILENAME = "rocketchat_auth.json"


class RocketChatBootstrapError(RuntimeError):
    """Raised when the Rocket.Chat bootstrap flow fails."""


@dataclass
class RocketChatRuntimeCredentials:
    """Persisted headless runtime credentials for Rocket.Chat."""

    url: str
    user_id: str
    auth_token: str
    auth_type: str = "auth_token"
    username: Optional[str] = None
    pat_name: Optional[str] = None
    created_at: Optional[float] = None
    expires_at: Optional[float] = None

    def to_artifact(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["version"] = 1
        return payload


@dataclass
class OAuthBootstrapConfig:
    """Configuration for the browser-assisted OAuth bootstrap flow."""

    rocketchat_url: str
    service_name: str
    authorize_url: str
    token_url: str
    client_id: str
    client_secret: str = ""
    redirect_uri: str = DEFAULT_REDIRECT_URI
    scope: str = DEFAULT_OAUTH_SCOPE
    pat_name: str = "hermes-agent"
    artifact_path: Optional[Path] = None
    timeout_seconds: float = 300.0
    extra_authorize_params: Optional[dict[str, str]] = None
    two_factor_code: str = ""
    two_factor_method: str = ""
    bypass_two_factor: bool = True
    open_browser: bool = True


def default_artifact_path() -> Path:
    """Default location for the persisted Rocket.Chat bootstrap artifact."""
    return get_hermes_home() / DEFAULT_ARTIFACT_FILENAME


def bootstrap_enabled(extra: Optional[dict[str, Any]] = None) -> bool:
    """Return True when bootstrap artifact auth is explicitly enabled."""
    extra = extra or {}
    if "bootstrap_enabled" in extra:
        return is_truthy_value(extra.get("bootstrap_enabled"), default=False)
    return is_truthy_value(os.getenv("ROCKETCHAT_BOOTSTRAP_ENABLED", ""), default=False)


def artifact_path(extra: Optional[dict[str, Any]] = None) -> Path:
    """Resolve the configured bootstrap artifact path."""
    extra = extra or {}
    raw = (
        extra.get("bootstrap_artifact")
        or os.getenv("ROCKETCHAT_BOOTSTRAP_ARTIFACT", "")
    ).strip()
    if raw:
        return Path(raw).expanduser()
    return default_artifact_path()


def validate_bootstrap_artifact(
    payload: dict[str, Any],
    *,
    expected_url: Optional[str] = None,
) -> RocketChatRuntimeCredentials:
    """Validate and normalize a bootstrap artifact payload."""
    if not isinstance(payload, dict):
        raise RocketChatBootstrapError("Rocket.Chat bootstrap artifact is not a JSON object")

    url = str(payload.get("url") or "").rstrip("/")
    user_id = str(payload.get("user_id") or "").strip()
    auth_token = str(payload.get("auth_token") or "").strip()
    if not url or not user_id or not auth_token:
        raise RocketChatBootstrapError(
            "Rocket.Chat bootstrap artifact is missing one of: url, user_id, auth_token"
        )
    if expected_url and expected_url.rstrip("/") != url:
        raise RocketChatBootstrapError(
            f"Rocket.Chat bootstrap artifact URL {url!r} does not match configured ROCKETCHAT_URL {expected_url!r}"
        )

    return RocketChatRuntimeCredentials(
        url=url,
        user_id=user_id,
        auth_token=auth_token,
        auth_type=str(payload.get("auth_type") or "auth_token"),
        username=str(payload.get("username") or "").strip() or None,
        pat_name=str(payload.get("pat_name") or "").strip() or None,
        created_at=_coerce_optional_float(payload.get("created_at")),
        expires_at=_coerce_optional_float(payload.get("expires_at")),
    )


def load_bootstrap_artifact(
    path: Path,
    *,
    expected_url: Optional[str] = None,
) -> RocketChatRuntimeCredentials:
    """Load persisted runtime credentials from a bootstrap artifact."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RocketChatBootstrapError(
            f"Rocket.Chat bootstrap artifact not found: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RocketChatBootstrapError(
            f"Rocket.Chat bootstrap artifact is not valid JSON: {path}"
        ) from exc
    return validate_bootstrap_artifact(raw, expected_url=expected_url)


def save_bootstrap_artifact(
    path: Path,
    creds: RocketChatRuntimeCredentials,
) -> Path:
    """Persist runtime credentials atomically with owner-only permissions."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(creds.to_artifact(), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o600)
        atomic_replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass
    return path


def resolve_runtime_credentials(
    base_url: str,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> RocketChatRuntimeCredentials:
    """Resolve runtime credentials from env/config, optionally via artifact."""
    extra = extra or {}
    base_url = (base_url or "").rstrip("/")
    if bootstrap_enabled(extra):
        creds = load_bootstrap_artifact(
            artifact_path(extra),
            expected_url=base_url or None,
        )
        if base_url and creds.url.rstrip("/") != base_url:
            raise RocketChatBootstrapError(
                "Rocket.Chat bootstrap artifact URL does not match the configured ROCKETCHAT_URL"
            )
        return creds

    user_id = str(extra.get("user_id") or os.getenv("ROCKETCHAT_USER_ID", "")).strip()
    auth_token = str(extra.get("auth_token") or os.getenv("ROCKETCHAT_AUTH_TOKEN", "")).strip()
    if not base_url or not user_id or not auth_token:
        raise RocketChatBootstrapError(
            "Rocket.Chat runtime auth requires ROCKETCHAT_URL, ROCKETCHAT_USER_ID, and ROCKETCHAT_AUTH_TOKEN "
            "(or enable ROCKETCHAT_BOOTSTRAP_ENABLED with a valid artifact)"
        )
    return RocketChatRuntimeCredentials(
        url=base_url,
        user_id=user_id,
        auth_token=auth_token,
        auth_type="auth_token",
    )


def bootstrap_config_from_env(rocketchat_url: Optional[str] = None) -> OAuthBootstrapConfig:
    """Build an OAuth bootstrap config from the current environment."""
    url = (rocketchat_url or os.getenv("ROCKETCHAT_URL", "")).rstrip("/")
    cfg = OAuthBootstrapConfig(
        rocketchat_url=url,
        service_name=os.getenv("ROCKETCHAT_OAUTH_SERVICE_NAME", "").strip(),
        authorize_url=os.getenv("ROCKETCHAT_OAUTH_AUTHORIZE_URL", "").strip(),
        token_url=os.getenv("ROCKETCHAT_OAUTH_TOKEN_URL", "").strip(),
        client_id=os.getenv("ROCKETCHAT_OAUTH_CLIENT_ID", "").strip(),
        client_secret=os.getenv("ROCKETCHAT_OAUTH_CLIENT_SECRET", "").strip(),
        redirect_uri=os.getenv("ROCKETCHAT_BOOTSTRAP_REDIRECT_URI", DEFAULT_REDIRECT_URI).strip(),
        scope=os.getenv("ROCKETCHAT_OAUTH_SCOPE", DEFAULT_OAUTH_SCOPE).strip() or DEFAULT_OAUTH_SCOPE,
        pat_name=os.getenv("ROCKETCHAT_BOOTSTRAP_PAT_NAME", "hermes-agent").strip() or "hermes-agent",
        artifact_path=artifact_path(),
        timeout_seconds=_coerce_optional_float(os.getenv("ROCKETCHAT_BOOTSTRAP_TIMEOUT_SECONDS")) or 300.0,
        two_factor_code=os.getenv("ROCKETCHAT_BOOTSTRAP_2FA_CODE", "").strip(),
        two_factor_method=os.getenv("ROCKETCHAT_BOOTSTRAP_2FA_METHOD", "").strip(),
        bypass_two_factor=is_truthy_value(os.getenv("ROCKETCHAT_BOOTSTRAP_PAT_BYPASS_2FA", "true"), default=True),
        open_browser=not is_truthy_value(os.getenv("ROCKETCHAT_BOOTSTRAP_NO_BROWSER", ""), default=False),
    )
    raw_extra = os.getenv("ROCKETCHAT_OAUTH_AUTHORIZE_PARAMS", "").strip()
    if raw_extra:
        try:
            parsed = json.loads(raw_extra)
            if isinstance(parsed, dict):
                cfg.extra_authorize_params = {
                    str(key): str(value)
                    for key, value in parsed.items()
                    if value is not None
                }
        except json.JSONDecodeError as exc:
            raise RocketChatBootstrapError(
                "ROCKETCHAT_OAUTH_AUTHORIZE_PARAMS must be valid JSON"
            ) from exc
    return cfg


async def bootstrap_via_oauth(
    config: OAuthBootstrapConfig,
) -> RocketChatRuntimeCredentials:
    """Run a browser-assisted OAuth bootstrap and persist runtime creds."""
    _validate_bootstrap_config(config)

    redirect = urlparse(config.redirect_uri)
    code_verifier = _make_pkce_verifier()
    state = secrets.token_urlsafe(24)
    callback = _CallbackCapture(redirect.path or "/")
    port = redirect.port if redirect.port is not None else 80
    server = _CallbackServer((redirect.hostname or "127.0.0.1", port), callback)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    authorize_url = build_authorize_url(
        config=config,
        state=state,
        code_verifier=code_verifier,
    )
    logger.info("Rocket.Chat bootstrap: opening browser for OAuth authorize URL")
    if config.open_browser:
        webbrowser.open(authorize_url)
    else:
        print(authorize_url)

    try:
        callback_params = callback.wait(timeout=config.timeout_seconds)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    if callback_params.get("state", [""])[0] != state:
        raise RocketChatBootstrapError("Rocket.Chat bootstrap callback state mismatch")
    if "error" in callback_params:
        error = callback_params.get("error", ["unknown_error"])[0]
        raise RocketChatBootstrapError(f"Rocket.Chat bootstrap provider returned error: {error}")
    code = callback_params.get("code", [""])[0]
    if not code:
        raise RocketChatBootstrapError("Rocket.Chat bootstrap callback did not include an authorization code")

    import aiohttp

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        provider_tokens = await _exchange_provider_code(
            session,
            config=config,
            code=code,
            code_verifier=code_verifier,
        )
        runtime = await _exchange_rocketchat_login(
            session,
            config=config,
            provider_tokens=provider_tokens,
        )
        runtime = await _validate_runtime_credentials(session, runtime)
        runtime = await _maybe_promote_to_pat(
            session,
            runtime,
            pat_name=config.pat_name,
            two_factor_code=config.two_factor_code,
            two_factor_method=config.two_factor_method,
            bypass_two_factor=config.bypass_two_factor,
        )

    if config.artifact_path:
        save_bootstrap_artifact(config.artifact_path, runtime)
    return runtime


def build_authorize_url(
    *,
    config: OAuthBootstrapConfig,
    state: str,
    code_verifier: str,
) -> str:
    """Build the upstream provider authorize URL for the bootstrap flow."""
    params = {
        "response_type": "code",
        "client_id": config.client_id,
        "redirect_uri": config.redirect_uri,
        "scope": config.scope,
        "state": state,
        "code_challenge": _pkce_challenge(code_verifier),
        "code_challenge_method": "S256",
    }
    for key, value in (config.extra_authorize_params or {}).items():
        if value:
            params[key] = value
    separator = "&" if "?" in config.authorize_url else "?"
    return f"{config.authorize_url}{separator}{urlencode(params)}"


async def _exchange_provider_code(
    session,
    *,
    config: OAuthBootstrapConfig,
    code: str,
    code_verifier: str,
) -> dict[str, Any]:
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": config.redirect_uri,
        "client_id": config.client_id,
        "code_verifier": code_verifier,
    }
    if config.client_secret:
        form["client_secret"] = config.client_secret

    async with session.post(
        config.token_url,
        data=form,
        headers={"Accept": "application/json"},
    ) as resp:
        payload = await _decode_response_payload(resp)
        if resp.status >= 400:
            raise RocketChatBootstrapError(
                f"Rocket.Chat bootstrap token exchange failed ({resp.status}): {payload}"
            )
    access_token = str(payload.get("access_token") or "").strip()
    if not access_token:
        raise RocketChatBootstrapError(
            "Rocket.Chat bootstrap token exchange did not return an access_token"
        )
    return payload


async def _exchange_rocketchat_login(
    session,
    *,
    config: OAuthBootstrapConfig,
    provider_tokens: dict[str, Any],
) -> RocketChatRuntimeCredentials:
    payload: dict[str, Any] = {
        "serviceName": config.service_name,
        "accessToken": provider_tokens["access_token"],
    }
    expires_in = provider_tokens.get("expires_in")
    if expires_in is not None:
        payload["expiresIn"] = expires_in
    secret = provider_tokens.get("secret") or provider_tokens.get("id_token")
    if secret:
        payload["secret"] = secret

    async with session.post(
        f"{config.rocketchat_url}/api/v1/login",
        json=payload,
        headers={"Accept": "application/json"},
    ) as resp:
        body = await _decode_response_payload(resp)
        if resp.status >= 400:
            raise RocketChatBootstrapError(
                f"Rocket.Chat OAuth login exchange failed ({resp.status}): {body}"
            )

    data = body.get("data") if isinstance(body, dict) else None
    auth_token = str((data or {}).get("authToken") or "").strip()
    user_id = str((data or {}).get("userId") or "").strip()
    if not auth_token or not user_id:
        raise RocketChatBootstrapError(
            "Rocket.Chat OAuth login exchange did not return authToken/userId"
        )
    me = (data or {}).get("me") or {}
    return RocketChatRuntimeCredentials(
        url=config.rocketchat_url,
        user_id=user_id,
        auth_token=auth_token,
        auth_type="auth_token",
        username=str(me.get("username") or "").strip() or None,
        created_at=time.time(),
    )


async def _validate_runtime_credentials(session, creds: RocketChatRuntimeCredentials) -> RocketChatRuntimeCredentials:
    headers = {
        "X-Auth-Token": creds.auth_token,
        "X-User-Id": creds.user_id,
        "Accept": "application/json",
    }
    async with session.get(f"{creds.url}/api/v1/me", headers=headers) as resp:
        body = await _decode_response_payload(resp)
        if resp.status >= 400:
            raise RocketChatBootstrapError(
                f"Rocket.Chat credentials failed validation against /api/v1/me ({resp.status}): {body}"
            )
    if isinstance(body, dict) and body.get("username"):
        creds.username = str(body.get("username") or "").strip() or creds.username
    return creds


async def _maybe_promote_to_pat(
    session,
    creds: RocketChatRuntimeCredentials,
    *,
    pat_name: str,
    two_factor_code: str = "",
    two_factor_method: str = "",
    bypass_two_factor: bool = True,
) -> RocketChatRuntimeCredentials:
    if not pat_name:
        return creds

    headers = {
        "X-Auth-Token": creds.auth_token,
        "X-User-Id": creds.user_id,
        "Accept": "application/json",
    }
    if two_factor_code:
        headers["x-2fa-code"] = two_factor_code
    if two_factor_method:
        headers["x-2fa-method"] = two_factor_method

    body = {
        "tokenName": pat_name,
        "bypassTwoFactor": bool(bypass_two_factor),
    }
    async with session.post(
        f"{creds.url}/api/v1/users.generatePersonalAccessToken",
        json=body,
        headers=headers,
    ) as resp:
        payload = await _decode_response_payload(resp)
        if resp.status >= 400:
            logger.info(
                "Rocket.Chat bootstrap: PAT generation skipped/fell back (%s): %s",
                resp.status,
                payload,
            )
            return creds

    token = str(payload.get("token") or "").strip() if isinstance(payload, dict) else ""
    if not token:
        return creds
    return RocketChatRuntimeCredentials(
        url=creds.url,
        user_id=creds.user_id,
        auth_token=token,
        auth_type="pat",
        username=creds.username,
        pat_name=pat_name,
        created_at=creds.created_at or time.time(),
    )


async def _decode_response_payload(resp) -> Any:
    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "application/json" in content_type:
        return await resp.json()
    return await resp.text()


def _validate_bootstrap_config(config: OAuthBootstrapConfig) -> None:
    missing = []
    for label, value in (
        ("rocketchat_url", config.rocketchat_url),
        ("service_name", config.service_name),
        ("authorize_url", config.authorize_url),
        ("token_url", config.token_url),
        ("client_id", config.client_id),
        ("redirect_uri", config.redirect_uri),
    ):
        if not str(value or "").strip():
            missing.append(label)
    if missing:
        raise RocketChatBootstrapError(
            "Rocket.Chat bootstrap is missing required fields: " + ", ".join(missing)
        )


def _make_pkce_verifier() -> str:
    raw = secrets.token_urlsafe(64)
    return raw[:96]


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _CallbackCapture:
    """Thread-safe holder for the local OAuth callback payload."""

    def __init__(self, expected_path: str) -> None:
        self.expected_path = expected_path or "/"
        self._event = threading.Event()
        self._params: dict[str, list[str]] | None = None

    def set(self, params: dict[str, list[str]]) -> None:
        self._params = params
        self._event.set()

    def wait(self, timeout: float) -> dict[str, list[str]]:
        if not self._event.wait(timeout):
            raise RocketChatBootstrapError(
                "Rocket.Chat bootstrap timed out waiting for the OAuth callback"
            )
        return self._params or {}


class _CallbackServer(ThreadingHTTPServer):
    """Local loopback server used during the OAuth bootstrap flow."""

    allow_reuse_address = True

    def __init__(self, server_address, capture: _CallbackCapture):
        self.capture = capture
        super().__init__(server_address, _CallbackHandler)


class _CallbackHandler(BaseHTTPRequestHandler):
    """Capture a single OAuth redirect and wake the waiting thread."""

    server: _CallbackServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != self.server.capture.expected_path:
            self.send_response(404)
            self.end_headers()
            return
        params = parse_qs(parsed.query)
        self.server.capture.set(params)
        body = (
            "<html><body><h1>Rocket.Chat bootstrap complete</h1>"
            "<p>You can close this window and return to Hermes.</p></body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Browser-assisted OAuth bootstrap for the Hermes Rocket.Chat platform plugin.",
    )
    parser.add_argument("--validate-artifact", action="store_true", help="Validate the configured artifact and exit.")
    parser.add_argument("--artifact", default="", help="Override the bootstrap artifact path.")
    parser.add_argument("--no-browser", action="store_true", help="Print the authorize URL instead of opening a browser.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint used by docs and interactive setup."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.validate_artifact:
        creds = load_bootstrap_artifact(
            Path(args.artifact).expanduser() if args.artifact else artifact_path(),
            expected_url=os.getenv("ROCKETCHAT_URL", "").rstrip("/") or None,
        )
        print(json.dumps(creds.to_artifact(), indent=2))
        return 0

    config = bootstrap_config_from_env()
    if args.artifact:
        config.artifact_path = Path(args.artifact).expanduser()
    if args.no_browser:
        config.open_browser = False

    runtime = asyncio.run(bootstrap_via_oauth(config))
    print(
        json.dumps(
            runtime.to_artifact(),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

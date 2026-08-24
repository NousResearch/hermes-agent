"""Shared constants, the lazy ``httpx`` proxy and :class:`AuthError` for the auth package.

Pure leaf: imports nothing from ``hermes_cli.auth`` so the per-provider modules
(``auth_nous``, ``auth_codex``, ...) can import it at module scope without cycles."""

from __future__ import annotations

import base64
import json
from typing import Any, Callable, Dict, Optional

# httpx is imported lazily (~30ms) because hermes_cli.auth is on the interactive-CLI startup path
# (credential_pool -> auxiliary_client -> cli_commands_mixin). The proxy resolves on first attribute
# access; ``from __future__ import annotations`` keeps ``httpx.Client`` annotations unevaluated.
import importlib as _importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx
else:
    class _LazyHttpx:
        __slots__ = ("_mod",)

        def __init__(self) -> None:
            object.__setattr__(self, "_mod", None)

        def _resolve(self):
            mod = object.__getattribute__(self, "_mod")
            if mod is None:
                mod = _importlib.import_module("httpx")
                object.__setattr__(self, "_mod", mod)
            return mod

        def __getattr__(self, name):
            return getattr(self._resolve(), name)

        # set/del forward to the real module so monkeypatch.setattr("hermes_cli.auth.httpx.Client")
        # keeps working in tests.
        def __setattr__(self, name, value):
            setattr(self._resolve(), name, value)

        def __delattr__(self, name):
            delattr(self._resolve(), name)

    httpx = _LazyHttpx()

# ── Constants ───────────────────────────────────────────────────────────────────────────────────────

AUTH_STORE_VERSION = 1
AUTH_LOCK_TIMEOUT_SECONDS = 15.0

# Nous Portal defaults
DEFAULT_NOUS_PORTAL_URL = "https://portal.nousresearch.com"
DEFAULT_NOUS_INFERENCE_URL = "https://inference-api.nousresearch.com/v1"
DEFAULT_NOUS_CLIENT_ID = "hermes-cli"
NOUS_INFERENCE_INVOKE_SCOPE = "inference:invoke"
NOUS_BILLING_MANAGE_SCOPE = "billing:manage"
DEFAULT_NOUS_SCOPE = NOUS_INFERENCE_INVOKE_SCOPE
NOUS_DEVICE_CODE_SOURCE = "device_code"
NOUS_AUTH_PATH_INVOKE_JWT = "invoke_jwt"
ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120       # refresh 2 min before expiry
NOUS_INVOKE_JWT_MIN_TTL_SECONDS = ACCESS_TOKEN_REFRESH_SKEW_SECONDS
DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS = 1     # poll at most every 1s
DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
_FORM_JSON_HEADERS = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_XAI_OAUTH_BASE_URL = "https://api.x.ai/v1"
MINIMAX_OAUTH_CLIENT_ID = "78257093-7e40-4613-99e0-527b14b39113"
MINIMAX_OAUTH_SCOPE = "group_id profile model.completion"
MINIMAX_OAUTH_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:user_code"
MINIMAX_OAUTH_GLOBAL_BASE = "https://api.minimax.io"
MINIMAX_OAUTH_CN_BASE = "https://api.minimaxi.com"
MINIMAX_OAUTH_GLOBAL_INFERENCE = "https://api.minimax.io/anthropic"
MINIMAX_OAUTH_CN_INFERENCE = "https://api.minimaxi.com/anthropic"
MINIMAX_OAUTH_REFRESH_SKEW_SECONDS = 60
DEFAULT_QWEN_BASE_URL = "https://portal.qwen.ai/v1"
DEFAULT_GITHUB_MODELS_BASE_URL = "https://api.githubcopilot.com"
DEFAULT_COPILOT_ACP_BASE_URL = "acp://copilot"
DEFAULT_OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"
DEFAULT_ACTUAL_BASE_URL = "https://api.actual.inc/v1"
DEFAULT_ACTUAL_LOCAL_BASE_URL = "http://127.0.0.1:8080/v1"
STEPFUN_STEP_PLAN_INTL_BASE_URL = "https://api.stepfun.ai/step_plan/v1"
STEPFUN_STEP_PLAN_CN_BASE_URL = "https://api.stepfun.com/step_plan/v1"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
try:  # Version tag for the Codex token-endpoint User-Agent; fall back if unavailable.
    from hermes_cli import __version__ as _HERMES_CLI_VERSION
except Exception:  # pragma: no cover - version import should always succeed
    _HERMES_CLI_VERSION = "unknown"
CODEX_OAUTH_USER_AGENT = f"hermes-cli/{_HERMES_CLI_VERSION}"
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
XAI_OAUTH_ISSUER = "https://auth.x.ai"
XAI_OAUTH_DISCOVERY_URL = f"{XAI_OAUTH_ISSUER}/.well-known/openid-configuration"
XAI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_OAUTH_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_OAUTH_DEVICE_CODE_URL = f"{XAI_OAUTH_ISSUER}/oauth2/device/code"
# xAI/Grok OAuth access tokens are short-lived (~6h). A two-minute refresh window leaves noisy
# credential-expiry gaps for gateway/cron workloads that touch the provider every ~30 min, so refresh
# up to an hour early.
XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 3600
QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
QWEN_OAUTH_TOKEN_URL = "https://chat.qwen.ai/api/v1/oauth2/token"
QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL = "https://accounts.spotify.com"
DEFAULT_SPOTIFY_API_BASE_URL = "https://api.spotify.com/v1"
DEFAULT_SPOTIFY_REDIRECT_URI = "http://127.0.0.1:43827/spotify/callback"
SPOTIFY_DOCS_URL = "https://hermes-agent.nousresearch.com/docs/user-guide/features/spotify"
SPOTIFY_DASHBOARD_URL = "https://developer.spotify.com/dashboard"
SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120

OAUTH_OVER_SSH_DOCS_URL = "https://hermes-agent.nousresearch.com/docs/guides/oauth-over-ssh"
DEFAULT_SPOTIFY_SCOPE = " ".join((
    "user-modify-playback-state", "user-read-playback-state", "user-read-currently-playing",
    "user-read-recently-played", "playlist-read-private", "playlist-read-collaborative",
    "playlist-modify-public", "playlist-modify-private", "user-library-read", "user-library-modify",
))
SERVICE_PROVIDER_NAMES: Dict[str, str] = {"spotify": "Spotify"}

# LM Studio's default no-auth mode still needs *some* non-empty bearer for the API-key code paths to
# treat the provider as configured. Sent only to LM Studio, never to a remote service.
LMSTUDIO_NOAUTH_PLACEHOLDER = "dummy-lm-api-key"
ACTUAL_LOCAL_NOAUTH_PLACEHOLDER = "dummy-actual-local-api-key"

# Upstream rate-limit / usage-quota exhaustion (HTTP 429): transient, re-authenticating cannot resolve
# it, so it must stay distinct from missing/expired-credential errors.
CODEX_RATE_LIMITED_CODE = "codex_rate_limited"


class AuthError(RuntimeError):
    """Structured auth error with UX mapping hints."""

    def __init__(
        self, message: str, *, provider: str = "", code: Optional[str] = None, relogin_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.relogin_required = relogin_required


def _provider_error_factory(provider: str) -> Callable[..., AuthError]:
    def factory(message: str, code: Optional[str] = None, *, relogin: bool = False) -> AuthError:
        return AuthError(message, provider=provider, code=code, relogin_required=relogin)

    return factory


# Per-provider AuthError constructors: ``_xai_err(message, code, relogin=True)``.
_nous_err = _provider_error_factory("nous")
_xai_err = _provider_error_factory("xai-oauth")
_codex_err = _provider_error_factory("openai-codex")
_spotify_err = _provider_error_factory("spotify")
_qwen_err = _provider_error_factory("qwen-oauth")
_minimax_err = _provider_error_factory("minimax-oauth")


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

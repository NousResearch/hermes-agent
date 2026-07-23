"""Vertex AI (Google Cloud) adapter for Hermes Agent.

Provides authentication and configuration for Vertex AI's OpenAI-compatible
endpoint. This allows Hermes to use Gemini models via Google Cloud with
enterprise-grade rate limits and quotas.

Requires: pip install google-auth

Environment variables honored (all optional):
  GOOGLE_APPLICATION_CREDENTIALS — path to a service account JSON file (secret).
  VERTEX_CREDENTIALS_PATH        — alias, takes precedence if set (secret).
  VERTEX_PROJECT_ID              — override the project_id embedded in creds.
  VERTEX_REGION                  — override default region ("global" unless set).

Non-secret routing settings (project_id, region) also live in config.yaml
under the ``vertex:`` section; env vars take precedence over config.yaml.
"""

import logging
import os
import threading
import time
from typing import Optional, Tuple

from agent.secret_scope import get_secret as _get_secret, is_multiplex_active

# Ensure google-auth is installed before importing. The [vertex] extra is no
# longer in [all] per the lazy-install policy added 2026-05-12 — lazy_deps
# handles on-demand installation so the Vertex provider still works for users
# who installed plain `hermes-agent` and only later selected a Gemini model.
try:
    from tools.lazy_deps import ensure as _lazy_ensure
    _lazy_ensure("provider.vertex", prompt=False)
except Exception:
    pass  # lazy_deps unavailable or install failed — fall through to the real ImportError below

try:
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
except ImportError:
    google = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_REGION = "global"

_creds_cache: dict = {}


def _vertex_config() -> dict:
    """Return the ``vertex:`` section of config.yaml, or {} on any failure.

    Non-secret routing settings (project_id, region) live in config.yaml per
    the .env-secrets-only rule. Env vars still take precedence — they are read
    directly at the call sites below, with config.yaml as the fallback.
    """
    try:
        from hermes_cli.config import load_config

        section = load_config().get("vertex")
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _resolve_region(explicit: Optional[str] = None) -> str:
    """Region precedence: explicit arg > VERTEX_REGION env > config.yaml > default."""
    if explicit:
        return explicit
    env_region = (_get_secret("VERTEX_REGION") or "").strip()
    if env_region:
        return env_region
    cfg_region = str(_vertex_config().get("region") or "").strip()
    return cfg_region or DEFAULT_REGION


def _resolve_project_override() -> Optional[str]:
    """Project-ID override precedence: VERTEX_PROJECT_ID env > config.yaml.

    Returns None when neither is set (the credentials' embedded project_id
    is used in that case).
    """
    env_project = (_get_secret("VERTEX_PROJECT_ID") or "").strip()
    if env_project:
        return env_project
    cfg_project = str(_vertex_config().get("project_id") or "").strip()
    return cfg_project or None


def _resolve_credentials_path(explicit: Optional[str]) -> Optional[str]:
    if explicit and os.path.exists(explicit):
        return explicit
    # Routed through get_secret (not a raw os.environ read): in a multiplex
    # gateway serving several profiles from one process, os.environ reflects
    # whichever profile's .env happened to be loaded at boot, not the profile
    # the current turn belongs to. Reading it directly here would let one
    # profile mint Vertex tokens from — and get billed against — a different
    # profile's service-account file. See agent/secret_scope.py.
    for env_var in ("VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"):
        path = _get_secret(env_var)
        if path and os.path.exists(path):
            return path
    return None


def _refresh_credentials(creds) -> None:
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)


def get_vertex_credentials(credentials_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Return a (fresh access_token, project_id) pair or (None, None) on failure.

    Caches the underlying Credentials object and refreshes it when within
    5 minutes of expiry, so repeated calls don't thrash the token endpoint.
    """
    if google is None:
        logger.warning("google-auth package not installed. Cannot use Vertex AI.")
        return None, None

    resolved_path = _resolve_credentials_path(credentials_path)
    cache_key = resolved_path or "__adc__"

    try:
        cached = _creds_cache.get(cache_key)
        if cached is None:
            if resolved_path:
                creds = service_account.Credentials.from_service_account_file(
                    resolved_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                project_id = creds.project_id
            else:
                # google.auth.default() reads GOOGLE_APPLICATION_CREDENTIALS
                # straight from os.environ internally — it has no notion of
                # the profile secret scope. _resolve_credentials_path already
                # confirmed (via get_secret) that *this* profile doesn't
                # define the var, but python-dotenv's load_dotenv() mutates
                # os.environ at boot for whichever profile happened to load
                # first, so a raw os.environ read here can still pick up a
                # different profile's service-account path. Refuse rather
                # than silently authenticating under a stranger's identity.
                if is_multiplex_active() and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                    logger.warning(
                        "Vertex ADC skipped for this profile: "
                        "GOOGLE_APPLICATION_CREDENTIALS is set in the process "
                        "environment (from another profile's .env) but not in "
                        "this profile's own config. Set VERTEX_CREDENTIALS_PATH "
                        "in this profile's .env instead of relying on ADC."
                    )
                    return None, None
                creds, project_id = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            _creds_cache[cache_key] = (creds, project_id)
        else:
            creds, project_id = cached

        needs_refresh = (
            not getattr(creds, "token", None)
            or getattr(creds, "expired", False)
            or (
                getattr(creds, "expiry", None) is not None
                and (creds.expiry.timestamp() - time.time()) < 300
            )
        )
        if needs_refresh:
            _refresh_credentials(creds)

        override_project = _resolve_project_override()
        if override_project:
            project_id = override_project

        return creds.token, project_id
    except Exception as e:
        logger.error(f"Failed to resolve Vertex AI credentials: {e}")
        _creds_cache.pop(cache_key, None)

        # If ADC failed (e.g. expired refresh token), try the SA file
        # before giving up — it may have been added after initial startup.
        if cache_key == "__adc__":
            sa_path = _resolve_credentials_path(credentials_path)
            if sa_path:
                logger.info("ADC failed, retrying with service account: %s", sa_path)
                return get_vertex_credentials(sa_path)

        return None, None


_refresh_lock = threading.Lock()
_last_forced_refresh: dict = {"ts": 0.0, "ok": False}
_FORCED_REFRESH_COOLDOWN_S = 30.0


def refresh_vertex_credentials(credentials_path: Optional[str] = None) -> bool:
    """Force a token re-mint, bypassing the cached Credentials object.

    Used by the auxiliary 401-recovery path (:func:`agent.auxiliary_client.
    _refresh_provider_credentials`). An auxiliary OpenAI client for the
    Gemini/openapi endpoint carries a FROZEN bearer token baked in at
    client-build time; once that token passes the ~1h lifetime the endpoint
    returns 401 UNAUTHENTICATED (ACCESS_TOKEN_TYPE_UNSUPPORTED) on every
    call. Clearing the credentials cache before re-minting guarantees the
    next client build gets a genuinely fresh token even when the cached
    object's local expiry check disagrees with the server (clock skew,
    wedged google-auth Credentials state).

    Thread-safe with a single-flight cooldown: when the token expires,
    several auxiliary tasks (compression, title_generation,
    background_review) tend to 401 in the same second — only the first
    caller mints; the rest reuse its outcome for
    ``_FORCED_REFRESH_COOLDOWN_S`` seconds. The cooldown also stops a
    non-expiry 401 (wrong credential *type*, where re-minting never helps)
    from hammering the token endpoint.

    Returns True when a fresh (token, project_id) pair was minted.
    """
    with _refresh_lock:
        now = time.time()
        if now - _last_forced_refresh["ts"] < _FORCED_REFRESH_COOLDOWN_S:
            return bool(_last_forced_refresh["ok"])
        # Capture the (possibly wedged) old token so we can detect a no-op
        # re-mint — same token back means recovery will keep failing.
        resolved_path = _resolve_credentials_path(credentials_path)
        cache_key = resolved_path or "__adc__"
        old_entry = _creds_cache.get(cache_key)
        old_token = getattr(old_entry[0], "token", None) if old_entry else None
        # Clear ALL cached credentials (not just this key): a task-level
        # credentials_path override caches under a different key, and a
        # stale sibling entry would hand the dead token to the next client
        # build. Objects already handed to the AnthropicVertex SDK keep
        # their own reference and self-refresh, so this is safe.
        _creds_cache.clear()
        token, project_id = get_vertex_credentials(credentials_path)
        ok = bool(token and project_id)
        if ok and old_token and token == old_token:
            logger.warning(
                "Vertex forced token refresh returned the SAME access token — "
                "the credential source may be wedged (revoked/cached upstream); "
                "the retry will likely 401 again."
            )
        _last_forced_refresh["ts"] = now
        _last_forced_refresh["ok"] = ok
        return ok


def build_vertex_base_url(project_id: str, region: str = DEFAULT_REGION) -> str:
    """Build the OpenAI-compatible base URL for Vertex AI.

    The `global` location uses a bare `aiplatform.googleapis.com` hostname,
    while regional locations use `{region}-aiplatform.googleapis.com`.
    Gemini 3.x preview models are only served via the global endpoint at
    the time of writing.
    """
    host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
    return f"https://{host}/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi"


def get_vertex_config(
    credentials_path: Optional[str] = None,
    region: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve (access_token, base_url) for Vertex AI, or (None, None) on failure."""
    token, project_id = get_vertex_credentials(credentials_path)
    if not token or not project_id:
        return None, None

    effective_region = _resolve_region(region)
    base_url = build_vertex_base_url(project_id, effective_region)
    return token, base_url


def has_vertex_credentials() -> bool:
    """Fast check for whether Vertex credentials appear configured.

    No network calls and no google-auth import — safe for provider
    auto-detection and setup-status display. True when either a service
    account JSON path is resolvable, or an explicit project ID is configured
    (env or config.yaml, implying ADC is intended).
    """
    if _resolve_credentials_path(None):
        return True
    if _resolve_project_override():
        return True
    return False


def is_anthropic_vertex_model(model_id: str) -> bool:
    """Return True if the model is an Anthropic Claude model on Vertex AI.

    Claude-on-Vertex speaks the Anthropic Messages protocol (rawPredict /
    streamRawPredict) — NOT the OpenAI-compatible endpoint that Gemini and
    partner MaaS models use. It must be routed through the AnthropicVertex
    SDK for full feature parity (prompt caching, thinking budgets,
    fine-grained tool streaming), so it needs its own detection.

    Matches Vertex Claude model IDs such as:
      - ``claude-sonnet-4-5@20250929``
      - ``claude-opus-4-1@20250805``
      - ``claude-3-5-sonnet-v2@20241022``
    and the OpenRouter-style ``anthropic/claude-*`` alias.
    """
    lower = (model_id or "").strip().lower()
    if lower.startswith("anthropic/"):
        lower = lower[len("anthropic/"):]
    return lower.startswith("claude")


def get_vertex_anthropic_config(
    credentials_path: Optional[str] = None,
    region: Optional[str] = None,
) -> Tuple[Optional[object], Optional[str], Optional[str]]:
    """Resolve (google-auth Credentials, project_id, region) for Claude-on-Vertex.

    Unlike ``get_vertex_config`` — which mints a short-lived bearer token for
    the OpenAI-compatible Gemini endpoint — the Anthropic SDK's
    ``AnthropicVertex`` client takes the google-auth Credentials object
    directly and refreshes the OAuth2 access token itself on expiry. Handing
    over the Credentials object (rather than a frozen token) means long-lived
    gateway sessions don't 401 after the ~1-hour token lifetime, with no
    per-turn refresh hack needed on Hermes's side.

    Returns ``(None, None, None)`` on any failure (missing google-auth,
    unresolvable credentials, no project ID).
    """
    if google is None:
        logger.warning(
            "google-auth package not installed. Cannot use Claude on Vertex AI. "
            "Install with: pip install 'hermes-agent[vertex]'."
        )
        return None, None, None

    # Reuse get_vertex_credentials so the Credentials object is built, cached,
    # and validated (it also refreshes when near expiry). We only need the
    # side effect of populating _creds_cache plus the resolved project_id.
    token, project_id = get_vertex_credentials(credentials_path)
    if not token or not project_id:
        return None, None, None

    resolved_path = _resolve_credentials_path(credentials_path)
    cache_key = resolved_path or "__adc__"
    cached = _creds_cache.get(cache_key)
    creds = cached[0] if cached else None
    if creds is None:
        return None, None, None

    effective_region = _resolve_region(region)
    return creds, project_id, effective_region

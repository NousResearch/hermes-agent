"""Vertex AI (Google Cloud) adapter for Hermes Agent.

Provides authentication and configuration for Vertex AI's OpenAI-compatible
endpoint. This allows Hermes to use Gemini models via Google Cloud with
enterprise-grade rate limits and quotas.

Supports two authentication methods:

1.  **API Key (Express Mode) — RECOMMENDED.**
    Set ``GOOGLE_VERTEX_API_KEY`` in your .env file. This is a static key that
    works with Vertex's Express Mode endpoint. No OAuth, no service-account
    JSON, no ADC setup. You also need:

      - ``GOOGLE_VERTEX_PROJECT`` — your GCP project ID.
      - ``GOOGLE_VERTEX_LOCATION`` — region (default: ``us-central1``).

    When using API key auth, requests use the ``x-goog-api-key`` header and route
    to the global ``aiplatform.googleapis.com`` host via ``GeminiNativeClient``.

2.  **OAuth2 / ADC (legacy)**
    Requires ``pip install google-auth`` and one of:

      - ``GOOGLE_APPLICATION_CREDENTIALS`` — path to a service account JSON.
      - ``VERTEX_CREDENTIALS_PATH`` — alias, takes precedence if set.
      - ``gcloud auth application-default login`` — local ADC.

    Additional routing settings (non-secret) live in ``config.yaml`` under
    the ``vertex:`` section — project_id and region.

Auth selection is automatic: if ``GOOGLE_VERTEX_API_KEY`` is set, the API key
path is used. Otherwise the adapter falls back to OAuth2 / ADC.

API key env vars (all optional):
  GOOGLE_VERTEX_API_KEY       — Vertex AI API key for Express Mode (secret).
  GOOGLE_VERTEX_PROJECT       — GCP project ID (non-secret routing config; shown
                                on the Keys tab, used for regional endpoints).
  GOOGLE_VERTEX_LOCATION      — Vertex region (default: us-central1).

OAuth2 / ADC env vars (all optional):
  GOOGLE_APPLICATION_CREDENTIALS — path to a service account JSON file (secret).
  VERTEX_CREDENTIALS_PATH        — alias, takes precedence if set (secret).
  VERTEX_PROJECT_ID              — override the project_id embedded in creds.
  VERTEX_REGION                  — override default region ("us-central1").
"""

import logging
import os
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

# Environment variable constants for API key auth (Express Mode)
GOOGLE_VERTEX_API_KEY = "GOOGLE_VERTEX_API_KEY"
GOOGLE_VERTEX_PROJECT = "GOOGLE_VERTEX_PROJECT"
GOOGLE_VERTEX_LOCATION = "GOOGLE_VERTEX_LOCATION"

# Default region — global is required for Gemini 3.x preview models via Vertex OAuth2/ADC.
# Regional overrides (e.g. us-central1) can be set via GOOGLE_VERTEX_LOCATION or VERTEX_REGION.
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
    """Region precedence: explicit arg > GOOGLE_VERTEX_LOCATION env > VERTEX_REGION env > config.yaml > default."""
    if explicit:
        return explicit
    # Check GOOGLE_VERTEX_LOCATION first (API key / Express Mode preferred env var)
    gv_location = (_get_secret(GOOGLE_VERTEX_LOCATION) or "").strip()
    if gv_location:
        return gv_location
    env_region = (_get_secret("VERTEX_REGION") or "").strip()
    if env_region:
        return env_region
    cfg_region = str(_vertex_config().get("region") or "").strip()
    return cfg_region or DEFAULT_REGION


def _resolve_project_override() -> Optional[str]:
    """Project-ID override precedence: GOOGLE_VERTEX_PROJECT env > VERTEX_PROJECT_ID env > config.yaml.

    Returns None when neither is set (the credentials' embedded project_id
    is used in that case for OAuth2; for API key mode the caller should
    prompt for a project if this returns None).
    """
    gv_project = (_get_secret(GOOGLE_VERTEX_PROJECT) or "").strip()
    if gv_project:
        return gv_project
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


def has_vertex_api_key() -> bool:
    """Check whether a Vertex AI API key is configured."""
    return bool(_get_secret(GOOGLE_VERTEX_API_KEY))


def resolve_vertex_api_key() -> Optional[str]:
    """Return the configured Vertex AI API key, or None."""
    return _get_secret(GOOGLE_VERTEX_API_KEY)


def build_vertex_api_key_base_url(project_id: str, region: str) -> str:
    """Build the base URL for Vertex AI Express Mode native API.

    Express Mode uses the native ``generateContent`` API (not the
    OpenAI-compatible endpoint). The URL points at the global
    ``aiplatform.googleapis.com`` host with the ``publishers/google``
    prefix so that the ``GeminiNativeClient`` constructs the correct
    ``:generateContent`` endpoint path.

    The project and region are embedded in the API key itself — they
    are not part of the URL in Express Mode native API calls.
    """
    return "https://aiplatform.googleapis.com/v1/publishers/google"


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
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve (access_token_or_api_key, base_url, auth_header_type) for Vertex AI.

    Two authentication paths, chosen automatically:

    1. **API Key (Express Mode)** — if ``GOOGLE_VERTEX_API_KEY`` is set.
       Returns (api_key, base_url_with_project, ``"x-goog-api-key"``).
       No ``google-auth`` needed. The caller must set the returned header
       name (not ``Authorization: Bearer``) on each request.

    2. **OAuth2 / ADC** — legacy path. Returns (oauth2_token, base_url, ``"Authorization"``).
       Requires ``google-auth`` and valid GCP credentials.

    Returns (None, None, None) when no credentials can be resolved.
    """
    # --- Path 1: API Key (Express Mode) ---
    api_key = resolve_vertex_api_key()
    if api_key:
        project_id = _resolve_project_override() or "default"
        effective_region = _resolve_region(region)
        base_url = build_vertex_api_key_base_url(project_id, effective_region)
        logger.debug(
            "get_vertex_config: using API key (Express Mode) for project %s in %s",
            project_id, effective_region,
        )
        return api_key, base_url, "x-goog-api-key"

    # --- Path 2: OAuth2 / ADC (legacy) ---
    token, project_id = get_vertex_credentials(credentials_path)
    if not token or not project_id:
        return None, None, None

    effective_region = _resolve_region(region)
    base_url = build_vertex_base_url(project_id, effective_region)
    return token, base_url, "Authorization"


def has_vertex_credentials() -> bool:
    """Fast check for whether Vertex credentials appear configured.

    Returns True when either:
    - A Vertex API key (GOOGLE_VERTEX_API_KEY) is set, OR
    - A service account JSON path is resolvable, OR
    - An explicit project ID is configured (ADC intended).

    No network calls and no ``google-auth`` import — safe for provider
    auto-detection and setup-status display.
    """
    if has_vertex_api_key():
        return True
    if _resolve_credentials_path(None):
        return True
    if _resolve_project_override():
        return True
    return False


# ── Model Discovery ──────────────────────────────────────────────────────────


def discover_vertex_models(
    api_key: str,
    project_id: str,
    region: str = DEFAULT_REGION,
    timeout: float = 10.0,
    auth_header: str = "x-goog-api-key",
) -> list[str]:
    """Query Vertex AI's publisher ``models`` endpoint for models available
    in the given project and region.

    ``auth_header`` must match the credential type: ``"x-goog-api-key"`` for
    Express Mode API keys and ``"Authorization"`` for OAuth2 / ADC tokens —
    pass the third element of the ``get_vertex_config()`` tuple. The header
    is selected explicitly rather than sniffed from the key prefix, because
    key formats (``AQ.``, ``AIza``, ``ya29.``, JWTs) are not a reliable
    discriminator.

    Note: Google's ``publishers.google/models`` list endpoint is not part of
    the public Express Mode API surface and 404s in practice, so this
    typically returns an empty list and callers fall back to the curated
    model catalog.

    Returns the sorted model list on success; an empty list on any error
    (network, auth, parse), so callers can fall back to the curated catalog.
    """
    import json
    import urllib.error
    import urllib.request

    host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
    url = (
        f"https://{host}/v1/projects/{project_id}/locations/{region}"
        "/publishers/google/models"
    )
    # Explicit auth mode (from get_vertex_config's 3rd tuple element):
    # x-goog-api-key for Express Mode API keys, Bearer for OAuth2/ADC tokens.
    if auth_header == "x-goog-api-key":
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }
    else:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())

        models: list[str] = []
        for entry in data.get("models", []):
            methods = entry.get("supportedGenerationMethods", [])
            if "generateContent" in methods:
                # The ``name`` field is a full resource path:
                #   projects/{project}/locations/{region}/publishers/google/models/{model_id}
                model_id = entry["name"].rsplit("/", 1)[-1]
                models.append(model_id)

        return sorted(set(models))

    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as exc:
        logger.debug("discover_vertex_models: model list endpoint unavailable (%s) — falling back to curated list", exc)
        return []
    except Exception as exc:
        logger.debug("discover_vertex_models: unexpected error (%s) — falling back to curated list", exc)
        return []

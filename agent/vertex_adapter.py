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
import time
from typing import Optional, Tuple

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
    env_region = os.environ.get("VERTEX_REGION", "").strip()
    if env_region:
        return env_region
    cfg_region = str(_vertex_config().get("region") or "").strip()
    return cfg_region or DEFAULT_REGION


def _resolve_project_override() -> Optional[str]:
    """Project-ID override precedence: VERTEX_PROJECT_ID env > config.yaml.

    Returns None when neither is set (the credentials' embedded project_id
    is used in that case).
    """
    env_project = os.environ.get("VERTEX_PROJECT_ID", "").strip()
    if env_project:
        return env_project
    cfg_project = str(_vertex_config().get("project_id") or "").strip()
    return cfg_project or None


def _resolve_credentials_path(explicit: Optional[str]) -> Optional[str]:
    if explicit and os.path.exists(explicit):
        return explicit
    for env_var in ("VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"):
        path = os.environ.get(env_var)
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

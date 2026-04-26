"""Vertex AI (Google Cloud) adapter for Hermes CLI.

Provides authentication and configuration for Vertex AI's OpenAI-compatible
endpoint, allowing Hermes to use Gemini models via Google Cloud with
enterprise-grade rate limits and quotas.

Requires: pip install google-auth

Environment variables honored (all optional — ADC is used as a fallback):
  VERTEX_CREDENTIALS_PATH        — path to a service account JSON file (takes precedence).
  GOOGLE_APPLICATION_CREDENTIALS — standard GCP credential path.
  VERTEX_PROJECT_ID              — override the project_id embedded in creds.
  VERTEX_REGION / VERTEX_LOCATION — override default region ("us-central1" unless set).
"""

import logging
import os
import time
from typing import Optional, Tuple

try:
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
except ImportError:
    google = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_REGION = "us-central1"
_creds_cache: dict = {}


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
    Supports both service account files and Application Default Credentials.
    """
    if google is None:
        logger.warning(
            "google-auth package not installed. "
            "Install it via: pip install hermes-agent[vertex]"
        )
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

        override_project = os.environ.get("VERTEX_PROJECT_ID")
        if override_project:
            project_id = override_project

        return creds.token, project_id

    except Exception as e:
        logger.error("Failed to resolve Vertex AI credentials: %s", e)
        _creds_cache.pop(cache_key, None)
        # If ADC failed, retry with a service account file that may have been
        # added after initial startup.
        if cache_key == "__adc__":
            sa_path = _resolve_credentials_path(credentials_path)
            if sa_path:
                logger.info("ADC failed, retrying with service account: %s", sa_path)
                return get_vertex_credentials(sa_path)
        return None, None


def build_vertex_base_url(project_id: str, region: str = DEFAULT_REGION) -> str:
    """Build the OpenAI-compatible base URL for Vertex AI."""
    return (
        f"https://{region}-aiplatform.googleapis.com"
        f"/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi"
    )


def get_vertex_config(
    credentials_path: Optional[str] = None,
    region: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve (access_token, base_url) for Vertex AI, or (None, None) on failure."""
    token, project_id = get_vertex_credentials(credentials_path)
    if not token or not project_id:
        return None, None

    effective_region = (
        region
        or os.environ.get("VERTEX_REGION")
        or os.environ.get("VERTEX_LOCATION")
        or DEFAULT_REGION
    )
    base_url = build_vertex_base_url(project_id, effective_region)
    return token, base_url

"""Vertex AI (Google Cloud) adapter for Hermes Agent.

Provides credential resolution and OAuth2 token management for Vertex AI's
OpenAI-compatible endpoint. This allows Hermes to use Gemini models via
Google Cloud with enterprise-grade rate limits and GCP billing credits.

Credential resolution priority:
  1. VERTEX_CREDENTIALS_PATH env var — explicit path to service account JSON
  2. GOOGLE_APPLICATION_CREDENTIALS env var — standard GCP ADC path
  3. gcloud ADC (application default credentials) — auto-detected

Project ID resolution priority:
  1. VERTEX_PROJECT_ID env var
  2. GOOGLE_CLOUD_PROJECT env var
  3. project_id embedded in the service account JSON

Region defaults to "global" (required for Gemini 3.x previews; us-central1
silently breaks them). Override with VERTEX_REGION.

Architecture follows the same pattern as ``agent/bedrock_adapter.py``:
  - All Vertex-specific logic isolated here.
  - Simple detection functions for auth flows (no heavy imports at startup).
  - OAuth2 token caching with refresh-on-expiry.
  - Standard OpenAI-compatible chat/completions — no custom SDK or message
    translation needed.

Requires: ``google-auth`` (optional — only needed when using Vertex provider).
"""

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Cached token with expiry — refreshed lazily by _resolve_vertex_token()
_token_cache: dict = {}


# ---------------------------------------------------------------------------
# Detection functions (fast, no heavy imports, no API calls)
# ---------------------------------------------------------------------------


def _resolve_credentials_path() -> Optional[str]:
    """Return the path to a service account JSON file, or None."""
    for env_var in ("VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"):
        path = os.environ.get(env_var, "").strip()
        if path and os.path.exists(path):
            return path
    return None


def _resolve_project_id_from_sa(path: str) -> Optional[str]:
    """Read project_id from a service account JSON file."""
    try:
        with open(path, "r") as f:
            sa = json.load(f)
        return sa.get("project_id")
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def _resolve_project_id() -> Optional[str]:
    """Resolve the GCP project ID for Vertex AI.

    Priority: VERTEX_PROJECT_ID > GOOGLE_CLOUD_PROJECT > SA JSON project_id.
    """
    for env_var in ("VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"):
        pid = os.environ.get(env_var, "").strip()
        if pid:
            return pid

    creds_path = _resolve_credentials_path()
    if creds_path:
        pid = _resolve_project_id_from_sa(creds_path)
        if pid:
            return pid

    return None


def resolve_vertex_region() -> str:
    """Return the Vertex AI region, defaulting to 'global'."""
    return os.environ.get("VERTEX_REGION", "global").strip() or "global"


def has_vertex_credentials() -> bool:
    """Return True if Vertex AI credentials are configured.

    Checks for a service account JSON at a known path or explicit project ID.
    Does NOT try to load google-auth or make network calls — this is a fast
    startup check used for provider auto-detection.
    """
    if _resolve_credentials_path():
        return True
    if _resolve_project_id():
        return True
    return False


def resolve_vertex_auth_source() -> Optional[str]:
    """Return a human-readable auth source label, or None."""
    if os.environ.get("VERTEX_CREDENTIALS_PATH", "").strip():
        return "VERTEX_CREDENTIALS_PATH"
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip():
        return "GOOGLE_APPLICATION_CREDENTIALS"
    if os.environ.get("VERTEX_PROJECT_ID", "").strip():
        return "VERTEX_PROJECT_ID"
    if os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip():
        return "GOOGLE_CLOUD_PROJECT"
    return None


# ---------------------------------------------------------------------------
# Token management (lazy google-auth import)
# ---------------------------------------------------------------------------


def _resolve_vertex_token() -> Optional[str]:
    """Return a valid OAuth2 access token for Vertex AI.

    Uses service account JSON if available, otherwise falls back to gcloud ADC.
    Tokens are cached and refreshed only when within 60 seconds of expiry.
    """
    now = time.time()
    if _token_cache and _token_cache.get("expires_at", 0) - now > 60:
        return _token_cache["token"]

    try:
        import google.auth
        import google.auth.transport.requests
        from google.oauth2 import service_account
    except ImportError:
        logger.warning(
            "google-auth not installed. Install with: pip install google-auth"
        )
        return None

    credentials_path = _resolve_credentials_path()

    try:
        if credentials_path:
            creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)

        _token_cache["token"] = creds.token
        # 55 min buffer (tokens are valid for 60 min)
        _token_cache["expires_at"] = now + 3300
        return creds.token

    except Exception as e:
        logger.warning("Failed to resolve Vertex AI credentials: %s", e)
        return None


def get_vertex_base_url(
    project_id: Optional[str] = None,
    region: Optional[str] = None,
) -> Optional[str]:
    """Build the Vertex AI OpenAI-compatible base URL.

    Returns None if no project ID can be resolved.
    """
    pid = project_id or _resolve_project_id()
    if not pid:
        return None
    reg = region or resolve_vertex_region()
    return (
        f"https://aiplatform.googleapis.com/v1/projects/{pid}"
        f"/locations/{reg}/endpoints/openapi"
    )

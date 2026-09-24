"""Vertex AI (Google Cloud) auth + base-URL resolution for its OpenAI-compatible endpoint.

Requires ``google-auth`` (lazy-installed). Secrets: GOOGLE_APPLICATION_CREDENTIALS /
VERTEX_CREDENTIALS_PATH (SA JSON path; the latter wins), VERTEX_PROJECT_ID,
VERTEX_REGION. project_id/region may also live in config.yaml ``vertex:``;
env wins over config.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional, Tuple

from agent.secret_scope import get_secret as _get_secret, is_multiplex_active

# The [vertex] extra is not in [all]; install google-auth on demand, else fall through to the ImportError below.
try:
    from tools.lazy_deps import ensure as _lazy_ensure
    _lazy_ensure("provider.vertex", prompt=False)
except Exception:
    pass

try:
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
except ImportError:
    google = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_REGION = "global"
_CLOUD_PLATFORM_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

_creds_cache: dict = {}


def _vertex_config() -> dict:
    """Return the ``vertex:`` section of config.yaml, or {} on any failure."""
    try:
        from hermes_cli.config import load_config

        section = load_config().get("vertex")
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _env_or_config(env_var: str, config_key: str) -> str:
    """Setting precedence: env/secret > config.yaml; "" when neither is set."""
    return (_get_secret(env_var) or "").strip() or str(_vertex_config().get(config_key) or "").strip()


def _resolve_region(explicit: Optional[str] = None) -> str:
    """Region precedence: explicit arg > VERTEX_REGION env > config.yaml > default."""
    return explicit or _env_or_config("VERTEX_REGION", "region") or DEFAULT_REGION


def _resolve_project_override() -> Optional[str]:
    """Project-ID override (VERTEX_PROJECT_ID env > config.yaml), or None to use the creds' embedded project_id."""
    return _env_or_config("VERTEX_PROJECT_ID", "project_id") or None


def _resolve_credentials_path(explicit: Optional[str]) -> Optional[str]:
    if explicit and os.path.exists(explicit):
        return explicit
    # get_secret, not os.environ: under a multiplex gateway os.environ reflects whichever
    # .env loaded at boot, and a raw read could mint (and bill) another profile's SA tokens.
    for env_var in ("VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"):
        path = _get_secret(env_var)
        if path and os.path.exists(path):
            return path
    return None


def _sa_snapshot(resolved_path: Optional[str]) -> Tuple[Optional[bytes], Tuple[Any, ...]]:
    """Resolve (bytes-or-None, cache key) for one credential attempt.

    - No path (ADC): (None, ("__adc__",)) sentinel key.
    - Readable file: (bytes, (path, sha256)).
    - Unreadable file: (None, (path,)) — the caller falls back to the SDK's own file read.

    The key fingerprints file CONTENT, not stat metadata (a metadata-preserving
    atomic replacement can swap the private key under an identical stat signature,
    and this cache guards an identity). Returning the bytes lets the caller build
    credentials from the SAME snapshot the key was computed from (no stat->read TOCTOU).
    """
    if not resolved_path:
        return None, ("__adc__",)
    try:
        with open(resolved_path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return None, (resolved_path,)
    return raw, (resolved_path, hashlib.sha256(raw).hexdigest())


def _load_credentials(resolved_path: Optional[str], sa_raw: Optional[bytes]) -> Optional[Tuple[Any, Optional[str]]]:
    """Build (credentials, project_id) for a cache miss; None when ADC must be refused."""
    if resolved_path:
        if sa_raw is not None:
            creds = service_account.Credentials.from_service_account_info(json.loads(sa_raw), scopes=_CLOUD_PLATFORM_SCOPES)
        else:
            # Unreadable at key time: let the SDK try the file directly.
            creds = service_account.Credentials.from_service_account_file(resolved_path, scopes=_CLOUD_PLATFORM_SCOPES)
        return creds, creds.project_id
    # google.auth.default() reads GOOGLE_APPLICATION_CREDENTIALS from os.environ (set by whichever
    # profile loaded first); this profile doesn't define it, so refuse a stranger's identity.
    if is_multiplex_active() and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.warning(
            "Vertex ADC skipped for this profile: GOOGLE_APPLICATION_CREDENTIALS is set in the process environment "
            "(from another profile's .env) but not in this profile's own config. Set VERTEX_CREDENTIALS_PATH in this "
            "profile's .env instead of relying on ADC."
        )
        return None
    return google.auth.default(scopes=_CLOUD_PLATFORM_SCOPES)


def _needs_refresh(creds) -> bool:
    """No token, expired, or within 5 minutes of expiry."""
    return (
        not getattr(creds, "token", None)
        or getattr(creds, "expired", False)
        or (getattr(creds, "expiry", None) is not None and (creds.expiry.timestamp() - time.time()) < 300)
    )


def get_vertex_credentials(credentials_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Return (fresh access_token, project_id) or (None, None); Credentials cached per file content."""
    if google is None:
        logger.warning("google-auth package not installed. Cannot use Vertex AI.")
        return None, None

    resolved_path = _resolve_credentials_path(credentials_path)
    # One read serves both the cache key and credential construction (creds always match the fingerprint).
    sa_raw, cache_key = _sa_snapshot(resolved_path)

    try:
        cached = _creds_cache.get(cache_key)
        if cached is None:
            cached = _load_credentials(resolved_path, sa_raw)
            if cached is None:
                return None, None
            _creds_cache[cache_key] = cached
            # A rotation leaves the old signature's entry behind; keep at most
            # one Credentials per file so stale identities can't be reused.
            for k in [k for k in _creds_cache if k != cache_key and k[0] == cache_key[0]]:
                _creds_cache.pop(k, None)
        creds, project_id = cached
        if _needs_refresh(creds):
            creds.refresh(google.auth.transport.requests.Request())
        return creds.token, _resolve_project_override() or project_id
    except Exception as e:
        logger.error(f"Failed to resolve Vertex AI credentials: {e}")
        _creds_cache.pop(cache_key, None)
        # If ADC failed (e.g. expired refresh token), try the SA file before giving
        # up — it may have been added after startup. Keyed on this attempt being ADC.
        sa_path = None if resolved_path else _resolve_credentials_path(credentials_path)
        if sa_path:
            logger.info("ADC failed, retrying with service account: %s", sa_path)
            return get_vertex_credentials(sa_path)
        return None, None


def build_vertex_base_url(project_id: str, region: str = DEFAULT_REGION) -> str:
    """OpenAI-compatible Vertex base URL; ``global`` uses the bare host (Gemini 3.x preview is global-only)."""
    host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
    return f"https://{host}/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi"


def get_vertex_config(
    credentials_path: Optional[str] = None, region: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve (access_token, base_url) for Vertex AI, or (None, None) on failure."""
    token, project_id = get_vertex_credentials(credentials_path)
    if not token or not project_id:
        return None, None
    return token, build_vertex_base_url(project_id, _resolve_region(region))


def has_vertex_credentials() -> bool:
    """Fast check (no network): a resolvable SA JSON path, or an explicit project ID (implies ADC)."""
    return bool(_resolve_credentials_path(None) or _resolve_project_override())


def has_explicit_vertex_config() -> bool:
    """True only when the user deliberately pointed Hermes at Vertex.

    Stricter than :func:`has_vertex_credentials`: an ambient ``GOOGLE_APPLICATION_CREDENTIALS``
    must NOT gate the model picker open (unknowing spend). Only Hermes-scoped signals count.
    """
    if _resolve_project_override():
        return True
    sa_path = _get_secret("VERTEX_CREDENTIALS_PATH")
    return bool(sa_path and os.path.isfile(sa_path) and os.access(sa_path, os.R_OK))

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

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional, Tuple

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


def _read_sa_file(resolved_path: str) -> Tuple[bytes, Tuple[Any, ...]]:
    """Read the service-account file once, returning (bytes, cache key).

    The cache key fingerprints the file CONTENT (sha256), not stat
    metadata. A (path, mtime_ns, size) signature — the idiom used for
    config caches — is not sufficient here: metadata-preserving atomic
    replacement (deployment tools that restore mtime; equal-length JSON)
    produces a different private key under an identical stat signature,
    and this cache guards an identity, not a parse (review finding on
    #97701, reproduced: inode/content changed, stat key equal). Reading
    the bytes also lets the caller construct credentials from the SAME
    snapshot the key was computed from, closing the stat->read TOCTOU.

    The file is a few KB of JSON; one read + sha256 per cache PROBE is
    noise next to the OAuth token mint the cache exists to avoid.
    """
    with open(resolved_path, "rb") as fh:
        raw = fh.read()
    digest = hashlib.sha256(raw).hexdigest()
    return raw, (resolved_path, digest)


def _creds_cache_key(resolved_path: Optional[str]) -> Tuple[Any, ...]:
    """Cache key for the Credentials object backing *resolved_path*.

    Content-digest keyed via _read_sa_file (see there for why stat
    signatures are not enough for credential identity). ADC has no file
    to fingerprint; it keeps a plain sentinel key and its existing
    refresh/expiry handling.

    A read failure falls back to the bare path: worst case we serve the
    cached credentials exactly as the pre-signature code did.
    """
    if not resolved_path:
        return ("__adc__",)
    try:
        _, key = _read_sa_file(resolved_path)
        return key
    except OSError:
        return (resolved_path,)


def get_vertex_credentials(credentials_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Return a (fresh access_token, project_id) pair or (None, None) on failure.

    Caches the underlying Credentials object and refreshes it when within
    5 minutes of expiry, so repeated calls don't thrash the token endpoint.
    """
    if google is None:
        logger.warning("google-auth package not installed. Cannot use Vertex AI.")
        return None, None

    resolved_path = _resolve_credentials_path(credentials_path)
    sa_raw: Optional[bytes] = None
    if resolved_path:
        try:
            # One read serves both the cache key and (on a miss) credential
            # construction, so the credentials always match the bytes the
            # key fingerprints — no stat/read or read/read TOCTOU.
            sa_raw, cache_key = _read_sa_file(resolved_path)
        except OSError:
            cache_key = (resolved_path,)
    else:
        cache_key = ("__adc__",)

    try:
        cached = _creds_cache.get(cache_key)
        if cached is None:
            if resolved_path:
                if sa_raw is not None:
                    creds = service_account.Credentials.from_service_account_info(
                        json.loads(sa_raw),
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                else:
                    # Unreadable at key time (bare-path key): let the SDK
                    # try the file directly — pre-signature behavior.
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
            # A rotation leaves the old signature's entry behind; drop any
            # other entries for the same path so the cache holds at most one
            # Credentials per file (bounded, and stale identities don't
            # linger for surprise reuse via an old key).
            for k in [
                k for k in _creds_cache
                if k is not cache_key and k != cache_key and k[0] == cache_key[0]
            ]:
                _creds_cache.pop(k, None)
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
        # Keyed on the RESOLVED PATH being absent (i.e. this attempt was
        # ADC), not on the cache-key literal: the signature-keyed cache
        # made keys tuples, and a tuple never equals the old "__adc__"
        # string (that comparison silently killed this retry path).
        if not resolved_path:
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


def has_explicit_vertex_config() -> bool:
    """True only when the user deliberately pointed Hermes at Vertex.

    Stricter than :func:`has_vertex_credentials`, which also returns True for
    an ambient ``GOOGLE_APPLICATION_CREDENTIALS`` path — a var commonly set
    globally for unrelated GCP work. That ambient signal must NOT mark Vertex
    "explicitly configured" for the model-picker gate, or a user who never set
    Hermes up for Vertex would suddenly see it (and could spend against those
    credentials). So this checks only Hermes-scoped signals:

      * ``VERTEX_PROJECT_ID`` env or ``vertex.project_id`` in config.yaml
        (``_resolve_project_override``), or
      * a resolvable ``VERTEX_CREDENTIALS_PATH`` service-account file
        (the Hermes-specific path var — NOT ``GOOGLE_APPLICATION_CREDENTIALS``).
    """
    if _resolve_project_override():
        return True
    sa_path = _get_secret("VERTEX_CREDENTIALS_PATH")
    if sa_path and os.path.isfile(sa_path) and os.access(sa_path, os.R_OK):
        return True
    return False

"""Vertex AI (Google Cloud) auth + base-URL resolution for its OpenAI-compatible endpoint.

Requires ``google-auth`` (lazy-installed). Secrets: GOOGLE_APPLICATION_CREDENTIALS /
VERTEX_CREDENTIALS_PATH (SA JSON path; the latter wins), VERTEX_PROJECT_ID,
VERTEX_REGION. project_id/region may also live in config.yaml ``vertex:``;
env wins over config.
"""

from datetime import UTC, datetime
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from typing import Any, Optional

from agent.credential_source import VERTEX_OAUTH, tag_credential_source
from agent.secret_scope import get_secret as _get_secret, is_multiplex_active

try:
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
except ImportError:
    google = None  # type: ignore[assignment]


def _ensure_google_auth() -> bool:
    """Bind ``google.auth`` on first use, installing the [vertex] extra through PM if needed.

    The extra left [all] under the lazy-install policy (2026-05-12) so a plain ``hermes-agent``
    install still reaches Vertex after selecting a Gemini model. This runs at the first
    credential request, never at import: an import-time sync would rebuild the dependency
    environment of whatever process happens to import this module.
    """
    global google, service_account
    if google is not None:
        return True
    try:
        from pm import ensure_import
        ensure_import("vertex")
        import google.auth as _auth
        import google.auth.transport.requests
        from google.oauth2 import service_account as _service_account
    except Exception as exc:
        logger.warning("google-auth package not installed (%s). Cannot use Vertex AI.", exc)
        return False
    import google as _google
    google, service_account = _google, _service_account
    return True

logger = logging.getLogger(__name__)

DEFAULT_REGION = "global"
GLOBAL_VERTEX_REGION = "global"
EUROPE_WEST2_VERTEX_REGION = "europe-west2"
EUROPE_WEST2_ARGON_MAX_CONCURRENCY = 5
EUROPE_WEST2_COOLDOWN_SECONDS = 15.0
# carried-fix-anchor: EUROPE_WEST2_STICKY_WAIT_SECONDS = 3.0 (raised to 8.0s)
EUROPE_WEST2_STICKY_WAIT_SECONDS = 8.0
EUROPE_WEST2_DEEP_EPOCH_WAIT_SECONDS = 8.0
GLOBAL_STICKY_EPOCH_MIN_TOKENS = 45_000

_prev_vertex_mod = sys.modules.get(__name__)
_EUROPE_WEST2_ARGON_SEMAPHORE = getattr(
    _prev_vertex_mod, "_EUROPE_WEST2_ARGON_SEMAPHORE", None
) or threading.BoundedSemaphore(EUROPE_WEST2_ARGON_MAX_CONCURRENCY)
_EUROPE_WEST2_STATE_LOCK = getattr(
    _prev_vertex_mod, "_EUROPE_WEST2_STATE_LOCK", None
) or threading.Lock()
_europe_west2_cooldown_until: float = float(
    getattr(_prev_vertex_mod, "_europe_west2_cooldown_until", 0.0) or 0.0
)
_prev_affinity = getattr(_prev_vertex_mod, "_session_region_affinity", None)
_session_region_affinity: dict[str, str] = (
    _prev_affinity if isinstance(_prev_affinity, dict) else {}
)
_SESSION_AFFINITY_CAP = 512
_VERTEX_BASE_URL_RE = re.compile(
    r"^https://(?:[a-z0-9-]+-)?aiplatform\.googleapis\.com/v1(?:beta1)?/projects/([^/]+)/locations/([^/]+)/endpoints/openapi/?$"
)
_CLOUD_PLATFORM_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

_creds_cache: dict = {}


def parse_vertex_base_url(base_url: Optional[str]) -> Optional[tuple[str, str]]:
    """Extract ``(project_id, region)`` from a Vertex OpenAI-compatible base URL, or None."""
    m = _VERTEX_BASE_URL_RE.match(str(base_url or "").strip())
    return (m.group(1), m.group(2)) if m else None


def is_vertex_argon_model(model: Optional[str]) -> bool:
    """Return True strictly for gemini-4-argon (never gemini-3.8-flash or gemini-3*)."""
    cleaned = str(model or "").strip().lower()
    if "/" in cleaned:
        cleaned = cleaned.rsplit("/", 1)[-1]
    if "@" in cleaned:
        cleaned = cleaned.split("@", 1)[0]
    return cleaned == "gemini-4-argon" or cleaned.startswith("gemini-4-argon-")


def is_vertex_global_only_model(model: Optional[str]) -> bool:
    """Models that return HTTP 404 on regional endpoints (e.g. europe-west2) and must use global."""
    cleaned = str(model or "").strip().lower()
    if "/" in cleaned:
        cleaned = cleaned.rsplit("/", 1)[-1]
    if "@" in cleaned:
        cleaned = cleaned.split("@", 1)[0]
    return (
        cleaned.startswith("gemini-3.8-flash")
        or cleaned.startswith("gemini-3-")
        or cleaned.startswith("gemini-3.")
    )


def record_vertex_regional_429(
    cooldown_seconds: float = EUROPE_WEST2_COOLDOWN_SECONDS,
    *,
    session_id: Optional[str] = None,
) -> float:
    """Trip the europe-west2 429 cooldown so subsequent calls route to locations/global."""
    global _europe_west2_cooldown_until
    now = time.monotonic()
    with _EUROPE_WEST2_STATE_LOCK:
        target = now + max(0.0, float(cooldown_seconds))
        _europe_west2_cooldown_until = max(_europe_west2_cooldown_until, target)
        until = _europe_west2_cooldown_until
    raw_sid = str(session_id or "").strip()
    if raw_sid and ":bg-review:" not in raw_sid and ":bg_review" not in raw_sid:
        _record_session_vertex_region_affinity(raw_sid, GLOBAL_VERTEX_REGION)
    return until


def vertex_regional_cooldown_remaining() -> float:
    """Seconds remaining on the europe-west2 429 cooldown (0.0 when ready)."""
    with _EUROPE_WEST2_STATE_LOCK:
        return max(0.0, _europe_west2_cooldown_until - time.monotonic())


def reset_vertex_hybrid_router_state() -> None:
    """Reset cooldown, session affinity, and semaphore state (used by unit tests)."""
    global _europe_west2_cooldown_until, _EUROPE_WEST2_ARGON_SEMAPHORE
    with _EUROPE_WEST2_STATE_LOCK:
        _europe_west2_cooldown_until = 0.0
        _EUROPE_WEST2_ARGON_SEMAPHORE = threading.BoundedSemaphore(EUROPE_WEST2_ARGON_MAX_CONCURRENCY)
        _session_region_affinity.clear()


def _canonical_affinity_session_id(session_id: Optional[str]) -> Optional[str]:
    """Strip ``:bg-review:<seq>`` or ``:bg_review`` suffix so background_review inherits parent session affinity."""
    if not session_id:
        return None
    raw = str(session_id).strip()
    if not raw:
        return None
    base = re.split(r":bg[-_]review(?::|$)", raw, maxsplit=1)[0].strip()
    return base or raw


def reset_vertex_sticky_region(session_id: Optional[str], *, only_if_global: bool = False) -> None:
    """Clear sticky Vertex region for *session_id* (or only if pinned to global on compaction)."""
    sid = _canonical_affinity_session_id(session_id)
    if not sid:
        return
    with _EUROPE_WEST2_STATE_LOCK:
        if only_if_global and _session_region_affinity.get(sid) != GLOBAL_VERTEX_REGION:
            return
        _session_region_affinity.pop(sid, None)


clear_session_vertex_region = reset_vertex_sticky_region
clear_session_vertex_region_affinity = reset_vertex_sticky_region


def get_session_vertex_region_affinity(session_id: Optional[str]) -> Optional[str]:
    """Return the sticky Vertex region recorded for *session_id*, if any."""
    sid = _canonical_affinity_session_id(session_id)
    if not sid:
        return None
    with _EUROPE_WEST2_STATE_LOCK:
        return _session_region_affinity.get(sid)


def _record_session_vertex_region_affinity(session_id: Optional[str], region: str) -> None:
    sid = _canonical_affinity_session_id(session_id)
    if not sid:
        return
    with _EUROPE_WEST2_STATE_LOCK:
        _session_region_affinity.pop(sid, None)
        _session_region_affinity[sid] = region
        while len(_session_region_affinity) > _SESSION_AFFINITY_CAP:
            del _session_region_affinity[next(iter(_session_region_affinity))]


def acquire_vertex_hybrid_region(
    model: Optional[str],
    *,
    configured_region: Optional[str] = None,
    interactive: bool = True,
    session_id: Optional[str] = None,
    sticky_wait_seconds: Optional[float] = None,
    last_prompt_tokens: int = 0,
) -> tuple[str, bool]:
    """Select the Vertex region for a single request and optionally acquire a regional slot.

    Returns ``(region, slot_acquired)``:
    - ``gemini-3.8-flash`` / ``gemini-3*`` -> always ``("global", False)`` (HTTP 404 on europe-west2).
    - ``gemini-4-argon`` when ``interactive=True`` (or ``:bg-review:`` for an ``europe-west2`` parent)
      and configured region is ``global`` or ``europe-west2`` -> tries checkout of
      ``_EUROPE_WEST2_ARGON_SEMAPHORE`` (max 5) when not in a 429 cooldown window. When *session_id*
      is already pinned to ``europe-west2``, waits up to ``sticky_wait_seconds`` (default 8.0s) for a
      regional slot before spilling to ``global``. Once a session runs on ``global``, it stays pinned
      to ``global`` for the rest of the compaction epoch (until ``last_prompt_tokens == -1`` or
      ``reset_vertex_sticky_region`` after compaction) so mid-epoch region jumps never wipe KV caches.
    - All other models/regions -> ``(configured_region or "global", False)``.
    """
    reg = (configured_region or DEFAULT_REGION).strip() or DEFAULT_REGION
    if is_vertex_global_only_model(model):
        return (GLOBAL_VERTEX_REGION, False)
    prompt_tok = int(last_prompt_tokens or 0)
    raw_sid = str(session_id or "")
    pinned_region = get_session_vertex_region_affinity(session_id)
    if prompt_tok == -1:
        if pinned_region == GLOBAL_VERTEX_REGION:
            reset_vertex_sticky_region(session_id)
            pinned_region = None
        prompt_tok = 0
    is_bg_review_for_ew2_parent = (
        pinned_region == EUROPE_WEST2_VERTEX_REGION
        and (bool(raw_sid) or ":bg-review:" in raw_sid)
    )
    if is_vertex_argon_model(model) and reg in (GLOBAL_VERTEX_REGION, EUROPE_WEST2_VERTEX_REGION):
        if (interactive or is_bg_review_for_ew2_parent) and pinned_region != GLOBAL_VERTEX_REGION:
            if vertex_regional_cooldown_remaining() <= 0.0:
                if sticky_wait_seconds is not None:
                    wait_s = max(0.0, float(sticky_wait_seconds))
                elif pinned_region == EUROPE_WEST2_VERTEX_REGION and prompt_tok >= GLOBAL_STICKY_EPOCH_MIN_TOKENS:
                    wait_s = EUROPE_WEST2_DEEP_EPOCH_WAIT_SECONDS
                else:
                    wait_s = EUROPE_WEST2_STICKY_WAIT_SECONDS
                if pinned_region == EUROPE_WEST2_VERTEX_REGION and wait_s > 0.0:
                    acquired = _EUROPE_WEST2_ARGON_SEMAPHORE.acquire(blocking=True, timeout=wait_s)
                else:
                    acquired = _EUROPE_WEST2_ARGON_SEMAPHORE.acquire(blocking=False)
                if acquired:
                    _record_session_vertex_region_affinity(session_id, EUROPE_WEST2_VERTEX_REGION)
                    return (EUROPE_WEST2_VERTEX_REGION, True)
            if session_id and (
                pinned_region is None
                or (interactive and prompt_tok < GLOBAL_STICKY_EPOCH_MIN_TOKENS)
            ):
                _record_session_vertex_region_affinity(session_id, GLOBAL_VERTEX_REGION)
        return (GLOBAL_VERTEX_REGION, False)
    return (reg, False)


def release_vertex_hybrid_slot(client: Any) -> bool:
    """Idempotently release a europe-west2 concurrency slot held on *client*."""
    if client is None or getattr(client, "_vertex_regional_slot_held", False) is not True:
        return False
    try:
        client._vertex_regional_slot_held = False
    except Exception:
        pass
    try:
        _EUROPE_WEST2_ARGON_SEMAPHORE.release()
        return True
    except ValueError:
        return False


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
    if not is_multiplex_active():
        cfg_path = str(_vertex_config().get("credentials_path") or "").strip()
        if cfg_path:
            expanded = os.path.expanduser(cfg_path)
            if os.path.exists(expanded):
                return expanded
    return None


def _sa_snapshot(resolved_path: Optional[str]) -> tuple[Optional[bytes], tuple[Any, ...]]:
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


def _load_credentials(resolved_path: Optional[str], sa_raw: Optional[bytes]) -> Optional[tuple[Any, Optional[str]]]:
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


def _expiry_epoch(expiry: Any) -> Optional[float]:
    """UTC epoch seconds for ``creds.expiry``.

    ``google-auth`` stores ``creds.expiry`` as a naive UTC ``datetime`` (``tzinfo=None``); calling
    ``.timestamp()`` directly on a naive ``datetime`` interprets it in the host's local timezone
    (e.g. CET ``UTC+01:00`` shifts a freshly minted 3600 s token back by 3600 s to ``now - 1 s``,
    making ``_needs_refresh`` True on every call).
    """
    if not isinstance(expiry, datetime):
        return None
    if expiry.tzinfo is None:
        return expiry.replace(tzinfo=UTC).timestamp()
    return expiry.timestamp()


def _needs_refresh(creds) -> bool:
    """No token, expired, or within 5 minutes of expiry."""
    expiry_ts = _expiry_epoch(getattr(creds, "expiry", None))
    return (
        not getattr(creds, "token", None)
        or getattr(creds, "expired", False)
        or (expiry_ts is not None and (expiry_ts - time.time()) < 300)
    )


def _cached_credentials(resolved_path: Optional[str], sa_raw: Optional[bytes], cache_key: tuple[Any, ...]):
    """(credentials, project_id) from the per-content cache, loading on a miss; None when refused."""
    cached = _creds_cache.get(cache_key)
    if cached is None:
        cached = _load_credentials(resolved_path, sa_raw)
        if cached is None:
            return None
        _creds_cache[cache_key] = cached
        # A rotation leaves the old signature's entry behind; keep at most
        # one Credentials per file so stale identities can't be reused.
        for k in [k for k in _creds_cache if k != cache_key and k[0] == cache_key[0]]:
            _creds_cache.pop(k, None)
    return cached


def get_vertex_credentials(
    credentials_path: Optional[str] = None, *, force_refresh: bool = False
) -> tuple[Optional[str], Optional[str]]:
    """Return (fresh access_token, project_id) or (None, None); Credentials cached per file content."""
    if not _ensure_google_auth():
        return None, None

    resolved_path = _resolve_credentials_path(credentials_path)
    # One read serves both the cache key and credential construction (creds always match the fingerprint).
    sa_raw, cache_key = _sa_snapshot(resolved_path)

    try:
        cached = _cached_credentials(resolved_path, sa_raw, cache_key)
        if cached is None:
            return None, None
        creds, project_id = cached
        if force_refresh or _needs_refresh(creds):
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
            return get_vertex_credentials(sa_path, force_refresh=force_refresh)
        return None, None


def build_vertex_base_url(project_id: str, region: str = DEFAULT_REGION) -> str:
    """OpenAI-compatible Vertex base URL; ``global`` uses the bare host (Gemini 3.x preview is global-only)."""
    host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
    return f"https://{host}/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi"


def get_vertex_config(
    credentials_path: Optional[str] = None, region: Optional[str] = None, *, force_refresh: bool = False
) -> tuple[Optional[str], Optional[str]]:
    """Resolve (access_token, base_url) for Vertex AI, or (None, None) on failure."""
    token, project_id = get_vertex_credentials(credentials_path, force_refresh=force_refresh)
    if not token or not project_id:
        return None, None
    return token, build_vertex_base_url(project_id, _resolve_region(region))


def get_vertex_base_url(credentials_path: Optional[str] = None, region: Optional[str] = None) -> Optional[str]:
    """The Vertex base_url WITHOUT minting a token (no OAuth exchange), or None when the credentials
    cannot be loaded or no project resolves. For processes that route but never send — the
    gateway-backed CLI, whose gateway session mints its own token for the turn."""
    if not _ensure_google_auth():
        return None
    resolved_path = _resolve_credentials_path(credentials_path)
    sa_raw, cache_key = _sa_snapshot(resolved_path)
    try:
        cached = _cached_credentials(resolved_path, sa_raw, cache_key)
    except Exception:  # health: allow BLE001 -- same boundary as get_vertex_credentials: unloadable credentials are main's AuthError copy
        logger.debug("Vertex AI credentials could not be loaded", exc_info=True)
        _creds_cache.pop(cache_key, None)
        return None
    project_id = _resolve_project_override() or (cached[1] if cached else None)
    return build_vertex_base_url(project_id, _resolve_region(region)) if cached and project_id else None


def vertex_access_token() -> str:
    """Callable API key: a fresh Vertex token, minted on first use and reused until near expiry.
    One stable object, so a credential re-resolution never reads as a key change."""
    token, _project_id = get_vertex_credentials()
    if not token:
        raise RuntimeError("Vertex AI OAuth token could not be minted (see the Vertex credential settings)")
    return token


tag_credential_source(vertex_access_token, VERTEX_OAUTH)


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

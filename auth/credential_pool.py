"""Persistent multi-credential pool for same-provider failover."""

from __future__ import annotations
import re  # noqa: F401 (pool collaborators consume these bindings)
from datetime import timezone  # noqa: F401 (pool collaborators consume these bindings)
from hermes_constants import OPENROUTER_BASE_URL  # noqa: F401 (pool collaborators consume these bindings)
from agent.secret_scope import get_secret as _get_secret  # noqa: F401 (pool collaborators consume these bindings)
from auth.plugin_hooks import plugin_refresh_hook  # noqa: F401 (pool collaborators consume these bindings)
from auth.credential_pool_plugin import apply_plugin_refresh_result, recover_failed_plugin_refresh  # noqa: F401 (pool collaborators consume these bindings)
from auth.persistence import is_borrowed_credential_source  # noqa: F401 (pool collaborators consume these bindings)
from auth.store import _same_path  # noqa: F401 (pool collaborators consume these bindings)
from auth.provider_state import _load_provider_state, _load_provider_state_with_source, _save_provider_state, _store_provider_state  # noqa: F401 (pool collaborators consume these bindings)

import auth.pool_persistence as auth_pool_persistence
import auth.provider_state as auth_provider_state
import auth.store as auth_storage

from auth.credential_pool_admin import CredentialPoolAdminMixin
from auth.pool_refresh import CredentialPoolRefreshMixin
from auth import pool_sources
from auth.credential_pool_model_cooldowns import (
    CredentialPoolModelCooldownMixin,
    model_cooldown_until,
)

import logging
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from agent.secret_scope import get_secret_str
from agent.retry_utils import reset_delay_from_message
from auth.persistence import (
    fingerprint_secret_value,
    sanitize_borrowed_credential_payload,
)
from auth.pool_environment import PoolEnvironment
from auth.oauth_grants import (
    SINGLE_USE_REFRESH_POOL_PROVIDERS,
    heal_forked_single_use_oauth_grants,
)
from auth.token_validation import _decode_jwt_claims, _nous_invoke_jwt_is_usable
from auth.store import (
    _auth_store_lock,
    _global_auth_file_path,
    _load_auth_store,
    _save_auth_store,
)
from auth.pool_persistence import read_credential_pool, write_credential_pool

logger = logging.getLogger(__name__)


def _load_config_safe(environment: PoolEnvironment) -> Optional[dict]:
    """Load config.yaml read-only, returning None on any error.

    ``load_config_readonly()`` skips the deepcopy ``load_config()`` pays per
    call; the picker calls ``load_pool()`` once per provider row, which made
    that copy the dominant cost of ``model.options``.
    """
    environment.require_current_scope()
    try:
        environment.require_current_scope()
        return environment.read_config()
    except Exception:
        return None


def _is_source_suppressed_fn() -> Callable[[str, str], bool]:
    '``auth.sources.is_source_suppressed`` (late-bound), or an always-False stub.'
    try:
        from auth.sources import is_source_suppressed
        return is_source_suppressed
    except ImportError:
        return lambda _p, _s: False


# --- Status and type constants ---

STATUS_OK = "ok"
STATUS_EXHAUSTED = "exhausted"
# Terminal failure — the credential will never recover on its own (upstream
# ``token_invalidated`` / ``token_revoked``). DEAD entries are excluded from
# rotation unconditionally and only clear when an explicit write-side sync
# (e.g. ``_save_codex_tokens`` after a fresh device-code login) rewrites tokens.
STATUS_DEAD = "dead"

# OAuth error reasons that mean the credential is permanently invalid
# server-side (OpenAI Codex, Anthropic, xAI, Google OAuth, RFC 6749/6750/7009).
_TERMINAL_AUTH_REASONS = frozenset({
    "token_invalidated",
    "token_revoked",
    "invalid_token",
    "invalid_grant",
    "unauthorized_client",
    "refresh_token_reused",  # single-use refresh token consumed by another process
})

# Locally generated terminal reason (no HTTP status): a refresh POST rotated a
# single-use pair but the replacement never reached its authoritative store, so
# the pre-rotation token still on disk is already spent. Kept out of
# _TERMINAL_AUTH_REASONS (upstream 401 reasons) and handled explicitly.
CREDENTIAL_PERSIST_FAILED_REASON = "credential_persist_failed"

# DEAD ``manual:*`` entries are pruned after this quiet window — they have no
# singleton to re-seed from and the user can re-add via ``hermes auth add``.
# Singleton-seeded entries (device_code, claude_code) are NOT pruned because
# ``_seed_from_singletons`` would re-create them from the same stale tokens.
DEAD_MANUAL_PRUNE_TTL_SECONDS = 24 * 60 * 60

AUTH_TYPE_OAUTH = "oauth"
AUTH_TYPE_API_KEY = "api_key"

SOURCE_MANUAL = "manual"
SOURCE_MANUAL_DEVICE_CODE = f"{SOURCE_MANUAL}:device_code"

STRATEGY_FILL_FIRST = "fill_first"
STRATEGY_ROUND_ROBIN = "round_robin"
STRATEGY_RANDOM = "random"
STRATEGY_LEAST_USED = "least_used"
SUPPORTED_POOL_STRATEGIES = {
    STRATEGY_FILL_FIRST,
    STRATEGY_ROUND_ROBIN,
    STRATEGY_RANDOM,
    STRATEGY_LEAST_USED,
}

# Cooldowns before retrying an exhausted credential. Transient 401s cool down
# briefly so single-key setups recover; 429/402/other take an hour.
# Provider-supplied reset_at timestamps override these defaults.
EXHAUSTED_TTL_401_SECONDS = 5 * 60
EXHAUSTED_TTL_429_SECONDS = 60 * 60
EXHAUSTED_TTL_DEFAULT_SECONDS = 60 * 60
# When the offending key is the sole non-DEAD entry, an hour-long bench means
# an hour of hard failures. Throttles (429/403/5xx) reset in seconds, so a sole
# credential cools down briefly instead.
EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS = 60

# ``FailoverReason.billing`` as a bare string: the pool persists classified
# failure semantics to JSON and must not import the classifier.
FAILURE_REASON_BILLING = "billing"

# Billing verdict resting on an ambiguous body (#82154): Anthropic's "out of
# extra usage" 400 is returned both for genuine overage and for a server-side
# content-filter rejection, which leaves the credential healthy. Unverified
# billing gets the short transient cooldown; genuine depletion re-latches.
FAILURE_REASON_BILLING_UNVERIFIED = "billing_unverified"

# Throttle window for the "no available entries" INFO line. Selection runs on
# every model call; on Windows several processes share one rotating log behind
# a cross-process lock, and per-selection logging stormed that lock, pegged a
# core, and stalled the event loop (Desktop backend readiness timeouts).
# Credential selection runs on a hot path (every model call, plus auxiliary tasks like
# compression/moa/titles), so when a pool is empty or fully exhausted the un-throttled log fires on *every*
# selection. On Windows several Hermes processes share one rotating log guarded by concurrent-log-handler's
# cross-process lock; that per-selection volume storms the lock (``RuntimeError: Cannot acquire lock after
# 20 attempts``), pegs a core, and stalls the asyncio event loop long enough to fail the Desktop backend
# readiness handshake ("Timed out connecting to Hermes backend after 15000ms"). Logging the condition at
# most once per window preserves the signal while removing the storm — same class of fix as the warn-once
# dedup in #58265.
NO_AVAILABLE_ENTRIES_LOG_THROTTLE_SECONDS = 60.0

# Pool key prefix for custom OpenAI-compatible endpoints: all share
# provider='custom' but are keyed 'custom:<normalized_name>'.
CUSTOM_POOL_PREFIX = "custom:"

# Fields only round-tripped through JSON — never used for logic as attributes.
_EXTRA_KEYS = frozenset({
    "token_type",
    "scope",
    "client_id",
    "portal_base_url",
    "obtained_at",
    "expires_in",
    "agent_key_id",
    "agent_key_expires_in",
    "agent_key_reused",
    "agent_key_obtained_at",
    "tls",
    "secret_source",
    "secret_fingerprint",
    # Nous guest identity (``auth_method: anonymous``): the anon_ credential is the refresh material.
    "auth_method",
    "account_tier",
    "anon_token",
    "user_id",
    "org_id",
    # Classified failure semantics for the last exhaustion (agent/error_classifier.py).
    # Providers return 403 for both an edge throttle and a spending limit, so the
    # raw status cannot size a cooldown; persisted so a restart doesn't downgrade
    # a billing bench to a 60s transient cooldown.
    "failure_reason",
})

# Nous singleton metadata mirrored between auth.json state and ``entry.extra``.
_NOUS_EXTRA_STATE_KEYS = (
    "obtained_at",
    "expires_in",
    "agent_key_id",
    "agent_key_expires_in",
    "agent_key_reused",
    "agent_key_obtained_at",
    "auth_method",
    "account_tier",
    "anon_token",
    "user_id",
    "org_id",
)

# ``replace(entry, **_CLEAR_STATUS)`` returns an entry with no error state.
_CLEAR_STATUS: Dict[str, Any] = {
    "last_status": None,
    "last_status_at": None,
    "last_error_code": None,
    "last_error_reason": None,
    "last_error_message": None,
    "last_error_reset_at": None,
}
_MARK_OK: Dict[str, Any] = {**_CLEAR_STATUS, "last_status": STATUS_OK}


def _normalize_pool_auth_type(provider: str, token: Any, auth_type: Any) -> str:
    """Infer pool auth metadata for token formats with one unambiguous meaning."""
    if provider == "anthropic" and isinstance(token, str) and token.startswith("sk-ant-oat"):
        return AUTH_TYPE_OAUTH
    return str(auth_type or AUTH_TYPE_API_KEY)


@dataclass
class PooledCredential:
    provider: str
    id: str
    label: str
    auth_type: str
    priority: int
    source: str
    access_token: str
    refresh_token: Optional[str] = None
    last_status: Optional[str] = None
    last_status_at: Optional[float] = None
    last_error_code: Optional[int] = None
    last_error_reason: Optional[str] = None
    last_error_message: Optional[str] = None
    last_error_reset_at: Optional[float] = None
    # Epoch of the last deliberate ``hermes auth reset`` of this entry. Sticky: a later exhaustion
    # stamps a newer ``last_status_at``, so "reset postdates status" stays decidable across processes.
    status_cleared_at: Optional[float] = None
    base_url: Optional[str] = None
    expires_at: Optional[str] = None
    expires_at_ms: Optional[int] = None
    last_refresh: Optional[str] = None
    inference_base_url: Optional[str] = None
    agent_key: Optional[str] = None
    agent_key_expires_at: Optional[str] = None
    request_count: int = 0
    # A provider may rate-limit one model while the same credential remains
    # usable for its sibling models.  Keep that observation separate from the
    # credential-wide status used for auth and billing failures.
    model_cooldowns: Optional[Dict[str, float]] = None
    extra: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
        self.auth_type = _normalize_pool_auth_type(self.provider, self.access_token, self.auth_type)

    def __getattr__(self, name: str):
        if name in _EXTRA_KEYS:
            return self.extra.get(name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

    @classmethod
    def from_dict(cls, provider: str, payload: Dict[str, Any]) -> "PooledCredential":
        field_names = {f.name for f in fields(cls) if f.name != "provider"}
        data = {k: payload.get(k) for k in field_names if k in payload}
        # Rehydrated last_status_at may be an ISO string from to_dict() — normalize to float epoch
        if isinstance(data.get("last_status_at"), str):
            data["last_status_at"] = _parse_absolute_timestamp(data["last_status_at"])
        # Every non-field key rides in ``extra`` (to_dict writes them all back), so metadata a plugin
        # stores on its own rows survives load -> save -> load. ``_EXTRA_KEYS`` stays the attribute
        # surface for core logic; unknown keys are opaque payload. ``provider`` is the row's owner
        # (excluded from ``field_names`` above), never metadata — sweeping it in would write a
        # stray provider name back over the row on to_dict().
        data["extra"] = {
            k: v for k, v in payload.items() if k not in field_names and k != "provider" and v is not None
        }
        data.setdefault("id", uuid.uuid4().hex[:6])
        data.setdefault("label", payload.get("source", provider))
        data.setdefault("auth_type", AUTH_TYPE_API_KEY)
        data.setdefault("priority", 0)
        data.setdefault("source", SOURCE_MANUAL)
        data.setdefault("access_token", "")
        return cls(provider=provider, **data)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for field_def in fields(self):
            if field_def.name in {"provider", "extra"}:
                continue
            value = getattr(self, field_def.name)
            if value is not None or field_def.name in _CLEAR_STATUS:
                result[field_def.name] = value
        for k, v in self.extra.items():
            if v is not None:
                result[k] = v
        return sanitize_borrowed_credential_payload(result, self.provider)

    @property
    def runtime_api_key(self) -> str:
        if self.provider == "nous":
            # Nous stores the runtime inference credential in agent_key for
            # compatibility. It must be a NAS invoke JWT.
            for token, expires_at in (
                (self.agent_key, self.agent_key_expires_at),
                (self.access_token, self.expires_at),
            ):
                if (
                    isinstance(token, str)
                    and token.strip()
                    and _nous_invoke_jwt_is_usable(
                        token, scope=getattr(self, "scope", None), expires_at=expires_at,
                    )
                ):
                    return token.strip()
            return ""
        return str(self.access_token or "")

    @property
    def runtime_base_url(self) -> Optional[str]:
        if self.provider == "nous":
            return self.inference_base_url or self.base_url
        if self.provider == "openai-codex":
            # Pool rows keep the canonical ChatGPT URL; the profile-scoped proxy override must win
            # for every reader of the row — initial resolution AND a 401/429 rotation
            # (client_lifecycle._swap_credential), or a rotation silently leaves the proxy.
            return get_secret_str("HERMES_CODEX_BASE_URL", "").strip().rstrip("/") or self.base_url
        return self.base_url


def label_from_token(token: str, fallback: str) -> str:
    claims = _decode_jwt_claims(token)
    for key in ("email", "preferred_username", "upn"):
        value = claims.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _codex_principal_identity(access_token: Any) -> Optional[Tuple[str, str]]:
    """``(chatgpt_account_id, sub)`` of a Codex access token, or None when either claim is missing.

    Decoded without signature verification: this only decides whether two credentials Hermes
    already holds belong to the same principal, never whether a token is valid. Both claims are
    required because members of one ChatGPT workspace share ``chatgpt_account_id`` yet have their
    own subjects and quotas.
    """
    claims = _decode_jwt_claims(access_token)
    auth_claims = claims.get("https://api.openai.com/auth") if isinstance(claims, dict) else None
    account_id = auth_claims.get("chatgpt_account_id") if isinstance(auth_claims, dict) else None
    subject = claims.get("sub") if isinstance(claims, dict) else None
    if not (isinstance(account_id, str) and account_id.strip() and isinstance(subject, str) and subject.strip()):
        return None
    return account_id.strip(), subject.strip()


def _codex_entry_tracks_singleton(entry: PooledCredential, singleton_tokens: Dict[str, Any]) -> bool:
    """Whether a Codex pool entry may adopt the auth.json singleton's token pair.

    ``device_code`` IS the singleton. ``manual:device_code`` is ambiguous: a legacy alias of the
    singleton (same account, must follow its rotations) or an independent account added with
    ``hermes auth add openai-codex`` (must never be overwritten — adopting turned two logins into
    one account, both hitting the same usage limit). Same principal proves the alias; unknown
    identity fails closed.
    """
    if entry.source == "device_code":
        return True
    if entry.source != SOURCE_MANUAL_DEVICE_CODE:
        return False
    entry_identity = _codex_principal_identity(entry.access_token)
    return entry_identity is not None and entry_identity == _codex_principal_identity(singleton_tokens.get("access_token"))


def _next_priority(entries: List[PooledCredential]) -> int:
    return max((entry.priority for entry in entries), default=-1) + 1


def _is_manual_source(source: str) -> bool:
    normalized = (source or "").strip().lower()
    return normalized == SOURCE_MANUAL or normalized.startswith(f"{SOURCE_MANUAL}:")


def _exhausted_ttl(
    error_code: Optional[int],
    *,
    sole_credential: bool = False,
    failure_reason: Optional[str] = None,
) -> int:
    """Return cooldown seconds based on the HTTP status that caused exhaustion.

    *sole_credential*: the pool has nothing to rotate to, so transient
    throttles (429 and the catch-all default covering 403/5xx/unknown) are
    capped to a brief cooldown; 401 keeps its own already-short TTL.

    *failure_reason* is the classifier verdict: an OpenRouter ``key limit
    exceeded`` and an xAI spending block both arrive as 403 but are billing,
    and a 60s retry on a spent account just re-fails. Billing keeps the full
    bench regardless of status; 402 is billing by definition.
    Unverified billing (#82154) gets the short cooldown regardless of pool
    size (the credential may be healthy), unless the status is a true 402.
    """
    if error_code == 401:
        return EXHAUSTED_TTL_401_SECONDS
    base = EXHAUSTED_TTL_429_SECONDS if error_code == 429 else EXHAUSTED_TTL_DEFAULT_SECONDS
    if failure_reason == FAILURE_REASON_BILLING_UNVERIFIED and error_code != 402:
        return min(base, EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS)
    is_billing = error_code == 402 or failure_reason == FAILURE_REASON_BILLING
    if sole_credential and not is_billing:
        return min(base, EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS)
    return base


def _parse_absolute_timestamp(value: Any) -> Optional[float]:
    """Best-effort parse of epoch seconds / epoch ms / ISO-8601 into epoch seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric <= 0:
            return None
        return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            numeric = float(raw)
            return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _singleton_predates_entry(state: Any, entry: "PooledCredential") -> bool:
    """True only when the auth.json singleton is PROVABLY older than *entry*.

    Both sides stamp ``last_refresh`` on every successful rotation. When
    either side lacks a parseable stamp this returns False (cannot prove),
    which keeps the historical adopt-on-difference behavior (#70111) intact
    for legacy writers.
    """
    entry_ts = _parse_absolute_timestamp(entry.last_refresh)
    if entry_ts is None:
        return False
    state_ts = _parse_absolute_timestamp(state.get("last_refresh") if isinstance(state, dict) else None)
    if state_ts is None:
        return False
    return state_ts < entry_ts


def _normalize_error_context(error_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(error_context, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for key in ("reason", "message"):
        value = error_context.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()
    reset_at = (
        error_context.get("reset_at")
        or error_context.get("resets_at")
        or error_context.get("retry_until")
    )
    parsed_reset_at = _parse_absolute_timestamp(reset_at)
    message = error_context.get("message")
    if parsed_reset_at is None and isinstance(message, str):
        retry_delay_seconds = reset_delay_from_message(message)
        if retry_delay_seconds is not None:
            parsed_reset_at = time.time() + retry_delay_seconds
    if parsed_reset_at is not None:
        normalized["reset_at"] = parsed_reset_at
    return normalized


def _exhausted_until(entry: PooledCredential, *, sole_credential: bool = False) -> Optional[float]:
    if entry.last_status != STATUS_EXHAUSTED:
        return None
    reset_at = _parse_absolute_timestamp(entry.last_error_reset_at)
    if reset_at is not None:
        return reset_at
    if entry.last_status_at:
        return entry.last_status_at + _exhausted_ttl(
            entry.last_error_code,
            sole_credential=sole_credential,
            failure_reason=entry.failure_reason,
        )
    return None


# --- Custom (OpenAI-compatible) endpoint pool keys ---


def _normalize_custom_pool_name(name: str) -> str:
    """Normalize a custom provider name for use as a pool key suffix."""
    return name.strip().lower().replace(" ", "-")


def _norm_url(url: Any) -> str:
    return str(url or "").strip().rstrip("/")


def _iter_custom_providers(
    config: Optional[dict] = None, *, environment: PoolEnvironment
):
    """Yield ``(normalized_name, entry)`` from the merged custom-provider config view."""
    environment.require_current_scope()
    if config is None:
        config = _load_config_safe(environment=environment)
    if config is None:
        return
    try:
        custom_providers = environment.custom_providers(config)
    except Exception:
        return
    for entry in custom_providers or ():
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            yield _normalize_custom_pool_name(entry["name"]), entry


def _custom_entry_name_aliases(norm_name: str, entry: Dict[str, Any]) -> set:
    aliases = {norm_name}
    provider_key = _normalize_custom_pool_name(str(entry.get("provider_key") or ""))
    if provider_key:
        aliases.add(provider_key)
    return aliases


def _requested_custom_name_aliases(provider_name: str) -> set:
    normalized = _normalize_custom_pool_name(provider_name)
    aliases = {normalized} if normalized else set()
    if normalized.startswith(CUSTOM_POOL_PREFIX):
        suffix = _normalize_custom_pool_name(normalized[len(CUSTOM_POOL_PREFIX):])
        if suffix:
            aliases.add(suffix)
    return aliases


def _pool_keys_for_custom_entry(norm_name: str, entry: Dict[str, Any]) -> List[str]:
    """Durable ``providers.<key>`` slug first, then legacy ``custom:<name>``."""
    keys: List[str] = []
    provider_key = _normalize_custom_pool_name(str(entry.get("provider_key") or ""))
    for key in (provider_key, f"{CUSTOM_POOL_PREFIX}{norm_name}" if norm_name else ""):
        normalized = key.strip().lower()
        if normalized and normalized not in keys:
            keys.append(normalized)
    return keys


def custom_provider_pool_key_candidates(
    base_url: Optional[str],
    provider_name: Optional[str] = None,
    *,
    environment: PoolEnvironment,
) -> List[str]:
    """Return pool keys to try for a custom endpoint.

    ``hermes auth add <key>`` stores ``providers.<key>`` credentials under the
    durable config slug; older rows and legacy ``custom_providers:`` entries
    live under ``custom:<display-name>``. Try the slug first, then the legacy
    namespace, so a populated pool is not skipped in favour of the
    ``no-key-required`` placeholder.
    """
    environment.require_current_scope()
    if not base_url:
        return []
    normalized_url = _norm_url(base_url)
    requested_aliases = (
        _requested_custom_name_aliases(provider_name) if provider_name else set()
    )

    if requested_aliases:
        for norm_name, entry in _iter_custom_providers(environment=environment):
            if requested_aliases & _custom_entry_name_aliases(norm_name, entry):
                return _pool_keys_for_custom_entry(norm_name, entry)

    for norm_name, entry in _iter_custom_providers(environment=environment):
        entry_url = _norm_url(entry.get("base_url"))
        if entry_url and entry_url == normalized_url:
            return _pool_keys_for_custom_entry(norm_name, entry)
    return []


def get_custom_provider_pool_key(
    base_url: Optional[str],
    provider_name: Optional[str] = None,
    *,
    environment: PoolEnvironment,
) -> Optional[str]:
    """Preferred pool key for a custom provider: durable slug, else ``custom:<name>``.

    When provider_name is given, match by name first so two custom providers
    sharing a base_url keep separate keys.
    """
    environment.require_current_scope()
    candidates = custom_provider_pool_key_candidates(
        base_url, provider_name, environment=environment
    )
    return candidates[0] if candidates else None


def list_custom_pool_providers() -> List[str]:
    """Return all 'custom:*' pool keys that have entries in auth.json."""
    pool_data = read_credential_pool(None)
    return sorted(
        key for key in pool_data
        if key.startswith(CUSTOM_POOL_PREFIX)
        and isinstance(pool_data.get(key), list)
        and pool_data[key]
    )


def _get_custom_provider_config(
    pool_key: str, *, environment: PoolEnvironment
) -> Optional[Dict[str, Any]]:
    """Return the custom_providers config entry matching a pool key like 'custom:together.ai'."""
    environment.require_current_scope()
    if not pool_key.startswith(CUSTOM_POOL_PREFIX):
        return None
    suffix = pool_key[len(CUSTOM_POOL_PREFIX) :]
    return next(
        (
            entry
            for norm_name, entry in _iter_custom_providers(environment=environment)
            if norm_name == suffix
        ),
        None,
    )


def get_pool_strategy(provider: str, *, environment: PoolEnvironment) -> str:
    """Return the configured selection strategy for a provider."""
    environment.require_current_scope()
    config = _load_config_safe(environment=environment)
    strategies = config.get("credential_pool_strategies") if config else None
    if not isinstance(strategies, dict):
        return STRATEGY_FILL_FIRST
    strategy = str(strategies.get(provider, "") or "").strip().lower()
    return strategy if strategy in SUPPORTED_POOL_STRATEGIES else STRATEGY_FILL_FIRST


def _keyed_custom_pool_matches(
    pool_provider: str,
    provider_norm: str,
    base_url: Optional[str],
    *,
    environment: PoolEnvironment,
) -> bool:
    """Match a durable ``providers.<key>`` pool against runtime identities."""
    environment.require_current_scope()
    runtime_url = _norm_url(base_url)
    if not runtime_url:
        return False
    try:
        for normalized_name, entry in _iter_custom_providers(environment=environment):
            provider_key = _normalize_custom_pool_name(
                str(entry.get("provider_key") or "")
            )
            if provider_key != pool_provider:
                continue
            aliases = _custom_entry_name_aliases(normalized_name, entry)
            aliases.add(f"{CUSTOM_POOL_PREFIX}{normalized_name}")
            if provider_key:
                aliases.add(f"{CUSTOM_POOL_PREFIX}{provider_key}")
            configured_url = _norm_url(entry.get("base_url"))
            if provider_norm == "custom":
                return runtime_url == configured_url
            runtime_aliases = _requested_custom_name_aliases(provider_norm)
            return bool(runtime_aliases & aliases) and runtime_url == configured_url
    except Exception:
        return False
    return False


def _legacy_custom_pool_matches(
    pool_provider: str,
    provider_norm: str,
    runtime_url: str,
    *,
    environment: PoolEnvironment,
) -> bool:
    """Match a legacy ``custom:<name>`` pool against a named runtime identity."""
    environment.require_current_scope()
    try:
        for normalized_name, entry in _iter_custom_providers(environment=environment):
            if f"{CUSTOM_POOL_PREFIX}{normalized_name}" != pool_provider:
                continue
            aliases = {normalized_name}
            for value in (entry.get("name"), entry.get("provider_key")):
                alias = _normalize_custom_pool_name(str(value or ""))
                if alias:
                    aliases.add(alias)
                    if alias.startswith(CUSTOM_POOL_PREFIX):
                        aliases.add(alias[len(CUSTOM_POOL_PREFIX) :])
            configured_url = _norm_url(entry.get("base_url"))
            runtime_aliases = {_normalize_custom_pool_name(provider_norm)}
            if provider_norm.startswith(CUSTOM_POOL_PREFIX):
                runtime_aliases.add(
                    _normalize_custom_pool_name(
                        provider_norm[len(CUSTOM_POOL_PREFIX) :]
                    )
                )
            return bool(runtime_aliases & aliases) and runtime_url == configured_url
    except Exception:
        return False
    return False


def credential_pool_entry_serves_endpoint(entry: Any, base_url: Any, *, environment: PoolEnvironment) -> bool:
    """Whether a pooled credential may be bound to a session running at ``base_url``. ``_swap_credential``
    adopts the entry's base_url too, so a same-provider entry for another endpoint (public OpenAI vs. an
    Azure resource) would send the session's requests — and the entry's key — to the wrong host (#68237).
    Entries or sessions without endpoint metadata (legacy adapters, test doubles) cannot rebind and are accepted."""
    if not isinstance(base_url, str) or not base_url:
        return True
    entry_url = getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None)
    if not isinstance(entry_url, str) or not entry_url:
        return True
    return environment.normalize_endpoint(entry_url) == environment.normalize_endpoint(base_url)


def credential_pool_matches_provider(
    pool_or_provider: Any,
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
    environment: PoolEnvironment,
) -> bool:
    """Return whether a pool belongs to the requested runtime provider.

    Named custom endpoints may use three identities: the live agent can retain
    the configured name/provider key, newer runtime paths normalize it to
    ``custom``, and the pool may be keyed as the durable ``providers.<key>``
    slug or as legacy ``custom:<name>``. Accept those aliases only when the
    runtime endpoint belongs to the same configured custom provider. Empty
    identities fail closed. Legacy pool adapters without a ``provider``
    attribute remain compatible; production pools are scoped.
    """
    environment.require_current_scope()
    raw_pool_provider = getattr(pool_or_provider, "provider", None)
    if raw_pool_provider is None:
        if not isinstance(pool_or_provider, str):
            # Lightweight/unscoped pool adapters (old plugins, tests) may
            # expose only select()/has_credentials().
            return True
        raw_pool_provider = pool_or_provider
    pool_provider = str(raw_pool_provider or "").strip().lower()
    provider_norm = str(provider or "").strip().lower()
    if not pool_provider or not provider_norm:
        return False
    if not pool_provider.startswith(CUSTOM_POOL_PREFIX):
        if pool_provider == provider_norm:
            return True
        return _keyed_custom_pool_matches(
            pool_provider, provider_norm, base_url, environment=environment
        )
    if provider_norm == "custom":
        try:
            matched_pool = get_custom_provider_pool_key(
                base_url or "", environment=environment
            )
            if str(matched_pool or "").strip().lower() == pool_provider:
                return True
            candidates = custom_provider_pool_key_candidates(
                base_url or "", environment=environment
            )
        except Exception:
            return False
        return pool_provider in {str(key).strip().lower() for key in candidates}

    runtime_url = _norm_url(base_url)
    if not runtime_url:
        return False
    return _legacy_custom_pool_matches(
        pool_provider, provider_norm, runtime_url, environment=environment
    )


def resolve_runtime_pool_key(
    provider: Optional[str], base_url: Optional[str], *, environment: PoolEnvironment
) -> str:
    """Resolve the credential-pool key for a runtime provider identity.

    Named custom runtimes retain their configured alias while their pool may
    be stored under the durable ``providers.<key>`` slug or legacy
    ``custom:<name>``. Return that scoped key only when the canonical
    provider/endpoint boundary accepts it; otherwise preserve the normalized
    runtime identity so callers fail closed.
    """
    environment.require_current_scope()
    provider_norm = str(provider or "").strip().lower()
    if not provider_norm:
        return ""

    def _accepts(candidate: str) -> bool:
        return credential_pool_matches_provider(
            candidate, provider_norm, base_url=base_url, environment=environment
        )

    try:
        if provider_norm == "custom":
            candidate = get_custom_provider_pool_key(base_url, environment=environment)
            if candidate and _accepts(candidate):
                return str(candidate).strip().lower()
        else:
            # Named/exact custom runtimes are keyed by identity: search the
            # configured candidates by identity before endpoint so a sibling
            # sharing the URL cannot lend its pool.
            for normalized_name, entry in _iter_custom_providers(
                environment=environment
            ):
                for candidate in _pool_keys_for_custom_entry(normalized_name, entry):
                    if _accepts(candidate):
                        return candidate
    except Exception:
        pass
    return provider_norm


DEFAULT_MAX_CONCURRENT_PER_CREDENTIAL = 1


# --- Multi-profile root write-through ---


def _guarded_global_root(global_path: Optional[Path]) -> Optional[Path]:
    """Apply the pytest seat belt to a resolved global-root auth.json path.

    ``None`` means classic mode (profile == root) or "refuse": under pytest,
    never write the real user's ``~/.hermes/auth.json`` even when HERMES_HOME
    points at a profile path (mirrors the read-side guard in
    ``_load_global_auth_store``). Uses the unmodified HOME env, not
    ``Path.home()`` which fixtures may monkeypatch.
    """
    if global_path is None:
        return None
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_env = os.environ.get("HOME", "")
        if real_home_env:
            real_root = Path(real_home_env) / ".hermes" / "auth.json"
            try:
                # Comparing the guard path must not probe the real auth store.
                if os.path.normcase(os.path.abspath(global_path)) == os.path.normcase(os.path.abspath(real_root)):
                    return None
            except Exception:
                return None
    return global_path


def _write_through_provider_state_to_global_root(
    provider_id: str, state: Dict[str, Any]
) -> None:
    "Persist a rotated OAuth ``state`` into the global-root auth.json.\n\n    Best-effort write-through for the multi-profile rotation hazard: nous,\n    openai-codex, and xai-oauth rotate the refresh_token on refresh, so when\n    a profile pool refresh rotates a grant it resolved from the root fallback,\n    the rotated chain must land back in root. Otherwise root keeps a revoked\n    refresh token and every other profile dies with ``refresh_token_reused``\n    / ``invalid_grant`` once its access token expires.\n\n    Only updates ``providers.<provider_id>`` in the root store; never touches\n    the profile store (the caller already saved that). Swallows all errors —\n    a failed write-through degrades to root-stale and must never break the\n    profile's own successful save. Mirrors\n    ``auth.providers.xai._write_through_xai_oauth_to_global_root``.\n\n    See #48415.\n"
    try:
        global_path = _guarded_global_root(auth_storage._global_auth_file_path())
    except Exception:
        return
    if global_path is None:
        return
    try:
        auth_provider_state._persist_provider_state_to_store(
            provider_id, state, global_path, set_active=False
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug(
            "%s pool refresh: write-through to global root failed: %s", provider_id, exc
        )


def _singleton_target_for_entry(
    pool: "CredentialPool", entry: "PooledCredential"
) -> Optional[Path]:
    """Root ``.anthropic_oauth.json`` when *entry* is a borrowed hermes_pkce row, else None."""
    if entry.source != "hermes_pkce" or entry.id not in getattr(
        pool, "_borrowed_root_ids", ()
    ):
        return None
    try:
        from auth.providers.anthropic import _root_hermes_oauth_file

        return _root_hermes_oauth_file()
    except Exception:
        return None


def _store_owns_pool_provider(auth_store: Dict[str, Any], provider: str) -> bool:
    """True when an already-loaded *auth_store* has its own rows for *provider*."""
    pool = auth_store.get("credential_pool")
    entries = pool.get(provider) if isinstance(pool, dict) else None
    return isinstance(entries, list) and bool(entries)


def _profile_owns_pool_provider(provider: str) -> bool:
    """True when the ACTIVE auth.json has its own rows for *provider*.

    Named profiles with no local rows read the provider through the
    ``read_credential_pool`` global-root fallback ("borrowing").
    """
    # Classic mode (profile == root) has no root fallback, so the answer is always "owns";
    # skip the per-call auth.json re-read on this hot load_pool path.
    if auth_storage._global_auth_file_path() is None:
        return True
    try:
        auth_store = _load_auth_store()
    except Exception:
        return True  # unreadable store: assume ownership, keep legacy path
    return _store_owns_pool_provider(auth_store, provider)


def _borrowed_single_use_pool_root() -> Optional[Path]:
    """Global-root auth.json when persisting a BORROWED single-use pool, else None.

    ``None`` means "persist to the active store as usual": classic mode
    (profile == root), or the profile owns its own rows for this provider.
    """
    try:
        return _guarded_global_root(_global_auth_file_path())
    except Exception:
        return None


def _update_root_pool_rows(
    provider: str, payloads: List[Dict[str, Any]], global_path: Path,
    *, status_cleared_ids: Optional[Iterable[str]] = None,
    token_bases: Optional[Dict[str, Tuple[Any, Any]]] = None,
) -> List[Dict[str, Any]]:
    """UPDATE-ONLY merge of *payloads* into the root store's rows for *provider*.

    A borrower may refresh the root's rows (rotation, cooldown state) but
    never add or delete them — the root owns their lifecycle. In particular a
    profile's singleton-prune (it has no ``.anthropic_oauth.json`` of its own)
    must not delete the root grant, so ``removed_ids`` is ignored by callers.
    """
    with _auth_store_lock(target_path=global_path):
        store = _load_auth_store(global_path)
        pool = store.get("credential_pool")
        if not isinstance(pool, dict):
            pool = {}
            store["credential_pool"] = pool
        existing = pool.get(provider)
        existing_list = existing if isinstance(existing, list) else []
        incoming_by_id = auth_pool_persistence._entry_ids(payloads)
        cleared = {cid for cid in (status_cleared_ids or ()) if cid}
        bases = token_bases or {}
        merged: List[Dict[str, Any]] = []
        changed = False
        for disk_entry in existing_list:
            did = disk_entry.get("id") if isinstance(disk_entry, dict) else None
            incoming = incoming_by_id.get(did) if did else None
            if incoming is None:
                merged.append(disk_entry)
                continue
            updated = auth_pool_persistence._merge_pool_row_generation(
                incoming, disk_entry, provider,
                base_pair=bases.get(did), status_cleared=did in cleared,
            )
            if updated != disk_entry:
                changed = True
            merged.append(updated)
        if changed:
            pool[provider] = merged
            _save_auth_store(store, target_path=global_path)
        return merged


def persist_pool_entries(
    provider: str,
    payloads: List[Dict[str, Any]],
    *,
    removed_ids: Optional[Iterable[str]] = None,
    status_cleared_ids: Optional[Iterable[str]] = None,
    token_bases: Optional[Dict[str, Tuple[Any, Any]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Persist a provider's pool rows to the store that OWNS them.

    A named profile that sees a single-use-refresh provider (see
    ``SINGLE_USE_REFRESH_POOL_PROVIDERS``) only through the global-root fallback must not
    materialize a local ``credential_pool.<provider>`` copy: that copy forks
    the single-use refresh token, the first profile to rotate commits the new
    pair only to its own file, and root plus every sibling die with
    ``invalid_grant`` (#100339). Such rows are written back to the root store
    (under the root lock); everything else goes to the active store.
    """
    if provider in SINGLE_USE_REFRESH_POOL_PROVIDERS and not _profile_owns_pool_provider(provider):
        global_path = _borrowed_single_use_pool_root()
        if global_path is not None:
            try:
                return _update_root_pool_rows(
                    provider, payloads, global_path,
                    status_cleared_ids=status_cleared_ids, token_bases=token_bases,
                )
            except Exception as exc:
                # Fail closed on the FORK, not on the save: never fall back to
                # writing a local copy (that IS the bug). The in-memory pool
                # still holds the rotated pair for this process.
                logger.warning(
                    "%s pool: write-through of borrowed root grant failed (%s); "
                    "not materializing a profile-local copy",
                    provider, exc,
                )
            return None
    return write_credential_pool(
        provider, payloads, removed_ids=removed_ids, status_cleared_ids=status_cleared_ids,
        token_bases=token_bases,
    )


# --- Per-provider singleton refresh plumbing -------------------------------
#
# Providers whose OAuth singleton lives in auth.json ``providers.<id>.tokens``
# (Codex, xAI): log names (sync-message form, "<name> OAuth" form),
# Canonical auth.providers refresh function and terminal-error predicate (looked
# up at call time so tests can patch them).
_TOKENS_SINGLETON_PROVIDERS: Dict[str, Tuple[str, str, str, str]] = {
    "openai-codex": (
        "Codex",
        "Codex",
        "refresh_codex_oauth_pure",
        "_is_terminal_codex_oauth_refresh_error",
    ),
    "xai-oauth": (
        "xAI OAuth",
        "xAI",
        "refresh_xai_oauth_pure",
        "_is_terminal_xai_oauth_refresh_error",
    ),
}

# Built-in providers whose pooled OAuth entries ``_refresh_entry_impl`` can actually refresh. Plugin
# providers are refreshable when their profile ships ``refresh_credential`` (see
# auth.plugin_hooks.is_refreshable_oauth_provider); any other provider is returned
# unchanged by that path, so callers must not report a refresh for them.
REFRESHABLE_OAUTH_PROVIDERS = frozenset({
    "anthropic",
    "nous",
    *_TOKENS_SINGLETON_PROVIDERS,
})

# Providers whose refresh tokens are single-use: the sync -> POST -> write-back
# sequence must be serialized across processes under the auth-store flock.
# ``nous`` is deliberately absent even though it is in SINGLE_USE_REFRESH_POOL_PROVIDERS:
# its refresh path serializes on its own auth-store lock (``_refresh_entry_impl`` nous branch).
_SINGLE_USE_REFRESH_PROVIDERS = ("openai-codex", "xai-oauth", "anthropic")

_REFRESH_TIMEOUT_ENV_VARS = {
    "openai-codex": "HERMES_CODEX_REFRESH_TIMEOUT_SECONDS",
    "xai-oauth": "HERMES_XAI_REFRESH_TIMEOUT_SECONDS",
}

# Singleton-seeded source whose exhausted/DEAD pool row may be revived by a
# re-auth another process wrote to the provider's store.
_RESYNC_SOURCE = {
    "anthropic": "claude_code",
    "nous": "device_code",
    "openai-codex": "device_code",
    "xai-oauth": "device_code",
}


class _RefreshDone(Exception):
    """Raised inside a provider refresher to short-circuit ``_refresh_entry_impl`` with ``result``."""

    def __init__(self, result: Optional["PooledCredential"]):
        super().__init__()
        self.result = result


class CredentialPool(CredentialPoolRefreshMixin, CredentialPoolAdminMixin, CredentialPoolModelCooldownMixin):
    def __init__(self, provider: str, entries: List[PooledCredential], *, environment: PoolEnvironment):
        environment.require_current_scope()
        self.environment = environment
        self._provider_hooks = environment.provider_hooks(provider)
        self.provider = provider
        self._entries = sorted(entries, key=lambda entry: entry.priority)
        self._current_id: Optional[str] = None
        # Ids of rows read via the global-root fallback (single-use OAuth
        # providers only); set by load_pool(), consumed by add_entry().
        self._borrowed_root_ids: Set[str] = set()
        self._persisted_token_pairs: Dict[str, Tuple[Any, Any]] = {}
        self._strategy = get_pool_strategy(provider, environment=self.environment)
        # RLock: _replace_entry/_persist self-acquire it so the DEFERRED
        # single-use-token refresh path (network I/O outside the lock by
        # design) still serializes its pool mutations; in-lock callers
        # re-acquire reentrantly.
        self._lock = threading.RLock()
        self._active_leases: Dict[str, int] = {}
        self._max_concurrent = DEFAULT_MAX_CONCURRENT_PER_CREDENTIAL
        # Monotonic timestamp of the last "no available entries" log (see
        # NO_AVAILABLE_ENTRIES_LOG_THROTTLE_SECONDS). Re-armed to None on every
        # successful selection so a recover->re-exhaust transition logs promptly.
        self._last_no_entries_log_at: Optional[float] = None
        # #70401: consecutive mark_exhausted_and_rotate() calls whose supplied
        # credential identity matched no pool entry. These mark nothing
        # exhausted, so without a cap the pool never converges to "no available
        # entries" and the caller's 401 retry loop runs unbounded. Reset when a
        # real entry is identified or an escape path returns None.
        self._unmatched_rotation_streak: int = 0

    # ---- read accessors ---------------------------------------------------

    def has_credentials(self) -> bool:
        self.environment.require_current_scope()
        with self._lock:
            return bool(self._entries)

    def has_available(self, *, model: Optional[str] = None) -> bool:
        """True if at least one entry is not currently in exhaustion cooldown.

        ``_available_entries`` is not read-only (it prunes aged-out DEAD
        manual entries and persists), so it must run under ``self._lock``
        like every other caller or a probe can race a concurrent rotation.
        """
        self.environment.require_current_scope()
        with self._lock:
            available, _pending = self._available_entries(model=model)
            return bool(available)

    def next_available_at(self, *, model: Optional[str] = None) -> Optional[float]:
        """Earliest epoch time (seconds) any entry re-enters rotation.

        ``None`` when an entry is available now, or when no exhausted entry
        carries a usable recovery time (empty pool, or only ``STATUS_DEAD``
        entries). Callers must treat ``None`` as "no wait information".
        Runs under ``self._lock`` for the same reason as ``has_available``.
        """
        self.environment.require_current_scope()
        with self._lock:
            available, _pending = self._available_entries(model=model)
            if available:
                return None
            # Mirror _available_entries: a sole credential's transient throttle
            # cools down in seconds, and the fallback restore gate must not
            # wait an hour for a 60s cooldown.
            sole_credential = self._is_sole_credential()
            candidates = [
                until
                for until in (
                    _exhausted_until(entry, sole_credential=sole_credential)
                    for entry in self._entries
                    if entry.last_status == STATUS_EXHAUSTED
                )
                if until is not None
            ]
            candidates.extend(
                until
                for entry in self._entries
                if entry.last_status != STATUS_DEAD
                for until in (model_cooldown_until(entry, model),)
                if until is not None
            )
            return min(candidates) if candidates else None

    def entries(self) -> List[PooledCredential]:
        self.environment.require_current_scope()
        with self._lock:
            return list(self._entries)

    def _is_sole_credential(self) -> bool:
        """DEAD entries never re-enter rotation, so <=1 non-DEAD entry means nothing to rotate to."""
        return sum(1 for e in self._entries if e.last_status != STATUS_DEAD) <= 1

    def _find(self, predicate: Callable[[PooledCredential], bool]) -> Optional[PooledCredential]:
        return next((e for e in self._entries if predicate(e)), None)

    def _current_unlocked(self) -> Optional[PooledCredential]:
        if not self._current_id:
            return None
        return self._find(lambda e: e.id == self._current_id)

    def current(self) -> Optional[PooledCredential]:
        self.environment.require_current_scope()
        with self._lock:
            return self._current_unlocked()

    def entry_id_for_api_key(self, api_key_hint: Any = None) -> Optional[str]:
        """Stable id for the runtime credential in use.

        Prefer the current selection when it still supplies ``api_key_hint``;
        if the cursor was cleared, fall back to an unambiguous key match.
        """
        self.environment.require_current_scope()
        with self._lock:
            current = self._current_unlocked()
            if current is not None and (api_key_hint is None or current.runtime_api_key == api_key_hint):
                return current.id
            if api_key_hint is None:
                return None
            matches = [e for e in self._entries if e.runtime_api_key == api_key_hint]
            return matches[0].id if len(matches) == 1 else None

    # ---- mutation primitives (self-locking) --------------------------------

    def _replace_entry(self, old: PooledCredential, new: PooledCredential) -> None:
        """Swap an entry in-place by id, preserving sort order.

        Self-locking (RLock) so the deferred refresh path — which runs outside
        the pool lock — cannot tear ``self._entries`` against a concurrent
        select()/rotation.
        """
        with self._lock:
            for idx, entry in enumerate(self._entries):
                if entry.id == old.id:
                    self._entries[idx] = new
                    return

    def _persist(
        self,
        *,
        removed_ids: Optional[List[str]] = None,
        status_cleared_ids: Optional[List[str]] = None,
    ) -> None:
        # Self-locking: snapshotting self._entries must not race a rotation.
        with self._lock:
            payloads = [entry.to_dict() for entry in self._entries]
            written = persist_pool_entries(
                self.provider, payloads,
                removed_ids=removed_ids,
                status_cleared_ids=status_cleared_ids,
                token_bases=self._persisted_token_pairs,
            )
            if written is None:
                return
            rows = auth_pool_persistence._entry_ids(written)
            self._persisted_token_pairs = auth_pool_persistence._token_pairs_by_id(written)
            for entry in self._entries:
                row = rows.get(entry.id)
                pair = self._persisted_token_pairs.get(entry.id, (None, None))
                # Reference-only rows are intentionally secret-free on disk; never dehydrate
                # their live in-memory credential while adopting a concurrent generation.
                if row is None or not any(pair):
                    continue
                # Adopt only rows the store overrode with a peer's newer pair; re-hydrating an
                # unchanged row would replace the live object (and pull peer cooldown state
                # merged into the written row) on every ordinary flush.
                if pair != (entry.access_token, entry.refresh_token):
                    self._replace_entry(entry, PooledCredential.from_dict(self.provider, row))

    def _adopt(self, entry: PooledCredential, *, persist: bool = True, **updates: Any) -> PooledCredential:
        """``replace(entry, **updates)``, swap it into the pool, optionally persist."""
        updated = replace(entry, **updates)
        self._replace_entry(entry, updated)
        if persist:
            self._persist()
            # _persist may have swapped in a peer's newer token generation; hand callers
            # the live entry so they don't rebind the client to the stale pair.
            return self._find(lambda e: e.id == updated.id) or updated
        return updated

    def _quarantine_sources(self, entry: PooledCredential, sources: Set[str]) -> None:
        """Drop every entry seeded from *sources* and persist the removal.

        Atomic read-modify-write of ``self._entries``: this runs on the
        DEFERRED refresh path (outside the pool lock), so take the RLock here;
        still-locked callers re-enter safely.
        """
        with self._lock:
            removed_ids = [item.id for item in self._entries if item.source in sources]
            self._entries = [item for item in self._entries if item.source not in sources]
            if self._current_id == entry.id:
                self._current_id = None
            self._persist(removed_ids=removed_ids)

    # ---- exhaustion --------------------------------------------------------

    def _is_terminal_auth_failure(
        self,
        status_code: Optional[int],
        normalized_error: Dict[str, Any],
    ) -> bool:
        """Detect upstream-permanent OAuth failures that won't recover on TTL.

        Only 401s whose reason is a known terminal OAuth state count;
        token_expired (refreshable) and reason-less 401s (possible glitch)
        stay transient, as do 429/402. The one status-independent case is
        ``CREDENTIAL_PERSIST_FAILED_REASON``: no upstream response is involved,
        the rotated pair never became durable and only a re-auth recovers it.
        """
        raw_reason = normalized_error.get("reason")
        reason = raw_reason.strip().lower() if isinstance(raw_reason, str) else ""
        if reason == CREDENTIAL_PERSIST_FAILED_REASON:
            return True
        return status_code == 401 and reason in _TERMINAL_AUTH_REASONS

    def _mark_exhausted(
        self,
        entry: PooledCredential,
        status_code: Optional[int],
        error_context: Optional[Dict[str, Any]] = None,
        *,
        persist: bool = True,
        failure_reason: Optional[str] = None,
    ) -> PooledCredential:
        normalized_error = _normalize_error_context(error_context)
        # Permanent OAuth failures become STATUS_DEAD, not STATUS_EXHAUSTED:
        # otherwise a revoked credential re-enters rotation every hour and
        # fails immediately until the user removes it (#32849).
        terminal = self._is_terminal_auth_failure(status_code, normalized_error)
        # Carry the classifier's verdict so the cooldown is sized by what
        # actually failed (a billing 403 must not get the sole-credential
        # transient cooldown); absent a classification, clear a stale one.
        updated_extra = dict(entry.extra)
        if failure_reason:
            updated_extra["failure_reason"] = failure_reason
        else:
            updated_extra.pop("failure_reason", None)
        return self._adopt(
            entry,
            persist=persist,
            last_status=STATUS_DEAD if terminal else STATUS_EXHAUSTED,
            last_status_at=time.time(),
            last_error_code=status_code,
            last_error_reason=normalized_error.get("reason"),
            last_error_message=normalized_error.get("message"),
            last_error_reset_at=normalized_error.get("reset_at"),
            extra=updated_extra,
        )

    # ---- cross-process token resync ---------------------------------------
    #
    # OAuth refresh tokens are single-use. When another process (CLI, another
    # profile, a concurrent cron) rotates a pair, our in-memory entry holds a
    # consumed refresh token; replaying it yields ``refresh_token_reused`` /
    # ``invalid_grant``. These helpers adopt the fresher pair from wherever the
    # provider's token authority lives, clearing stale exhaustion state.


    # ---- selection ---------------------------------------------------------

    def select(self, *, model: Optional[str] = None) -> Optional[PooledCredential]:
        self.environment.require_current_scope()
        entry, pending_refresh = self._select_under_lock(model=model)
        if pending_refresh:
            self._refresh_pending_entries(pending_refresh)
            # Re-select now that the refreshed entries are back in the pool.
            if entry is None:
                entry, _ = self._select_under_lock(model=model)
        if entry is not None:
            self._unmatched_rotation_streak = 0
        return entry

    def _select_under_lock(self, *, model: Optional[str] = None) -> Tuple[Optional[PooledCredential], List[PooledCredential]]:
        with self._lock:
            return self._select_unlocked(model=model)

    def _refresh_pending_entries(self, pending: List[PooledCredential]) -> None:
        """Refresh deferred single-use-token entries OUTSIDE the pool lock.

        Each refresh takes the cross-process ``_auth_store_lock`` (20+ s
        possible) and merges into the pool through the self-locking mutation
        primitives; failures are silently skipped.
        """
        for entry in pending:
            self._refresh_entry(entry, force=False)

    def _reset_cleared_after(self, entry: PooledCredential) -> Optional[float]:
        """Epoch of a ``hermes auth reset`` persisted by another process AFTER *entry*'s status, else None."""
        try:
            row = next((p for p in read_credential_pool(self.provider)
                        if isinstance(p, dict) and p.get("id") == entry.id), None)
            cleared = _parse_absolute_timestamp((row or {}).get("status_cleared_at"))
        except Exception as exc:
            logger.debug("Pool entry %s: could not read reset marker: %s", entry.id, exc)
            return None
        return cleared if cleared and cleared > (entry.last_status_at or 0.0) else None

    def _resync_stale_entry(self, entry: PooledCredential) -> PooledCredential:
        """Re-read an exhausted/DEAD singleton-seeded entry from its token authority.

        The user may have re-authed (``hermes model`` / ``hermes auth``, the
        Claude Code CLI, another profile) leaving fresh tokens on disk while
        the pool entry is frozen behind ``last_error_reset_at``. A ``hermes auth
        reset`` run from another process while this pool is live is honoured the
        same way (#89415): the in-memory cooldown would otherwise outlive it.
        """
        if entry.last_status not in {STATUS_EXHAUSTED, STATUS_DEAD}:
            return entry
        cleared_at = self._reset_cleared_after(entry)
        if cleared_at is not None:
            return self._adopt(entry, persist=False, **_MARK_OK, status_cleared_at=cleared_at)
        if entry.source != _RESYNC_SOURCE.get(self.provider):
            return entry
        if self.provider == "anthropic":
            return self._sync_anthropic_entry_from_credentials_file(entry)
        if self.provider == "nous":
            return self._sync_nous_entry_from_auth_store(entry)
        return self._sync_entry_from_auth_store(entry)

    def _available_entries(
        self, *, clear_expired: bool = False, refresh: bool = False, model: Optional[str] = None,
    ) -> Tuple[List[PooledCredential], List[PooledCredential]]:
        """Return (available, pending_refresh) for entries not in cooldown.

        *clear_expired* resets elapsed cooldowns to STATUS_OK and persists.
        *refresh* refreshes entries needing a token refresh (skipped on
        failure) — except single-use-token providers (openai-codex,
        xai-oauth), which are returned as *pending_refresh* so the caller
        refreshes them outside the lock instead of stalling every pool
        consumer during cross-process flock acquisition + OAuth network I/O.
        """
        now = time.time()
        cleared_any = False
        entries_to_prune: List[str] = []
        available: List[PooledCredential] = []
        pending_refresh: List[PooledCredential] = []
        sole_credential = self._is_sole_credential()
        for entry in self._entries:
            # Borrowed credentials persist as metadata-only references and are
            # hydrated from their live source on load; never lease an
            # unhydrated duplicate as an empty key.
            if entry.auth_type == AUTH_TYPE_API_KEY and not entry.runtime_api_key:
                continue
            synced = self._resync_stale_entry(entry)
            if synced is not entry:
                entry = synced
                cleared_any = True
            if entry.last_status == STATUS_DEAD:
                # Manual DEAD credentials are pruned after a 24h quiet window;
                # singleton-seeded ones stay (audit trail, and the seeder would
                # re-create them anyway). DEAD never re-enters via TTL — only a
                # write-side re-auth sync clears it.
                if _is_manual_source(entry.source):
                    dead_at = entry.last_status_at or 0
                    if dead_at and now - dead_at > DEAD_MANUAL_PRUNE_TTL_SECONDS:
                        logger.warning(
                            "credential pool: pruning DEAD manual entry %s "
                            "(reason=%s, age=%.1fh) — re-add via `hermes auth add %s`",
                            entry.label or entry.id[:8],
                            entry.last_error_reason or "unknown",
                            (now - dead_at) / 3600.0,
                            self.provider,
                        )
                        entries_to_prune.append(entry.id)  # can't mutate while iterating
                        cleared_any = True
                continue
            if model_cooldown_until(entry, model) is not None:
                continue
            if entry.last_status == STATUS_EXHAUSTED:
                exhausted_until = _exhausted_until(entry, sole_credential=sole_credential)
                # Codex quota windows can reopen EARLY; a throttled live probe
                # lifts a stale cooldown (issue #43747).
                if (
                    exhausted_until is not None
                    and now < exhausted_until
                    and not (clear_expired and self._codex_quota_restored_upstream(entry))
                ):
                    continue
                if clear_expired:
                    entry = self._adopt(entry, persist=False, **_MARK_OK)
                    cleared_any = True
            if refresh and self._entry_needs_refresh(entry):
                if self.provider in _TOKENS_SINGLETON_PROVIDERS:
                    pending_refresh.append(entry)
                    continue
                refreshed = self._refresh_entry(entry, force=False)
                if refreshed is None:
                    continue
                entry = refreshed
            if entry.auth_type == AUTH_TYPE_OAUTH and not (entry.access_token or "").strip():
                # A borrowed OAuth row that failed to hydrate (or a sanitized
                # row read straight off disk); leasing it would send an empty
                # bearer. The API-key guard above does not cover it.
                continue
            available.append(entry)
        if entries_to_prune:
            pruned_ids = set(entries_to_prune)
            self._entries = [e for e in self._entries if e.id not in pruned_ids]
        if cleared_any:
            self._persist(removed_ids=entries_to_prune)
        return available, pending_refresh

    def _log_no_available_entries(self) -> None:
        """Emit the empty-pool INFO line at most once per throttle window."""
        now = time.monotonic()
        last = self._last_no_entries_log_at
        if last is not None and (now - last) < NO_AVAILABLE_ENTRIES_LOG_THROTTLE_SECONDS:
            return
        self._last_no_entries_log_at = now
        logger.info("credential pool: no available entries (all exhausted or empty)")

    def _select_unlocked(
        self, *, refresh: bool = True, count: bool = True, model: Optional[str] = None,
    ) -> Tuple[Optional[PooledCredential], List[PooledCredential]]:
        """Select the best available entry; returns ``(entry, pending_refresh)``.

        ``count=False`` skips the ``request_count`` bump for selections that are
        not going to serve a request (a forced-refresh target lookup).
        """
        available, pending_refresh = self._available_entries(clear_expired=True, refresh=refresh, model=model)
        if not available:
            self._current_id = None
            self._log_no_available_entries()
            return None, pending_refresh

        # The pool recovered; re-arm the throttle so a later re-exhaustion
        # logs immediately.
        self._last_no_entries_log_at = None

        if self._strategy == STRATEGY_RANDOM:
            entry = random.choice(available)
        elif self._strategy == STRATEGY_LEAST_USED and len(available) > 1:
            entry = min(available, key=lambda e: e.request_count)
        else:
            entry = available[0]
        # Count the selection under every strategy. The counter is ``least_used``'s
        # baseline and reaches auth.json on the next persist (exhaustion, rotation,
        # refresh); it used to move only while ``least_used`` was active.
        if count:
            entry = self._adopt(entry, persist=False, request_count=entry.request_count + 1)
        if self._strategy == STRATEGY_ROUND_ROBIN and len(available) > 1:
            rotated = [candidate for candidate in self._entries if candidate.id != entry.id]
            rotated.append(replace(entry, priority=len(self._entries) - 1))
            self._entries = [replace(candidate, priority=idx) for idx, candidate in enumerate(rotated)]
            self._persist()
            entry = self._find(lambda candidate: candidate.id == entry.id) or entry
        self._current_id = entry.id
        return entry, pending_refresh

    def peek(self) -> Optional[PooledCredential]:
        self.environment.require_current_scope()
        with self._lock:
            current = self._current_unlocked()
            if current is not None:
                return current
            available, _pending = self._available_entries()
            return available[0] if available else None

    def reclaim(self, credential_id: str, *, model: Optional[str] = None) -> Optional[PooledCredential]:
        """Entry *credential_id* once its cooldown has lifted (cleared and token-refreshed the way
        ``select`` would), else ``None``. Never bumps ``request_count`` or round-robin order: a
        live session asking "may I go back?" every turn is not a request."""
        self.environment.require_current_scope()
        with self._lock:
            available, pending = self._available_entries(clear_expired=True, refresh=True, model=model)
        if any(e.id == credential_id for e in pending):
            self._refresh_pending_entries([e for e in pending if e.id == credential_id])
            with self._lock:
                available, _pending = self._available_entries(clear_expired=True, refresh=True, model=model)
        return next((e for e in available if e.id == credential_id), None)

    # ---- rotation ----------------------------------------------------------

    def _identify_failed_entry(
        self, credential_id: Optional[str], api_key_hint: Optional[str],
    ) -> Optional[PooledCredential]:
        """Resolve the entry that issued a failed request from its supplied identity."""
        entry = None
        if credential_id:
            entry = self._find(lambda e: e.id == credential_id)
            # #79156: when both identities disagree, trust the key that made
            # the request. A stale ``_credential_pool_entry_id`` (per-turn env
            # refresh rewrote ``api_key`` without rebinding the id) would
            # otherwise quarantine a healthy fallback for days.
            if entry is not None and api_key_hint and entry.runtime_api_key != api_key_hint:
                hint_entry = self._find(lambda e: e.runtime_api_key == api_key_hint)
                if hint_entry is not None:
                    logger.info(
                        "credential pool: credential_id %s runtime key "
                        "does not match api_key_hint; attributing failure "
                        "to key-matched entry %s instead (#79156)",
                        (entry.label or entry.id[:8]),
                        (hint_entry.label or hint_entry.id[:8]),
                    )
                # Otherwise the id is stale and the request key is not in the
                # pool — drop the id so we do not mark the wrong entry.
                entry = hint_entry
        if entry is None and api_key_hint:
            # Prefer the entry whose key actually failed: on a pool freshly
            # loaded from disk current() is None and _select_unlocked() would
            # return the NEXT key — the wrong one.
            entry = self._find(lambda e: e.runtime_api_key == api_key_hint)
        return entry

    def _rotate_unmatched(self) -> Optional[PooledCredential]:
        """Rotate without marking anything when the failed identity matches no entry.

        Falling through to current()/_select_unlocked() would bench an
        innocent healthy key for the full TTL. But this must be BOUNDED
        (#70401): with OAuth-token auth the 401's key hint never matches any
        ``runtime_api_key``, so every retry lands here, nothing is marked, and
        the caller retries the same dead token forever (~6/sec, starving the
        event loop). Cap consecutive no-mark rotations at one lap of the
        available entries, then surface the error; no cooldown is written.
        """
        self._unmatched_rotation_streak += 1
        available_count = len(self._available_entries()[0])
        if self._unmatched_rotation_streak > max(available_count, 1):
            logger.warning(
                "credential pool: failed credential identity matched no "
                "%s entry for %d consecutive rotations (pool size %d) — "
                "surfacing the error instead of rotating again",
                self.provider, self._unmatched_rotation_streak, available_count,
            )
            self._unmatched_rotation_streak = 0
            self._current_id = None
            return None
        logger.info(
            "credential pool: failed credential identity matched no %s "
            "entry; rotating without marking any credential exhausted",
            self.provider,
        )
        self._current_id = None
        next_entry, _pending = self._select_unlocked(refresh=False)
        if next_entry is not None and len(self._available_entries()[0]) == 1:
            # A single-entry pool cannot rotate: returning its only entry would
            # report a recovery without changing the credential, and the
            # caller retries the same 401 indefinitely.
            self._unmatched_rotation_streak = 0
            self._current_id = None
            return None
        return next_entry

    def mark_exhausted_and_rotate(
        self,
        *,
        status_code: Optional[int],
        error_context: Optional[Dict[str, Any]] = None,
        api_key_hint: Optional[str] = None,
        credential_id: Optional[str] = None,
        failure_reason: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Optional[PooledCredential]:
        self.environment.require_current_scope()
        with self._lock:
            identity_supplied = bool(credential_id or api_key_hint)
            entry = self._identify_failed_entry(credential_id, api_key_hint)
            if entry is None and identity_supplied:
                return self._rotate_unmatched()
            # A real entry was identified — any prior unmatched streak is stale.
            self._unmatched_rotation_streak = 0
            if entry is None:
                entry = self._current_unlocked() or self._select_unlocked(refresh=False)[0]
            if entry is None:
                return None
            _label = entry.label or entry.id[:8]
            if self._is_model_scoped_failure(status_code, model, failure_reason):
                # A generic Anthropic 429 (per-model rate limit) or a Codex account model
                # entitlement rejection: bench this model only, the credential stays
                # available for its siblings.
                self._cool_down_model(entry, model, error_context, failure_reason=failure_reason)
                logger.info("credential pool: %s unavailable for model %s; other models stay available", _label, model)
                self._current_id = None
                next_entry, _pending = self._select_unlocked(refresh=False, model=model)
                return next_entry
            self._mark_exhausted(entry, status_code, error_context, failure_reason=failure_reason)
            # A 402/429/401 is a key-level failure, and the same key can back
            # several entries (an explicit entry plus a ``model_config`` row
            # auto-seeded from ``model.api_key``). Marking only the first
            # leaves siblings OK, ``_select_unlocked()`` keeps handing back
            # the depleted key, and rotation never converges (~2.5 min hang).
            # Mark every entry sharing the failed key.
            failed_runtime_key = entry.runtime_api_key
            if identity_supplied and failed_runtime_key:
                siblings = [
                    s for s in self._entries if s.id != entry.id and s.runtime_api_key == failed_runtime_key
                ]
                for sibling in siblings:
                    self._mark_exhausted(
                        sibling, status_code, error_context, persist=False, failure_reason=failure_reason,
                    )
                if siblings:
                    self._persist()
            # Re-read the updated entry to log the correct terminal state.
            updated_entry = self._find(lambda e: e.id == entry.id) or entry
            if updated_entry.last_status == STATUS_DEAD:
                logger.warning(
                    "credential pool: marking %s DEAD (status=%s, reason=%s) — "
                    "permanently failed, will NOT re-enter rotation until re-auth",
                    _label, status_code, updated_entry.last_error_reason or "unknown",
                )
            else:
                logger.info("credential pool: marking %s exhausted (status=%s), rotating", _label, status_code)
            self._current_id = None
            next_entry, _pending = self._select_unlocked(refresh=False)
            if next_entry is not None and next_entry.id == entry.id:
                # No-recovery guard (#97315): selection handed back the very entry that was
                # just marked (the auth-store sync adopted fresher tokens, or a quota probe
                # false-positive lifted the bench mid-selection). Returning it reports a
                # successful rotation without changing the credential, so the caller retries
                # the same 429 forever (~2 req/s for hours). Mirror the single-entry guard on
                # the unmatched-identity branch: surface the failure instead.
                logger.warning(
                    "credential pool: rotation returned the just-marked entry %s — "
                    "treating as no-recovery so the failure surfaces", _label,
                )
                self._current_id = None
                return None
            if next_entry:
                logger.info("credential pool: rotated to %s", next_entry.label or next_entry.id[:8])
            return next_entry

    # ---- leases ------------------------------------------------------------

    def acquire_lease(self, credential_id: Optional[str] = None) -> Optional[str]:
        """Acquire a soft lease on a credential.

        With *credential_id*, lease that entry directly. Otherwise prefer the
        least-leased available credential (priority as tie-breaker); when
        every credential is at the soft cap, still return the least-leased
        one instead of blocking.
        """
        self.environment.require_current_scope()
        chosen_id, pending_refresh = self._acquire_lease_under_lock(credential_id)
        if pending_refresh:
            self._refresh_pending_entries(pending_refresh)
            # Mirror select(): a pool whose entries all needed a deferred
            # refresh must retry once they are back in rotation, or the caller
            # sees "no credentials available" after a successful refresh.
            if chosen_id is None:
                chosen_id, _ = self._acquire_lease_under_lock(credential_id)
        return chosen_id

    def _acquire_lease_under_lock(
        self, credential_id: Optional[str],
    ) -> Tuple[Optional[str], List[PooledCredential]]:
        with self._lock:
            if credential_id:
                self._active_leases[credential_id] = self._active_leases.get(credential_id, 0) + 1
                self._current_id = credential_id
                return credential_id, []

            available, pending_refresh = self._available_entries(clear_expired=True, refresh=True)
            if not available:
                return None, pending_refresh

            below_cap = [e for e in available if self._active_leases.get(e.id, 0) < self._max_concurrent]
            chosen = min(
                below_cap or available,
                key=lambda entry: (self._active_leases.get(entry.id, 0), entry.priority),
            )
            self._active_leases[chosen.id] = self._active_leases.get(chosen.id, 0) + 1
            self._current_id = chosen.id
            return chosen.id, pending_refresh

    def release_lease(self, credential_id: str) -> None:
        self.environment.require_current_scope()
        with self._lock:
            count = self._active_leases.get(credential_id, 0)
            if count <= 1:
                self._active_leases.pop(credential_id, None)
            else:
                self._active_leases[credential_id] = count - 1

    # ---- explicit refresh / admin ------------------------------------------

    def try_refresh_current(self) -> Optional[PooledCredential]:
        self.environment.require_current_scope()
        with self._lock:
            return self._try_refresh_current_unlocked()

    def try_refresh_matching(
        self,
        api_key_hint: Optional[str] = None,
        credential_id: Optional[str] = None,
    ) -> Optional[PooledCredential]:
        """Force-refresh the entry that supplied the failed request.

        Direct integrations may reload the pool after a request failed, so
        ``current_id`` cannot identify the issuing credential. With no hint,
        select WITHOUT the normal proactive refresh: the forced refresh below
        must consume a rotating refresh token exactly once.
        """
        self.environment.require_current_scope()
        with self._lock:
            entry = self._find(lambda e: e.id == credential_id) if credential_id else None
            if entry is None:
                if api_key_hint:
                    entry = self._find(lambda e: e.runtime_api_key == api_key_hint)
                else:
                    entry = self._current_unlocked() or self._select_unlocked(refresh=False, count=False)[0]
            if entry is None:
                return None
            self._current_id = entry.id
            return self._try_refresh_current_unlocked()

    def _try_refresh_current_unlocked(self) -> Optional[PooledCredential]:
        entry = self._current_unlocked()
        if entry is None:
            return None
        refreshed = self._refresh_entry(entry, force=True)
        if refreshed is not None:
            self._current_id = refreshed.id
        return refreshed


# --- Seeding --------------------------------------------------------------


def _upsert_entry(entries: List[PooledCredential], provider: str, source: str, payload: Dict[str, Any]) -> bool:
    matching_indices = [idx for idx, entry in enumerate(entries) if entry.source == source]
    existing_idx = matching_indices[0] if matching_indices else None
    duplicate_indices = set(matching_indices[1:])
    if duplicate_indices:
        entries[:] = [entry for idx, entry in enumerate(entries) if idx not in duplicate_indices]

    if existing_idx is None:
        payload.setdefault("id", uuid.uuid4().hex[:6])
        payload.setdefault("priority", _next_priority(entries))
        payload.setdefault("label", payload.get("label") or source)
        entries.append(PooledCredential.from_dict(provider, payload))
        return True

    existing = entries[existing_idx]
    field_updates: Dict[str, Any] = {}
    extra_updates: Dict[str, Any] = {}
    _field_names = {f.name for f in fields(existing)}
    incoming_token = payload.get("access_token")
    token_changed = incoming_token is not None and incoming_token != existing.access_token
    if token_changed and not existing.access_token:
        # Borrowed sources (claude_code, env-backed rows) are written to
        # auth.json without their secret, so a reloaded entry carries only a
        # ``secret_fingerprint``. Comparing against the empty string reported
        # a rotation on EVERY load and cleared the DEAD/exhausted state the
        # previous process had just persisted. Compare fingerprints instead.
        known_fingerprint = existing.extra.get("secret_fingerprint")
        if isinstance(known_fingerprint, str) and known_fingerprint:
            token_changed = fingerprint_secret_value(incoming_token) != known_fingerprint
    for key, value in payload.items():
        if key in {"id", "priority"} or value is None or (key == "label" and existing.label):
            continue
        if key in _field_names:
            if getattr(existing, key) != value:
                field_updates[key] = value
        elif key in _EXTRA_KEYS and existing.extra.get(key) != value:
            extra_updates[key] = value
    # A rotated token makes the old exhaustion/error state stale.
    if token_changed and existing.last_status is not None:
        field_updates.update(_CLEAR_STATUS)
    if field_updates or extra_updates:
        if extra_updates:
            field_updates["extra"] = {**existing.extra, **extra_updates}
        updated = replace(existing, **field_updates)
        entries[existing_idx] = updated
        # Runtime-only borrowed secret updates refresh the in-memory entry
        # without forcing auth.json churn when the disk-safe payload is
        # unchanged (e.g. env keys with the same fingerprint).
        return bool(duplicate_indices) or existing.to_dict() != updated.to_dict()
    return bool(duplicate_indices)


_ANTHROPIC_SOURCE_RANK = {
    "env:ANTHROPIC_TOKEN": 0,
    "env:CLAUDE_CODE_OAUTH_TOKEN": 1,
    "hermes_pkce": 2,
    "claude_code": 3,
    "env:ANTHROPIC_API_KEY": 4,
}


def _normalize_pool_priorities(provider: str, entries: List[PooledCredential]) -> bool:
    if provider != "anthropic":
        return False
    manual_entries = sorted(
        (entry for entry in entries if _is_manual_source(entry.source)),
        key=lambda entry: entry.priority,
    )
    seeded_entries = sorted(
        (entry for entry in entries if not _is_manual_source(entry.source)),
        key=lambda entry: (
            _ANTHROPIC_SOURCE_RANK.get(entry.source, len(_ANTHROPIC_SOURCE_RANK)),
            entry.priority,
            entry.label,
        ),
    )
    id_to_idx = {entry.id: idx for idx, entry in enumerate(entries)}
    changed = False
    for new_priority, entry in enumerate([*manual_entries, *seeded_entries]):
        if entry.priority != new_priority:
            entries[id_to_idx[entry.id]] = replace(entry, priority=new_priority)
            changed = True
    return changed


def _retain_sources_not_in(entries: List[PooledCredential], drop: Set[str]) -> bool:
    """Remove entries whose source is in *drop*; True if anything was removed."""
    retained = [entry for entry in entries if entry.source not in drop]
    if len(retained) == len(entries):
        return False
    entries[:] = retained
    return True


def load_pool(provider: str, *, environment: PoolEnvironment) -> CredentialPool:
    environment.require_current_scope()
    environment.require_current_scope()
    provider = (provider or "").strip().lower()
    if provider in SINGLE_USE_REFRESH_POOL_PROVIDERS:
        # One-time heal for installs that forked this grant across profiles
        # before the clone-strip / root write-through existed (#100339).
        heal_forked_single_use_oauth_grants(provider)
    raw_entries = read_credential_pool(provider)
    disk_ids = {e.get("id") for e in raw_entries if isinstance(e, dict) and e.get("id")}
    # Ownership (auth.json read) after the heal above; re-read at the tail only if _persist() ran.
    owns_provider: Optional[bool] = None
    changed = any(
        isinstance(payload, dict)
        and sanitize_borrowed_credential_payload(payload, provider) != payload
        for payload in raw_entries
    )
    entries = [PooledCredential.from_dict(provider, payload) for payload in raw_entries]
    raw_needs_auth_normalization = any(
        isinstance(payload, dict)
        and _normalize_pool_auth_type(
            provider,
            payload.get("access_token"),
            payload.get("auth_type", AUTH_TYPE_API_KEY),
        )
        != payload.get("auth_type", AUTH_TYPE_API_KEY)
        for payload in raw_entries
    )
    if raw_needs_auth_normalization:
        # A profile may be reading this provider from the global-root fallback.
        # Keep that fallback read-only: only the owning store may rewrite these
        # rows; loading the default/root profile heals global rows.
        active_pool = _load_auth_store().get("credential_pool")
        active_entries = (
            active_pool.get(provider) if isinstance(active_pool, dict) else None
        )
        changed |= bool(active_entries)

    if provider.startswith(CUSTOM_POOL_PREFIX):
        custom_changed, custom_sources = pool_sources._seed_custom_pool(
            provider, entries, environment=environment
        )
        changed |= custom_changed
        changed |= pool_sources._prune_stale_seeded_entries(entries, custom_sources)
    else:
        singleton_changed, singleton_sources = pool_sources._seed_from_singletons(
            provider, entries, environment=environment
        )
        env_changed, env_sources = pool_sources._seed_from_env(
            provider, entries, environment=environment
        )
        changed |= singleton_changed or env_changed
        # ``load_pool()`` is a non-destructive read for env-seeded entries
        # (#9331); file-backed singletons still prune when their file is gone.
        if provider in SINGLE_USE_REFRESH_POOL_PROVIDERS and disk_ids:
            owns_provider = _profile_owns_pool_provider(provider)
        if owns_provider is False:
            # Rows read through the global-root fallback are seeded from the
            # ROOT's singleton files, which this profile cannot see; pruning
            # them would hide (and, via write-through, delete) the shared
            # grant. The root's own load_pool() prunes.
            borrowed = [e for e in entries if e.id in disk_ids]
            others = [e for e in entries if e.id not in disk_ids]
            changed |= pool_sources._prune_stale_seeded_entries(
                others,
                singleton_sources | env_sources,
                prune_env_sources=False,
            )
            entries[:] = borrowed + others
        else:
            changed |= pool_sources._prune_stale_seeded_entries(
                entries,
                singleton_sources | env_sources,
                prune_env_sources=False,
            )
        changed |= _normalize_pool_priorities(provider, entries)

    pool = CredentialPool(provider, entries, environment=environment)
    pool._persisted_token_pairs = auth_pool_persistence._token_pairs_by_id(raw_entries)
    if changed:
        pool._persist(removed_ids=sorted(disk_ids - {entry.id for entry in entries}))
    # Remember the root's borrowed rows so a later ``add_entry`` in this
    # profile leaves them out of the profile's own store (#100339).
    # No disk rows -> nothing borrowed; the ``set()`` default already applies.
    if provider in SINGLE_USE_REFRESH_POOL_PROVIDERS and disk_ids:
        # Reuse the pre-persist ownership answer unless _persist() just rewrote
        # the store (it can give the profile its own rows); nothing else between
        # the two checks touches auth.json.
        if changed:
            owns_provider = _profile_owns_pool_provider(provider)
        if not owns_provider:
            pool._borrowed_root_ids = set(disk_ids)
    return pool

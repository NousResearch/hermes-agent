"""Persistent multi-credential pool for same-provider failover."""

from __future__ import annotations

import logging
import os
import random
import threading
import time
import uuid
import re
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from hermes_constants import OPENROUTER_BASE_URL
from hermes_cli.config import load_env
from agent.secret_scope import get_secret as _get_secret
from agent.credential_persistence import (
    fingerprint_secret_value,
    is_borrowed_credential_source,
    sanitize_borrowed_credential_payload,
)
import hermes_cli.auth as auth_mod
from hermes_cli.auth import (
    CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    PROVIDER_REGISTRY,
    SINGLE_USE_REFRESH_POOL_PROVIDERS,
    _auth_store_lock,
    _codex_access_token_is_expiring,
    _decode_jwt_claims,
    _global_auth_file_path,
    _load_auth_store,
    _load_provider_state,
    _load_provider_state_with_source,
    _resolve_kimi_base_url,
    _resolve_zai_base_url,
    _same_path,
    _save_auth_store,
    _save_provider_state,
    _store_provider_state,
    read_credential_pool,
    write_credential_pool,
)

logger = logging.getLogger(__name__)


def _load_config_safe() -> Optional[dict]:
    """Load config.yaml read-only, returning None on any error.

    ``load_config_readonly()`` skips the deepcopy ``load_config()`` pays per
    call; the picker calls ``load_pool()`` once per provider row, which made
    that copy the dominant cost of ``model.options``.
    """
    try:
        from hermes_cli.config import load_config_readonly

        return load_config_readonly()
    except Exception:
        return None


def _is_source_suppressed_fn() -> Callable[[str, str], bool]:
    """``hermes_cli.auth.is_source_suppressed`` (late-bound), or an always-False stub."""
    try:
        from hermes_cli.auth import is_source_suppressed
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
    "token_type", "scope", "client_id", "portal_base_url", "obtained_at",
    "expires_in", "agent_key_id", "agent_key_expires_in", "agent_key_reused",
    "agent_key_obtained_at", "tls", "secret_source", "secret_fingerprint",
    # Classified failure semantics for the last exhaustion (agent/error_classifier.py).
    # Providers return 403 for both an edge throttle and a spending limit, so the
    # raw status cannot size a cooldown; persisted so a restart doesn't downgrade
    # a billing bench to a 60s transient cooldown.
    "failure_reason",
})

# Nous singleton metadata mirrored between auth.json state and ``entry.extra``.
_NOUS_EXTRA_STATE_KEYS = (
    "obtained_at", "expires_in", "agent_key_id",
    "agent_key_expires_in", "agent_key_reused", "agent_key_obtained_at",
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
    base_url: Optional[str] = None
    expires_at: Optional[str] = None
    expires_at_ms: Optional[int] = None
    last_refresh: Optional[str] = None
    inference_base_url: Optional[str] = None
    agent_key: Optional[str] = None
    agent_key_expires_at: Optional[str] = None
    request_count: int = 0
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
        data["extra"] = {k: payload[k] for k in _EXTRA_KEYS if payload.get(k) is not None}
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
                    and auth_mod._nous_invoke_jwt_is_usable(
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
        return self.base_url


def label_from_token(token: str, fallback: str) -> str:
    claims = _decode_jwt_claims(token)
    for key in ("email", "preferred_username", "upn"):
        value = claims.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


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


# (regex, seconds-from-match) pairs tried in order against provider error text.
_RETRY_DELAY_PATTERNS: Tuple[Tuple[re.Pattern, Callable[[re.Match], float]], ...] = (
    (
        re.compile(r"quotaResetDelay[:\s\"]+(\d+(?:\.\d+)?)(ms|s)", re.IGNORECASE),
        lambda m: float(m.group(1)) / 1000.0 if m.group(2).lower() == "ms" else float(m.group(1)),
    ),
    (
        re.compile(r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)", re.IGNORECASE),
        lambda m: float(m.group(1)),
    ),
    # "Resets in 4hr 5min" format used by OpenCode Go weekly usage limits
    (
        re.compile(r"resets?\s+in\s+(\d+)\s*hr\s+(\d+)\s*min", re.IGNORECASE),
        lambda m: int(m.group(1)) * 3600 + int(m.group(2)) * 60,
    ),
    (re.compile(r"resets?\s+in\s+(\d+)\s*hr\b", re.IGNORECASE), lambda m: int(m.group(1)) * 3600),
    (re.compile(r"resets?\s+in\s+(\d+)\s*min\b", re.IGNORECASE), lambda m: int(m.group(1)) * 60),
)


def _extract_retry_delay_seconds(message: str) -> Optional[float]:
    if not message:
        return None
    for pattern, to_seconds in _RETRY_DELAY_PATTERNS:
        match = pattern.search(message)
        if match:
            return to_seconds(match)
    return None


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
        retry_delay_seconds = _extract_retry_delay_seconds(message)
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


def _iter_custom_providers(config: Optional[dict] = None):
    """Yield ``(normalized_name, entry)`` from the merged custom-provider config view."""
    if config is None:
        config = _load_config_safe()
    if config is None:
        return
    try:
        from hermes_cli.config import get_compatible_custom_providers

        custom_providers = get_compatible_custom_providers(config)
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
) -> List[str]:
    """Return pool keys to try for a custom endpoint.

    ``hermes auth add <key>`` stores ``providers.<key>`` credentials under the
    durable config slug; older rows and legacy ``custom_providers:`` entries
    live under ``custom:<display-name>``. Try the slug first, then the legacy
    namespace, so a populated pool is not skipped in favour of the
    ``no-key-required`` placeholder.
    """
    if not base_url:
        return []
    normalized_url = _norm_url(base_url)
    requested_aliases = _requested_custom_name_aliases(provider_name) if provider_name else set()

    if requested_aliases:
        for norm_name, entry in _iter_custom_providers():
            if requested_aliases & _custom_entry_name_aliases(norm_name, entry):
                return _pool_keys_for_custom_entry(norm_name, entry)

    for norm_name, entry in _iter_custom_providers():
        entry_url = _norm_url(entry.get("base_url"))
        if entry_url and entry_url == normalized_url:
            return _pool_keys_for_custom_entry(norm_name, entry)
    return []


def get_custom_provider_pool_key(base_url: Optional[str], provider_name: Optional[str] = None) -> Optional[str]:
    """Preferred pool key for a custom provider: durable slug, else ``custom:<name>``.

    When provider_name is given, match by name first so two custom providers
    sharing a base_url keep separate keys.
    """
    candidates = custom_provider_pool_key_candidates(base_url, provider_name)
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


def _get_custom_provider_config(pool_key: str) -> Optional[Dict[str, Any]]:
    """Return the custom_providers config entry matching a pool key like 'custom:together.ai'."""
    if not pool_key.startswith(CUSTOM_POOL_PREFIX):
        return None
    suffix = pool_key[len(CUSTOM_POOL_PREFIX):]
    return next((entry for norm_name, entry in _iter_custom_providers() if norm_name == suffix), None)


def get_pool_strategy(provider: str) -> str:
    """Return the configured selection strategy for a provider."""
    config = _load_config_safe()
    strategies = config.get("credential_pool_strategies") if config else None
    if not isinstance(strategies, dict):
        return STRATEGY_FILL_FIRST
    strategy = str(strategies.get(provider, "") or "").strip().lower()
    return strategy if strategy in SUPPORTED_POOL_STRATEGIES else STRATEGY_FILL_FIRST


def _keyed_custom_pool_matches(
    pool_provider: str,
    provider_norm: str,
    base_url: Optional[str],
) -> bool:
    """Match a durable ``providers.<key>`` pool against runtime identities."""
    runtime_url = _norm_url(base_url)
    if not runtime_url:
        return False
    try:
        for normalized_name, entry in _iter_custom_providers():
            provider_key = _normalize_custom_pool_name(str(entry.get("provider_key") or ""))
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
) -> bool:
    """Match a legacy ``custom:<name>`` pool against a named runtime identity."""
    try:
        for normalized_name, entry in _iter_custom_providers():
            if f"{CUSTOM_POOL_PREFIX}{normalized_name}" != pool_provider:
                continue
            aliases = {normalized_name}
            for value in (entry.get("name"), entry.get("provider_key")):
                alias = _normalize_custom_pool_name(str(value or ""))
                if alias:
                    aliases.add(alias)
                    if alias.startswith(CUSTOM_POOL_PREFIX):
                        aliases.add(alias[len(CUSTOM_POOL_PREFIX):])
            configured_url = _norm_url(entry.get("base_url"))
            runtime_aliases = {_normalize_custom_pool_name(provider_norm)}
            if provider_norm.startswith(CUSTOM_POOL_PREFIX):
                runtime_aliases.add(_normalize_custom_pool_name(provider_norm[len(CUSTOM_POOL_PREFIX):]))
            return bool(runtime_aliases & aliases) and runtime_url == configured_url
    except Exception:
        return False
    return False


def credential_pool_matches_provider(
    pool_or_provider: Any,
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
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
        return _keyed_custom_pool_matches(pool_provider, provider_norm, base_url)
    if provider_norm == "custom":
        try:
            matched_pool = get_custom_provider_pool_key(base_url or "")
            if str(matched_pool or "").strip().lower() == pool_provider:
                return True
            candidates = custom_provider_pool_key_candidates(base_url or "")
        except Exception:
            return False
        return pool_provider in {str(key).strip().lower() for key in candidates}

    runtime_url = _norm_url(base_url)
    if not runtime_url:
        return False
    return _legacy_custom_pool_matches(pool_provider, provider_norm, runtime_url)


def resolve_runtime_pool_key(provider: Optional[str], base_url: Optional[str]) -> str:
    """Resolve the credential-pool key for a runtime provider identity.

    Named custom runtimes retain their configured alias while their pool may
    be stored under the durable ``providers.<key>`` slug or legacy
    ``custom:<name>``. Return that scoped key only when the canonical
    provider/endpoint boundary accepts it; otherwise preserve the normalized
    runtime identity so callers fail closed.
    """
    provider_norm = str(provider or "").strip().lower()
    if not provider_norm:
        return ""

    def _accepts(candidate: str) -> bool:
        return credential_pool_matches_provider(candidate, provider_norm, base_url=base_url)

    try:
        if provider_norm == "custom":
            candidate = get_custom_provider_pool_key(base_url)
            if candidate and _accepts(candidate):
                return str(candidate).strip().lower()
        else:
            # Named/exact custom runtimes are keyed by identity: search the
            # configured candidates by identity before endpoint so a sibling
            # sharing the URL cannot lend its pool.
            for normalized_name, entry in _iter_custom_providers():
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
                if global_path.resolve(strict=False) == real_root.resolve(strict=False):
                    return None
            except Exception:
                return None
    return global_path


def _write_through_provider_state_to_global_root(
    provider_id: str, state: Dict[str, Any]
) -> None:
    """Persist a rotated OAuth ``state`` into the global-root auth.json.

    Best-effort write-through for the multi-profile rotation hazard: nous,
    openai-codex, and xai-oauth rotate the refresh_token on refresh, so when
    a profile pool refresh rotates a grant it resolved from the root fallback,
    the rotated chain must land back in root. Otherwise root keeps a revoked
    refresh token and every other profile dies with ``refresh_token_reused``
    / ``invalid_grant`` once its access token expires.

    Only updates ``providers.<provider_id>`` in the root store; never touches
    the profile store (the caller already saved that). Swallows all errors —
    a failed write-through degrades to root-stale and must never break the
    profile's own successful save. Mirrors
    ``hermes_cli.auth._write_through_xai_oauth_to_global_root``.

    See #48415.
    """
    try:
        global_path = _guarded_global_root(auth_mod._global_auth_file_path())
    except Exception:
        return
    if global_path is None:
        return
    try:
        auth_mod._persist_provider_state_to_store(provider_id, state, global_path, set_active=False)
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("%s pool refresh: write-through to global root failed: %s", provider_id, exc)


def _singleton_target_for_entry(pool: "CredentialPool", entry: "PooledCredential") -> Optional[Path]:
    """Root ``.anthropic_oauth.json`` when *entry* is a borrowed hermes_pkce row, else None."""
    if entry.source != "hermes_pkce" or entry.id not in getattr(pool, "_borrowed_root_ids", ()):
        return None
    try:
        from agent.anthropic_credentials import _root_hermes_oauth_file
        return _root_hermes_oauth_file()
    except Exception:
        return None


def _profile_owns_pool_provider(provider: str) -> bool:
    """True when the ACTIVE auth.json has its own rows for *provider*.

    Named profiles with no local rows read the provider through the
    ``read_credential_pool`` global-root fallback ("borrowing").
    """
    try:
        pool = _load_auth_store().get("credential_pool")
    except Exception:
        return True  # unreadable store: assume ownership, keep legacy path
    entries = pool.get(provider) if isinstance(pool, dict) else None
    return isinstance(entries, list) and bool(entries)


def _borrowed_single_use_pool_root() -> Optional[Path]:
    """Global-root auth.json when persisting a BORROWED single-use pool, else None.

    ``None`` means "persist to the active store as usual": classic mode
    (profile == root), or the profile owns its own rows for this provider.
    """
    try:
        return _guarded_global_root(_global_auth_file_path())
    except Exception:
        return None


def _update_root_pool_rows(provider: str, payloads: List[Dict[str, Any]], global_path: Path) -> None:
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
        incoming_by_id = {p.get("id"): p for p in payloads if isinstance(p, dict) and p.get("id")}
        merged: List[Dict[str, Any]] = []
        changed = False
        for disk_entry in existing_list:
            did = disk_entry.get("id") if isinstance(disk_entry, dict) else None
            incoming = incoming_by_id.get(did) if did else None
            if incoming is None:
                merged.append(disk_entry)
                continue
            updated = auth_mod._merge_disk_cooldown_state(incoming, disk_entry, provider)
            if updated != disk_entry:
                changed = True
            merged.append(updated)
        if changed:
            pool[provider] = merged
            _save_auth_store(store, target_path=global_path)


def persist_pool_entries(
    provider: str,
    payloads: List[Dict[str, Any]],
    *,
    removed_ids: Optional[Iterable[str]] = None,
) -> None:
    """Persist a provider's pool rows to the store that OWNS them.

    A named profile that sees a single-use-refresh provider (Anthropic,
    Codex, xAI OAuth) only through the global-root fallback must not
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
                _update_root_pool_rows(provider, payloads, global_path)
            except Exception as exc:
                # Fail closed on the FORK, not on the save: never fall back to
                # writing a local copy (that IS the bug). The in-memory pool
                # still holds the rotated pair for this process.
                logger.warning(
                    "%s pool: write-through of borrowed root grant failed (%s); "
                    "not materializing a profile-local copy",
                    provider, exc,
                )
            return
    write_credential_pool(provider, payloads, removed_ids=removed_ids)


# --- Per-provider singleton refresh plumbing -------------------------------
#
# Providers whose OAuth singleton lives in auth.json ``providers.<id>.tokens``
# (Codex, xAI): log names (sync-message form, "<name> OAuth" form),
# ``hermes_cli.auth`` refresh function and terminal-error predicate (looked
# up at call time so tests can patch them).
_TOKENS_SINGLETON_PROVIDERS: Dict[str, Tuple[str, str, str, str]] = {
    "openai-codex": ("Codex", "Codex", "refresh_codex_oauth_pure", "_is_terminal_codex_oauth_refresh_error"),
    "xai-oauth": ("xAI OAuth", "xAI", "refresh_xai_oauth_pure", "_is_terminal_xai_oauth_refresh_error"),
}

# Providers whose refresh tokens are single-use: the sync -> POST -> write-back
# sequence must be serialized across processes under the auth-store flock.
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


class CredentialPool:
    def __init__(self, provider: str, entries: List[PooledCredential]):
        self.provider = provider
        self._entries = sorted(entries, key=lambda entry: entry.priority)
        self._current_id: Optional[str] = None
        # Ids of rows read via the global-root fallback (single-use OAuth
        # providers only); set by load_pool(), consumed by add_entry().
        self._borrowed_root_ids: Set[str] = set()
        self._strategy = get_pool_strategy(provider)
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
        with self._lock:
            return bool(self._entries)

    def has_available(self) -> bool:
        """True if at least one entry is not currently in exhaustion cooldown.

        ``_available_entries`` is not read-only (it prunes aged-out DEAD
        manual entries and persists), so it must run under ``self._lock``
        like every other caller or a probe can race a concurrent rotation.
        """
        with self._lock:
            available, _pending = self._available_entries()
            return bool(available)

    def next_available_at(self) -> Optional[float]:
        """Earliest epoch time (seconds) any entry re-enters rotation.

        ``None`` when an entry is available now, or when no exhausted entry
        carries a usable recovery time (empty pool, or only ``STATUS_DEAD``
        entries). Callers must treat ``None`` as "no wait information".
        Runs under ``self._lock`` for the same reason as ``has_available``.
        """
        with self._lock:
            available, _pending = self._available_entries()
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
            return min(candidates) if candidates else None

    def entries(self) -> List[PooledCredential]:
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
        with self._lock:
            return self._current_unlocked()

    def entry_id_for_api_key(self, api_key_hint: Any = None) -> Optional[str]:
        """Stable id for the runtime credential in use.

        Prefer the current selection when it still supplies ``api_key_hint``;
        if the cursor was cleared, fall back to an unambiguous key match.
        """
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

    def _persist(self, *, removed_ids: Optional[List[str]] = None) -> None:
        # Self-locking: snapshotting self._entries must not race a rotation.
        with self._lock:
            persist_pool_entries(
                self.provider,
                [entry.to_dict() for entry in self._entries],
                removed_ids=removed_ids,
            )

    def _adopt(self, entry: PooledCredential, *, persist: bool = True, **updates: Any) -> PooledCredential:
        """``replace(entry, **updates)``, swap it into the pool, optionally persist."""
        updated = replace(entry, **updates)
        self._replace_entry(entry, updated)
        if persist:
            self._persist()
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

    def _sync_anthropic_entry_from_credentials_file(self, entry: PooledCredential) -> PooledCredential:
        """Sync a claude_code entry from ~/.claude/.credentials.json if tokens differ."""
        if self.provider != "anthropic" or entry.source != "claude_code":
            return entry
        try:
            from agent.anthropic_credentials import read_claude_code_credentials
            creds = read_claude_code_credentials()
            if not creds:
                return entry
            file_refresh = creds.get("refreshToken", "")
            file_access = creds.get("accessToken", "")
            # Access tokens can be re-issued without a new refresh token, so
            # checking only refresh_token leaves a stale access_token in the
            # pool -> 401 on every request until the exhausted TTL expires.
            if (file_access or file_refresh) and (
                (file_access and file_access != (entry.access_token or ""))
                or (file_refresh and file_refresh != (entry.refresh_token or ""))
            ):
                logger.debug("Pool entry %s: syncing tokens from credentials file (tokens changed)", entry.id)
                return self._adopt(
                    entry,
                    access_token=file_access or entry.access_token,
                    refresh_token=file_refresh or entry.refresh_token,
                    expires_at_ms=creds.get("expiresAt", 0) or entry.expires_at_ms,
                    **_CLEAR_STATUS,
                )
        except Exception as exc:
            logger.debug("Failed to sync from credentials file: %s", exc)
        return entry

    def _sync_entry_from_pool_store(self, entry: PooledCredential) -> PooledCredential:
        """Adopt a token pair rotated by another pool instance (anthropic, xai-oauth).

        Re-reads the exact persisted row from the credential-pool store while
        the shared cross-process auth-store lock is held. Direct integrations
        load a fresh ``CredentialPool`` per request, so in-memory locks cannot
        protect a single-use refresh token across requests or processes.

        Anthropic borrowed sources (``claude_code``) are excluded: they are
        reference-only rows whose secrets are stripped before reaching
        auth.json, so re-reading yields empty tokens that would be adopted as
        a "rotation" — blanking a usable credential. The singleton file, not
        the pool store, is token authority for those sources; a row with no
        token material at all is refused for the same reason.
        """
        if self.provider not in ("anthropic", "xai-oauth"):
            return entry
        is_anthropic = self.provider == "anthropic"
        if is_anthropic and is_borrowed_credential_source(entry.source, self.provider):
            return entry
        try:
            persisted = next(
                (p for p in read_credential_pool(self.provider) if isinstance(p, dict) and p.get("id") == entry.id),
                None,
            )
            if not isinstance(persisted, dict):
                return entry
            stored = PooledCredential.from_dict(self.provider, persisted)
            if is_anthropic and not (stored.access_token or "").strip() and not (stored.refresh_token or "").strip():
                return entry
            if stored.access_token != entry.access_token or stored.refresh_token != entry.refresh_token:
                logger.debug(
                    "Pool entry %s: adopting %s OAuth tokens rotated by another pool instance",
                    entry.id, "Anthropic" if is_anthropic else "xAI",
                )
                self._replace_entry(entry, stored)
                return stored
        except Exception as exc:
            logger.debug(
                "Failed to sync %s OAuth entry from credential pool: %s",
                "Anthropic" if is_anthropic else "xAI", exc,
            )
        return entry

    _sync_anthropic_entry_from_pool_store = _sync_entry_from_pool_store

    def _sync_entry_from_auth_store(self, entry: PooledCredential) -> PooledCredential:
        """Sync a Codex / xAI device_code entry from auth.json ``providers.<id>.tokens``.

        A fresh ``hermes model`` / ``hermes auth`` login writes new tokens
        under ``_auth_store_lock`` while the pool entry may sit frozen behind
        a ``last_error_reset_at`` hours in the future; without this sync every
        request fails with "no available entries" despite fresh credentials on
        disk. Only singleton-seeded entries apply — env/API-key rows have no
        auth.json shadow.
        """
        spec = _TOKENS_SINGLETON_PROVIDERS.get(self.provider)
        if spec is None:
            return entry
        display = spec[0]
        is_codex = self.provider == "openai-codex"
        sources = ("device_code", "manual:device_code") if is_codex else ("device_code",)
        if entry.source not in sources:
            return entry
        try:
            with _auth_store_lock():
                state = _load_provider_state(_load_auth_store(), self.provider)
            tokens = state.get("tokens") if isinstance(state, dict) else None
            if not isinstance(tokens, dict):
                return entry
            store_access = tokens.get("access_token", "")
            store_refresh = tokens.get("refresh_token", "")
            entry_refresh = entry.refresh_token or ""
            # Adopt when either side differs: a fresh refresh_token from
            # another process means our pair is consumed/stale.
            should_adopt = bool(store_access) and (
                store_access != (entry.access_token or "")
                or (store_refresh and store_refresh != entry_refresh)
            )
            if not should_adopt and is_codex and store_refresh and store_refresh != entry_refresh and not store_access:
                # Store has only a refresh_token — another process rotated the
                # pair and the access_token was consumed. Adopt the
                # refresh_token so we don't replay the consumed one.
                logger.info(
                    "Pool entry %s: auth.json has newer refresh_token "
                    "but no access_token; adopting refresh_token to "
                    "avoid replaying consumed token",
                    entry.id,
                )
                should_adopt = True
            if should_adopt:
                logger.debug(
                    "Pool entry %s: syncing %s tokens from auth.json (refreshed by another process)",
                    entry.id, display,
                )
                field_updates: Dict[str, Any] = {
                    "access_token": store_access or entry.access_token,
                    "refresh_token": store_refresh or entry.refresh_token,
                    **_CLEAR_STATUS,
                }
                if state.get("last_refresh"):
                    field_updates["last_refresh"] = state["last_refresh"]
                return self._adopt(entry, **field_updates)
        except Exception as exc:
            logger.debug("Failed to sync %s entry from auth.json: %s", display, exc)
        return entry

    def _sync_nous_entry_from_auth_store(self, entry: PooledCredential) -> PooledCredential:
        """Sync a Nous device_code entry from auth.json ``providers.nous`` if state differs.

        Another process refreshing via ``resolve_nous_runtime_credentials``
        writes fresh tokens under ``_auth_store_lock``; adopting them avoids a
        "refresh token reuse" revocation on the Nous Portal.
        """
        if self.provider != "nous" or entry.source != "device_code":
            return entry
        try:
            with _auth_store_lock():
                state = _load_provider_state(_load_auth_store(), "nous")
            if not state:
                return entry
            comparable = {
                key: state.get(key)
                for key in (
                    "access_token", "refresh_token", "expires_at",
                    "agent_key", "agent_key_expires_at", "inference_base_url",
                )
            }
            if not any(v not in (None, "") and getattr(entry, k, None) != v for k, v in comparable.items()):
                return entry
            logger.debug("Pool entry %s: syncing Nous state from auth.json", entry.id)
            field_updates: Dict[str, Any] = dict(_CLEAR_STATUS)
            field_updates.update({k: v for k, v in comparable.items() if v})
            extra_updates = dict(entry.extra)
            extra_updates.update(
                {k: state[k] for k in _NOUS_EXTRA_STATE_KEYS if state.get(k) is not None}
            )
            return self._adopt(entry, extra=extra_updates, **field_updates)
        except Exception as exc:
            logger.debug("Failed to sync Nous entry from auth.json: %s", exc)
        return entry

    def _sync_device_code_entry_to_auth_store(self, entry: PooledCredential) -> None:
        """Write refreshed pool entry tokens back to auth.json ``providers.<id>``.

        Otherwise the next ``load_pool()`` re-seeds the stale singleton state
        over the fresh entry — potentially a consumed single-use refresh
        token. Applies to Nous, OpenAI Codex and xAI OAuth singletons.

        ``set_active=False`` everywhere: a sync-back is a token-rotation side
        effect, not the user choosing a provider; ``_save_provider_state``
        would flip ``active_provider`` to whichever provider refreshed last.

        #74339: decide the root write-through on WHERE the state resolved
        from (``_load_provider_state_with_source``), not on whether the
        profile has a ``providers.<id>`` key — ``_store_provider_state``
        creates that key unconditionally, which self-sealed the check after
        the first refresh. When the grant came from the global root, write
        back to root ONLY and skip the profile store so it never accrues a
        shadowing key that blocks both the fallback and the write-through.
        """
        # Only singleton-seeded entries sync back; ``manual:*`` entries are
        # independent credentials and must not write to the singleton.
        if entry.source != "device_code" or self.provider not in ("nous", *_TOKENS_SINGLETON_PROVIDERS):
            return
        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                state, source_path = _load_provider_state_with_source(auth_store, self.provider)
                if not isinstance(state, dict):
                    return
                global_root = _global_auth_file_path()
                is_from_root = bool(
                    source_path is not None and global_root is not None and _same_path(source_path, global_root)
                )
                if not self._apply_entry_to_singleton_state(entry, state):
                    return
                if is_from_root:
                    _write_through_provider_state_to_global_root(self.provider, state)
                else:
                    _store_provider_state(auth_store, self.provider, state, set_active=False)
                    _save_auth_store(auth_store)
        except Exception as exc:
            logger.debug("Failed to sync %s pool entry back to auth store: %s", self.provider, exc)

    def _apply_entry_to_singleton_state(self, entry: PooledCredential, state: Dict[str, Any]) -> bool:
        """Copy *entry*'s tokens into the provider's auth.json ``state`` in place."""
        if self.provider == "nous":
            state["access_token"] = entry.access_token
            for key in ("refresh_token", "expires_at", "agent_key", "agent_key_expires_at"):
                if getattr(entry, key):
                    state[key] = getattr(entry, key)
            for extra_key in _NOUS_EXTRA_STATE_KEYS:
                val = entry.extra.get(extra_key)
                if val is not None:
                    state[extra_key] = val
            if entry.inference_base_url:
                state["inference_base_url"] = entry.inference_base_url
            return True
        tokens = state.get("tokens")
        if not isinstance(tokens, dict):
            return False
        tokens["access_token"] = entry.access_token
        if entry.refresh_token:
            tokens["refresh_token"] = entry.refresh_token
        if entry.last_refresh:
            state["last_refresh"] = entry.last_refresh
        return True

    # ---- refresh -----------------------------------------------------------

    def _refresh_entry(self, entry: PooledCredential, *, force: bool) -> Optional[PooledCredential]:
        if entry.auth_type != AUTH_TYPE_OAUTH or not entry.refresh_token:
            if force:
                self._mark_exhausted(entry, None)
            return None
        if self.provider not in _SINGLE_USE_REFRESH_PROVIDERS:
            return self._refresh_entry_impl(entry, force=force)

        # Single-use refresh tokens: sync -> POST -> write-back must be atomic
        # across Hermes processes, or two processes adopt the same on-disk
        # token, both POST it, and the loser gets ``refresh_token_reused`` /
        # ``invalid_grant`` (for Anthropic sources other than claude_code
        # there was no recovery path at all). Serialize through the shared
        # cross-process auth-store flock; a waiter's in-lock re-sync picks up
        # the winner's rotated token and skips the POST.
        with _auth_store_lock(timeout_seconds=self._single_use_refresh_lock_timeout()):
            if self.provider == "openai-codex":
                synced = self._sync_entry_from_auth_store(entry)
                if synced is not entry and not force and not self._entry_needs_refresh(synced):
                    return synced
                return self._refresh_entry_impl(synced, force=force)
            synced = self._sync_entry_from_pool_store(entry)
            if self.provider == "anthropic" and synced.source == "claude_code":
                # claude_code entries are NOT profile-owned: the refresh token
                # lives in one shared ~/.claude/.credentials.json (or Keychain)
                # every profile reads. The profile-scoped lock above only covers
                # THIS profile's auth.json, so take the dedicated shared-file
                # lock (inner, per the ordering invariant on ``_auth_store_lock``)
                # and re-read that authoritative file before any
                # adopt-and-return shortcut fires. The official ``claude`` CLI
                # rotating out-of-band is handled by the sync-and-retry-once
                # fallback in ``_recover_failed_refresh``.
                with self._claude_code_credentials_lock():
                    synced = self._sync_anthropic_entry_from_credentials_file(synced)
                    if synced.refresh_token != entry.refresh_token:
                        return synced
                    return self._refresh_entry_impl(synced, force=force)
            if synced.access_token != entry.access_token or synced.refresh_token != entry.refresh_token:
                return synced
            return self._refresh_entry_impl(synced, force=force)

    def _claude_code_credentials_lock(self):
        """Cross-process lock keyed to the shared claude_code credentials file.

        Unlike the per-profile ``_auth_store_lock()`` this serializes every
        profile and process that might refresh a ``claude_code`` entry.
        """
        from agent.anthropic_credentials import claude_code_credentials_path

        return _auth_store_lock(
            timeout_seconds=self._single_use_refresh_lock_timeout(),
            target_path=claude_code_credentials_path(),
        )

    def _fail_closed_unpersisted_rotation(
        self,
        entry: PooledCredential,
        exc: BaseException,
        *,
        store: str,
    ) -> None:
        """Quarantine an entry whose rotated pair never reached its store.

        For ``claude_code`` / ``hermes_pkce`` the singleton file — not
        auth.json — is authoritative: ``_seed_from_singletons()`` re-reads it
        on every ``load_pool()``. When the refresh POST succeeded but the
        singleton write failed, the replacement pair exists only in memory
        while the consumed pair survives on disk and would be re-seeded over
        any row we persisted; the next refresh would replay the spent token.
        So never expose or persist the rotated pair; mark the entry terminally
        so it surfaces as an explicit re-auth requirement.
        """
        logger.error(
            "Anthropic %s refresh rotated the single-use token but could not commit it "
            "to %s (%s) — failing closed and quarantining the credential; "
            "re-authenticate to recover",
            entry.source, store, exc,
        )
        try:
            from agent.anthropic_credentials import (
                mark_rotation_consumed_uncommitted,
                spent_rotation_source_path,
            )

            # The singleton still holds the spent pair and load_pool() re-seeds
            # it, so record the fingerprints — persisted to the shared source's
            # sidecar registry (we hold its path-keyed lock here) so OTHER
            # processes/profiles adopt the terminal verdict instead of leasing
            # the stale pair or re-POSTing the spent refresh token.
            mark_rotation_consumed_uncommitted(
                entry.access_token,
                entry.refresh_token,
                source_path=spent_rotation_source_path(entry.source),
            )
        except Exception:  # pragma: no cover - never block the quarantine
            logger.debug("Failed to record consumed rotation fingerprints", exc_info=True)
        self._mark_exhausted(
            entry,
            None,
            {
                "reason": CREDENTIAL_PERSIST_FAILED_REASON,
                "message": f"rotated credential was not durably written to {store}: {exc}",
            },
        )
        return None

    def _single_use_refresh_lock_timeout(self) -> float:
        """Configured refresh POST timeout plus margin, so a slow token endpoint cannot starve the flock."""
        env_var = _REFRESH_TIMEOUT_ENV_VARS.get(self.provider, "HERMES_ANTHROPIC_REFRESH_TIMEOUT_SECONDS")
        refresh_timeout_seconds = auth_mod.env_float(env_var, 20)
        return max(float(auth_mod.AUTH_LOCK_TIMEOUT_SECONDS), float(refresh_timeout_seconds) + 5.0)

    def _commit_anthropic_rotation(
        self, entry: PooledCredential, refreshed: Dict[str, Any]
    ) -> None:
        """Write a rotated Anthropic pair to its authoritative singleton, or fail closed.

        claude_code -> ~/.claude/.credentials.json (so the fallback resolver
        and other profiles see it). hermes_pkce -> ~/.hermes/.anthropic_oauth.json
        (``_seed_from_singletons`` re-seeds it every load; a borrowed row commits
        to the ROOT's file, never a new profile-local copy, #100339). Not
        ``endswith``: manual:hermes_pkce is pool-owned and a singleton for it
        would be a second authority for the same refresh-token family.
        """
        if entry.source == "claude_code":
            store = "~/.claude/.credentials.json"
        elif entry.source == "hermes_pkce":
            store = "~/.hermes/.anthropic_oauth.json"
        else:
            return
        try:
            from agent import anthropic_credentials as ac
            args = (refreshed["access_token"], refreshed["refresh_token"], refreshed["expires_at_ms"])
            if entry.source == "claude_code":
                ac._write_claude_code_credentials(*args)
            else:
                ac._write_hermes_oauth_credentials(*args, target=_singleton_target_for_entry(self, entry))
        except Exception as wexc:
            # Authoritative commit failed: do not mark, persist or return the
            # rotation as successful, and bypass the re-POST recovery path —
            # there is nothing left to retry with.
            raise _RefreshDone(self._fail_closed_unpersisted_rotation(entry, wexc, store=store))

    def _refresh_anthropic(self, entry: PooledCredential) -> PooledCredential:
        """POST the Anthropic refresh, commit to the singleton, return the rotated (unpersisted) entry."""
        from agent.anthropic_credentials import (
            is_rotation_consumed_uncommitted,
            refresh_anthropic_oauth_pure,
            spent_rotation_source_path,
        )

        # Never POST a refresh token another process already spent: the
        # durable sidecar verdict is what a fresh interpreter sees here.
        source_path = spent_rotation_source_path(entry.source)
        if is_rotation_consumed_uncommitted(entry.refresh_token, source_path=source_path) or (
            is_rotation_consumed_uncommitted(entry.access_token, source_path=source_path)
        ):
            raise _RefreshDone(self._fail_closed_unpersisted_rotation(
                entry,
                RuntimeError(
                    "credential pair was rotated by another process but the "
                    "rotation never committed (spent-rotation sidecar verdict)"
                ),
                store=str(source_path or "credential store"),
            ))
        refreshed = refresh_anthropic_oauth_pure(entry.refresh_token, use_json=entry.source.endswith("hermes_pkce"))
        updated = replace(
            entry,
            access_token=refreshed["access_token"],
            refresh_token=refreshed["refresh_token"],
            expires_at_ms=refreshed["expires_at_ms"],
        )
        self._commit_anthropic_rotation(entry, refreshed)
        return updated

    def _post_tokens_refresh(self, entry: PooledCredential) -> PooledCredential:
        """Codex / xAI: POST the refresh and return the rotated (unpersisted) entry."""
        refresh_fn_name = _TOKENS_SINGLETON_PROVIDERS[self.provider][2]
        refreshed = getattr(auth_mod, refresh_fn_name)(entry.access_token, entry.refresh_token)
        return replace(
            entry,
            access_token=refreshed["access_token"],
            refresh_token=refreshed["refresh_token"],
            last_refresh=refreshed.get("last_refresh"),
        )

    def _refresh_entry_impl(self, entry: PooledCredential, *, force: bool) -> Optional[PooledCredential]:
        # Single-use-token providers adopt fresher tokens from their store
        # BEFORE spending the refresh_token; ``entry`` is rebound to the synced
        # row so the failure path below recovers against the pair we POSTed.
        try:
            if self.provider == "anthropic":
                updated = self._refresh_anthropic(entry)
            elif self.provider in _TOKENS_SINGLETON_PROVIDERS:
                entry = self._sync_entry_from_auth_store(entry)
                updated = self._post_tokens_refresh(entry)
            elif self.provider == "nous":
                stale_key = entry.runtime_api_key or entry.agent_key or entry.access_token
                synced = self._sync_nous_entry_from_auth_store(entry)
                if synced is not entry:
                    entry = synced
                    # A peer already rotated and persisted a usable key: adopt
                    # it without consuming the single-use refresh token again.
                    if force and entry.runtime_api_key and entry.runtime_api_key != stale_key:
                        logger.debug("Nous entry %s: adopting peer-rotated token, skipping refresh", entry.id)
                        return entry
                auth_mod.resolve_nous_runtime_credentials(force_refresh=force, stale_access_token=stale_key or None)
                updated = self._sync_nous_entry_from_auth_store(entry)
            else:
                return entry
        except _RefreshDone as done:
            return done.result
        except Exception as exc:
            logger.debug("Credential refresh failed for %s/%s: %s", self.provider, entry.id, exc)
            return self._recover_failed_refresh(entry, exc)

        updated = replace(updated, **_MARK_OK)
        self._replace_entry(entry, updated)
        self._persist()
        # Sync back so _seed_from_singletons() on the next load_pool() sees
        # fresh state instead of re-seeding consumed tokens.
        self._sync_device_code_entry_to_auth_store(updated)
        return updated

    def _recover_failed_refresh(self, entry: PooledCredential, exc: Exception) -> Optional[PooledCredential]:
        """After a failed refresh POST: adopt a peer's rotation, quarantine a dead grant, or bench.

        Another process may have consumed the refresh token between our
        pre-POST sync and the HTTP call; re-read the provider's token
        authority once more and adopt fresher tokens before giving up.
        """
        if self.provider == "anthropic":
            if entry.source == "claude_code":
                synced = self._sync_anthropic_entry_from_credentials_file(entry)
                if synced.refresh_token != entry.refresh_token:
                    logger.debug("Retrying refresh with synced token from credentials file")
                    try:
                        from agent.anthropic_credentials import refresh_anthropic_oauth_pure
                        refreshed = refresh_anthropic_oauth_pure(
                            synced.refresh_token, use_json=synced.source.endswith("hermes_pkce"),
                        )
                        # Commit to the authoritative singleton BEFORE marking or
                        # persisting the pool row, or a failed write leaves an
                        # "ok" row that the next load_pool() re-seeds over.
                        self._commit_anthropic_rotation(synced, refreshed)
                        return self._adopt(
                            synced,
                            access_token=refreshed["access_token"],
                            refresh_token=refreshed["refresh_token"],
                            expires_at_ms=refreshed["expires_at_ms"],
                            last_status=STATUS_OK,
                            last_status_at=None,
                            last_error_code=None,
                        )
                    except _RefreshDone as done:
                        return done.result
                    except Exception as retry_exc:
                        logger.debug("Retry refresh also failed: %s", retry_exc)
                elif not self._entry_needs_refresh(synced):
                    logger.debug("Credentials file has valid token, using without refresh")
                    return synced
            else:
                # Backstop for pool-owned sources (hermes_pkce, manual:dashboard_pkce):
                # the winner may have persisted between our pre-check and our POST.
                synced = self._sync_entry_from_pool_store(entry)
                if synced.refresh_token != entry.refresh_token:
                    logger.debug("Anthropic OAuth refresh failed but pool store has newer tokens — adopting")
                    return self._adopt(synced, **_MARK_OK)
        elif self.provider in _TOKENS_SINGLETON_PROVIDERS:
            _, display, _, terminal_fn_name = _TOKENS_SINGLETON_PROVIDERS[self.provider]
            synced = self._sync_entry_from_auth_store(entry)
            if synced.refresh_token != entry.refresh_token:
                logger.debug("%s OAuth refresh failed but auth.json has newer tokens — adopting", display)
                return self._adopt(synced, **_MARK_OK)
            # Terminal error with no newer tokens: the stored refresh_token is
            # dead. Clear it from auth.json so the next session does not
            # re-seed the revoked credentials, and drop singleton-seeded
            # entries from the pool (mirrors the Nous quarantine path).
            if getattr(auth_mod, terminal_fn_name)(exc):
                logger.debug("%s OAuth refresh token is terminally invalid; clearing local token state", display)
                self._clear_terminal_tokens_state(entry, exc)
                self._quarantine_sources(entry, {"device_code"})
                return None
        elif self.provider == "nous":
            synced = self._sync_nous_entry_from_auth_store(entry)
            if synced.refresh_token != entry.refresh_token:
                logger.debug("Nous refresh failed but auth.json has newer tokens — adopting")
                updated = self._adopt(synced, **_MARK_OK)
                self._sync_device_code_entry_to_auth_store(updated)
                return updated
            if isinstance(exc, TimeoutError):
                # Lost the auth-store lock race under heavy fan-out. That says
                # nothing about the credential — benching it here emptied the
                # pool for ~120 sessions ("matched no nous entry ... pool size
                # 0"). The caller's retry re-syncs once the winner persisted.
                logger.debug("Nous refresh skipped: auth store lock busy; not benching entry")
                return entry
            if auth_mod._is_terminal_nous_refresh_error(exc):
                logger.debug("Nous refresh token is terminally invalid; clearing local token state")
                self._clear_terminal_nous_state(entry, exc)
                self._quarantine_sources(
                    entry,
                    {auth_mod.NOUS_DEVICE_CODE_SOURCE, f"manual:{auth_mod.NOUS_DEVICE_CODE_SOURCE}"},
                )
                return None
        self._mark_exhausted(entry, None)
        return None

    def _clear_terminal_tokens_state(self, entry: PooledCredential, exc: Exception) -> None:
        """Drop the dead Codex/xAI token pair from auth.json unless a peer already rotated it."""
        display = _TOKENS_SINGLETON_PROVIDERS[self.provider][1]
        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                state = _load_provider_state(auth_store, self.provider) or {}
                tokens = (state.get("tokens") or {}) if isinstance(state, dict) else None
                if isinstance(tokens, dict):
                    store_refresh = str(tokens.get("refresh_token") or "").strip()
                    if not store_refresh or store_refresh == str(entry.refresh_token or "").strip():
                        tokens.pop("access_token", None)
                        tokens.pop("refresh_token", None)
                        state["tokens"] = tokens
                        state["last_auth_error"] = {
                            "provider": self.provider,
                            "code": getattr(exc, "code", "unknown"),
                            "message": str(exc),
                            "reason": "credential_pool_refresh_failure",
                            "relogin_required": True,
                            "at": datetime.now(timezone.utc).isoformat(),
                        }
                        _save_provider_state(auth_store, self.provider, state)
                        _save_auth_store(auth_store)
        except Exception as clear_exc:
            logger.debug("Failed to clear terminal %s OAuth state: %s", display, clear_exc)

    def _clear_terminal_nous_state(self, entry: PooledCredential, exc: Exception) -> None:
        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                state = _load_provider_state(auth_store, "nous") or {
                    "client_id": entry.client_id,
                    "portal_base_url": entry.portal_base_url,
                    "inference_base_url": entry.inference_base_url,
                    "token_type": entry.token_type,
                    "scope": entry.scope,
                    "tls": entry.tls,
                }
                store_refresh = str(state.get("refresh_token") or "").strip()
                if not store_refresh or store_refresh == str(entry.refresh_token or "").strip():
                    auth_mod._quarantine_nous_oauth_state(state, exc, reason="credential_pool_refresh_failure")
                    auth_mod._quarantine_nous_pool_entries(auth_store, exc, reason="credential_pool_refresh_failure")
                    _save_provider_state(auth_store, "nous", state)
                    _save_auth_store(auth_store)
        except Exception as clear_exc:
            logger.debug("Failed to clear terminal Nous OAuth state: %s", clear_exc)

    def _codex_quota_restored_upstream(self, entry: PooledCredential) -> bool:
        """Live-check whether an exhausted Codex entry's quota reset early.

        A Codex 429 persists a ``last_error_reset_at`` that can be days out
        (weekly windows), but the window can reopen before then (redeemed
        reset, plan upgrade, OpenAI reset) — issue #43747. Only fires for
        429/quota-shaped errors; the probe is throttled per token (5 min) so
        it is safe on the hot selection path.
        """
        if self.provider != "openai-codex" or entry.last_status != STATUS_EXHAUSTED:
            return False
        if not auth_mod._is_codex_rate_limit_shaped(
            entry.last_error_code, entry.last_error_reason, entry.last_error_message,
        ):
            return False
        token = entry.access_token or ""
        if not token:
            return False
        try:
            return bool(auth_mod._probe_codex_quota_restored(token, base_url=entry.base_url))
        except Exception:
            logger.debug("Codex quota-restored probe failed", exc_info=True)
            return False

    def _entry_needs_refresh(self, entry: PooledCredential) -> bool:
        if entry.auth_type != AUTH_TYPE_OAUTH:
            return False
        if self.provider == "anthropic":
            if entry.expires_at_ms is None:
                return False
            return int(entry.expires_at_ms) <= int(time.time() * 1000) + 120_000
        if self.provider == "openai-codex":
            return _codex_access_token_is_expiring(entry.access_token, CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS)
        if self.provider == "xai-oauth":
            return auth_mod._xai_access_token_is_expiring(
                entry.access_token, auth_mod._xai_proactive_refresh_skew_seconds(entry.access_token),
            )
        # Nous refresh can require network access and happens when runtime
        # credentials are actually resolved, not on enumeration/selection.
        return False

    # ---- selection ---------------------------------------------------------

    def select(self) -> Optional[PooledCredential]:
        entry, pending_refresh = self._select_under_lock()
        if pending_refresh:
            self._refresh_pending_entries(pending_refresh)
            # Re-select now that the refreshed entries are back in the pool.
            if entry is None:
                entry, _ = self._select_under_lock()
        if entry is not None:
            self._unmatched_rotation_streak = 0
        return entry

    def _select_under_lock(self) -> Tuple[Optional[PooledCredential], List[PooledCredential]]:
        with self._lock:
            return self._select_unlocked()

    def _refresh_pending_entries(self, pending: List[PooledCredential]) -> None:
        """Refresh deferred single-use-token entries OUTSIDE the pool lock.

        Each refresh takes the cross-process ``_auth_store_lock`` (20+ s
        possible) and merges into the pool through the self-locking mutation
        primitives; failures are silently skipped.
        """
        for entry in pending:
            self._refresh_entry(entry, force=False)

    def _resync_stale_entry(self, entry: PooledCredential) -> PooledCredential:
        """Re-read an exhausted/DEAD singleton-seeded entry from its token authority.

        The user may have re-authed (``hermes model`` / ``hermes auth``, the
        Claude Code CLI, another profile) leaving fresh tokens on disk while
        the pool entry is frozen behind ``last_error_reset_at``.
        """
        if entry.source != _RESYNC_SOURCE.get(self.provider) or entry.last_status not in {STATUS_EXHAUSTED, STATUS_DEAD}:
            return entry
        if self.provider == "anthropic":
            return self._sync_anthropic_entry_from_credentials_file(entry)
        if self.provider == "nous":
            return self._sync_nous_entry_from_auth_store(entry)
        return self._sync_entry_from_auth_store(entry)

    def _available_entries(
        self, *, clear_expired: bool = False, refresh: bool = False,
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

    def _select_unlocked(self, *, refresh: bool = True) -> Tuple[Optional[PooledCredential], List[PooledCredential]]:
        """Select the best available entry; returns ``(entry, pending_refresh)``."""
        available, pending_refresh = self._available_entries(clear_expired=True, refresh=refresh)
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
            # Bump the usage counter so subsequent selections distribute load
            self._current_id = entry.id
            return self._adopt(entry, persist=False, request_count=entry.request_count + 1), pending_refresh
        elif self._strategy == STRATEGY_ROUND_ROBIN and len(available) > 1:
            entry = available[0]
            rotated = [candidate for candidate in self._entries if candidate.id != entry.id]
            rotated.append(replace(entry, priority=len(self._entries) - 1))
            self._entries = [replace(candidate, priority=idx) for idx, candidate in enumerate(rotated)]
            self._persist()
            self._current_id = entry.id
            return self._current_unlocked() or entry, pending_refresh
        else:
            entry = available[0]
        self._current_id = entry.id
        return entry, pending_refresh

    def peek(self) -> Optional[PooledCredential]:
        with self._lock:
            current = self._current_unlocked()
            if current is not None:
                return current
            available, _pending = self._available_entries()
            return available[0] if available else None

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
    ) -> Optional[PooledCredential]:
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
        with self._lock:
            count = self._active_leases.get(credential_id, 0)
            if count <= 1:
                self._active_leases.pop(credential_id, None)
            else:
                self._active_leases[credential_id] = count - 1

    # ---- explicit refresh / admin ------------------------------------------

    def try_refresh_current(self) -> Optional[PooledCredential]:
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
        with self._lock:
            entry = self._find(lambda e: e.id == credential_id) if credential_id else None
            if entry is None:
                if api_key_hint:
                    entry = self._find(lambda e: e.runtime_api_key == api_key_hint)
                else:
                    entry = self._current_unlocked() or self._select_unlocked(refresh=False)[0]
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

    def reset_statuses(self) -> int:
        with self._lock:
            stale = [e for e in self._entries if e.last_status or e.last_status_at or e.last_error_code]
            if stale:
                stale_ids = {e.id for e in stale}
                self._entries = [
                    replace(e, **_CLEAR_STATUS) if e.id in stale_ids else e for e in self._entries
                ]
                self._persist()
            return len(stale)

    def remove_index(self, index: int) -> Optional[PooledCredential]:
        with self._lock:
            if index < 1 or index > len(self._entries):
                return None
            removed = self._entries.pop(index - 1)
            self._entries = [replace(e, priority=p) for p, e in enumerate(self._entries)]
            persist_pool_entries(
                self.provider,
                [entry.to_dict() for entry in self._entries],
                removed_ids=[removed.id],
            )
            if self._current_id == removed.id:
                self._current_id = None
            return removed

    def resolve_target(self, target: Any) -> Tuple[Optional[int], Optional[PooledCredential], Optional[str]]:
        raw = str(target or "").strip()
        if not raw:
            return None, None, "No credential target provided."

        with self._lock:
            for idx, entry in enumerate(self._entries, start=1):
                if entry.id == raw:
                    return idx, entry, None

            label_matches = [
                (idx, entry)
                for idx, entry in enumerate(self._entries, start=1)
                if entry.label.strip().lower() == raw.lower()
            ]
            if len(label_matches) == 1:
                return label_matches[0][0], label_matches[0][1], None
            if len(label_matches) > 1:
                return None, None, f'Ambiguous credential label "{raw}". Use the numeric index or entry id instead.'
            if raw.isdigit():
                index = int(raw)
                if 1 <= index <= len(self._entries):
                    return index, self._entries[index - 1], None
                return None, None, f"No credential #{index}."
            return None, None, f'No credential matching "{raw}".'

    def add_entry(self, entry: PooledCredential) -> PooledCredential:
        with self._lock:
            entry = replace(entry, priority=_next_priority(self._entries))
            self._entries.append(entry)
            borrowed_ids = getattr(self, "_borrowed_root_ids", None)
            if borrowed_ids:
                # ``hermes -p <profile> auth add <single-use provider>``: the
                # profile claims its OWN credential. Persist only profile-owned
                # rows — copying the borrowed root grant alongside would fork
                # its single-use refresh token (#100339). Once the profile owns
                # rows, the root fallback for this provider is shadowed.
                self._entries = [e for e in self._entries if e.id not in borrowed_ids]
                write_credential_pool(self.provider, [e.to_dict() for e in self._entries])
                self._borrowed_root_ids = set()
            else:
                self._persist()
            return entry


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


class _Seeder:
    """Accumulates ``_upsert_entry`` results for one ``load_pool`` seeding pass."""

    def __init__(self, provider: str, entries: List[PooledCredential]):
        self.provider = provider
        self.entries = entries
        self.changed = False
        self.active_sources: Set[str] = set()
        self.is_suppressed = _is_source_suppressed_fn()

    def upsert(self, source: str, payload: Dict[str, Any]) -> bool:
        """Upsert unless suppressed (``hermes auth remove`` must stay stable across loads)."""
        if self.is_suppressed(self.provider, source):
            return False
        self.active_sources.add(source)
        ingested = _upsert_entry(self.entries, self.provider, source, {"source": source, **payload})
        self.changed |= ingested
        return ingested

    @property
    def result(self) -> Tuple[bool, Set[str]]:
        return self.changed, self.active_sources


def _seed_anthropic_singletons(seed: _Seeder) -> None:
    # Only auto-discover external credentials (Claude Code, Hermes PKCE) when
    # the user explicitly configured anthropic; otherwise auxiliary fallback
    # chains would read ~/.claude/.credentials.json without consent (PR #4210).
    try:
        from hermes_cli.auth import is_provider_explicitly_configured
        if not is_provider_explicitly_configured("anthropic"):
            return
    except ImportError:
        pass

    # API-key vs OAuth is a user-visible choice at `hermes setup`. The API-key
    # signal is ANTHROPIC_API_KEY set AND no OAuth env vars (the save_* helpers
    # zero the other side). Then we MUST NOT seed autodiscovered OAuth tokens:
    # rotation on a 401/429 would silently flip the session onto OAuth, which
    # forces the Claude Code identity injection, `mcp_` tool-name rewrite and
    # claude-cli User-Agent the user explicitly opted out of. Prefer
    # ~/.hermes/.env over os.environ, as `_seed_from_env` does.
    _env_file = load_env()

    def _env_val(key: str) -> str:
        return (_env_file.get(key) or _get_secret(key, "") or "").strip()

    anthropic_oauth_env = _env_val("ANTHROPIC_TOKEN") or _env_val("CLAUDE_CODE_OAUTH_TOKEN")
    if _env_val("ANTHROPIC_API_KEY") and not anthropic_oauth_env:
        # Prune stale autodiscovered OAuth entries from a previous OAuth
        # session so a transient 401 cannot revive them.
        seed.changed |= _retain_sources_not_in(seed.entries, {"hermes_pkce", "claude_code"})
        return

    from agent.anthropic_credentials import (
        read_claude_code_credentials,
        read_hermes_oauth_credentials,
    )

    for source_name, creds in (
        ("hermes_pkce", read_hermes_oauth_credentials()),
        ("claude_code", read_claude_code_credentials()),
    ):
        if creds and creds.get("accessToken"):
            seed.upsert(source_name, {
                "auth_type": AUTH_TYPE_OAUTH,
                "access_token": creds.get("accessToken", ""),
                "refresh_token": creds.get("refreshToken"),
                "expires_at_ms": creds.get("expiresAt"),
                "label": label_from_token(creds.get("accessToken", ""), source_name),
            })


def _seed_nous_singleton(seed: _Seeder, auth_store: Dict[str, Any]) -> None:
    state = _load_provider_state(auth_store, "nous")
    has_runtime_material = bool(
        isinstance(state, dict)
        and (str(state.get("access_token") or "").strip() or str(state.get("agent_key") or "").strip())
    )
    if state and not has_runtime_material:
        seed.changed |= _retain_sources_not_in(seed.entries, {"device_code", "manual:device_code"})
    if not (state and has_runtime_material):
        return
    # Prefer a user-supplied label embedded in the singleton state (``hermes
    # auth add nous --label <name>``) over the token-derived fingerprint.
    custom_label = str(state.get("label") or "").strip()
    seed.upsert("device_code", {
        "auth_type": AUTH_TYPE_OAUTH,
        "access_token": state.get("access_token", ""),
        "refresh_token": state.get("refresh_token"),
        "expires_at": state.get("expires_at"),
        "token_type": state.get("token_type"),
        "scope": state.get("scope"),
        "client_id": state.get("client_id"),
        "portal_base_url": state.get("portal_base_url"),
        "inference_base_url": state.get("inference_base_url"),
        "agent_key": state.get("agent_key"),
        "agent_key_expires_at": state.get("agent_key_expires_at"),
        # Refresh timestamps let freshness-sensitive consumers (self-heal
        # hooks, pruning by age) tell just-refreshed credentials from stale
        # ones (#15099).
        **{key: state.get(key) for key in _NOUS_EXTRA_STATE_KEYS},
        "tls": state.get("tls") if isinstance(state.get("tls"), dict) else None,
        "label": custom_label or label_from_token(state.get("access_token", ""), "device_code"),
    })


def _seed_copilot_singleton(seed: _Seeder) -> None:
    # Copilot tokens are resolved dynamically via `gh auth token` or env vars
    # (COPILOT_GITHUB_TOKEN / GH_TOKEN); they don't live in the auth store.
    try:
        from hermes_cli.copilot_auth import (
            COPILOT_ENV_VARS,
            resolve_copilot_token,
            get_copilot_api_token,
        )
        # All-sources gate BEFORE any work: resolve_copilot_token() shells out
        # and the exchange retries 3x with backoff (~35s worst case); a user
        # who suppressed every copilot source must not pay that on every pool
        # load. The source space here matches credential_sources._remove_copilot_gh.
        copilot_sources = ["gh_cli"] + [f"env:{v}" for v in COPILOT_ENV_VARS]
        if all(seed.is_suppressed(seed.provider, s) for s in copilot_sources):
            return
        token, source = resolve_copilot_token()
        if not token:
            return
        # Exact match: a substring test would classify GH_TOKEN/GITHUB_TOKEN
        # as gh_cli and bypass a user's per-env-var suppression.
        source_name = "gh_cli" if source == "gh auth token" else f"env:{source}"
        # Per-source gate BEFORE the (~35s worst case) network exchange.
        if seed.is_suppressed(seed.provider, source_name):
            return
        api_token, enterprise_base_url = get_copilot_api_token(token)
        # get_copilot_api_token falls back to the RAW token when the exchange
        # fails; the Copilot API then routes it to the fallback
        # "copilot-language-server" integrator whose allowlist omits
        # enterprise-only models -> HTTP 400 on every turn. Surface it.
        if api_token == token and not enterprise_base_url:
            logger.warning(
                "Copilot token exchange degraded to RAW token (exchange "
                "unavailable); enterprise-only models may 400 with "
                "model_not_available_for_integrator until exchange recovers."
            )
        pconfig = PROVIDER_REGISTRY.get(seed.provider)
        seed.upsert(source_name, {
            "auth_type": AUTH_TYPE_API_KEY,
            "access_token": api_token,
            "base_url": enterprise_base_url or (pconfig.inference_base_url if pconfig else ""),
            "label": source,
        })
    except Exception as exc:
        logger.debug("Copilot token seed failed: %s", exc)


def _seed_qwen_singleton(seed: _Seeder) -> None:
    # Qwen OAuth tokens live in ~/.qwen/oauth_creds.json (written by the Qwen
    # CLI). refresh_if_expiring=False avoids network calls during pool loading.
    try:
        from hermes_cli.auth import resolve_qwen_runtime_credentials
        creds = resolve_qwen_runtime_credentials(refresh_if_expiring=False)
        token = creds.get("api_key", "")
        if token:
            source_name = creds.get("source", "qwen-cli")
            seed.upsert(source_name, {
                "auth_type": AUTH_TYPE_OAUTH,
                "access_token": token,
                "expires_at_ms": creds.get("expires_at_ms"),
                "base_url": creds.get("base_url", ""),
                "label": creds.get("auth_file", source_name),
            })
    except Exception as exc:
        logger.debug("Qwen OAuth token seed failed: %s", exc)


def _seed_minimax_singleton(seed: _Seeder) -> None:
    # Read the raw auth.json state rather than resolve_minimax_oauth_runtime_credentials,
    # which always refreshes on expiry (surprise network calls during discovery).
    try:
        from hermes_cli.auth import get_provider_auth_state
        state = get_provider_auth_state("minimax-oauth")
        if not (state and state.get("access_token")):
            return
        expires_at_ms = None
        try:
            raw = state.get("expires_at", "")
            if raw:
                expires_at_ms = int(datetime.fromisoformat(raw).timestamp() * 1000)
        except Exception:
            expires_at_ms = None
        seed.upsert("oauth", {
            "auth_type": AUTH_TYPE_OAUTH,
            "access_token": state["access_token"],
            "refresh_token": state.get("refresh_token"),
            "expires_at_ms": expires_at_ms,
            "base_url": str(state.get("inference_base_url", "") or "").rstrip("/"),
            "label": state.get("label", "") or label_from_token(state.get("access_token", ""), "oauth"),
        })
    except Exception as exc:
        logger.debug("MiniMax OAuth token seed failed: %s", exc)


def _seed_tokens_singleton(seed: _Seeder, auth_store: Dict[str, Any]) -> None:
    """Codex / xAI: surface the auth.json ``providers.<id>.tokens`` singleton as ``device_code``.

    Hermes owns its own Codex auth state and does NOT auto-import
    ~/.codex/auth.json: refresh tokens are single-use, so sharing them with
    Codex CLI / VS Code causes refresh_token_reused races. Adoption is an
    explicit one-time prompt via `hermes auth openai-codex`.
    """
    state = _load_provider_state(auth_store, seed.provider)
    tokens = state.get("tokens") if isinstance(state, dict) else None
    if not (isinstance(tokens, dict) and tokens.get("access_token")):
        return
    if seed.provider == "openai-codex":
        base_url = auth_mod.DEFAULT_CODEX_BASE_URL
        custom_label = str(state.get("label") or "").strip()
    else:
        base_url = auth_mod.DEFAULT_XAI_OAUTH_BASE_URL
        custom_label = ""
    seed.upsert("device_code", {
        "auth_type": AUTH_TYPE_OAUTH,
        "access_token": tokens.get("access_token", ""),
        "refresh_token": tokens.get("refresh_token"),
        "base_url": base_url,
        "last_refresh": state.get("last_refresh"),
        "label": custom_label or label_from_token(tokens.get("access_token", ""), "device_code"),
    })


def _seed_from_singletons(provider: str, entries: List[PooledCredential]) -> Tuple[bool, Set[str]]:
    seed = _Seeder(provider, entries)
    auth_store = _load_auth_store()
    if provider == "anthropic":
        _seed_anthropic_singletons(seed)
    elif provider == "nous":
        _seed_nous_singleton(seed, auth_store)
    elif provider == "copilot":
        _seed_copilot_singleton(seed)
    elif provider == "qwen-oauth":
        _seed_qwen_singleton(seed)
    elif provider == "minimax-oauth":
        _seed_minimax_singleton(seed)
    elif provider in _TOKENS_SINGLETON_PROVIDERS:
        # `hermes auth remove openai-codex` suppresses device_code; without
        # this gate the removal is undone on the next load_pool().
        if provider == "openai-codex" and seed.is_suppressed(provider, "device_code"):
            return seed.result
        _seed_tokens_singleton(seed, auth_store)
    return seed.result


def get_env_prefer_dotenv(key: str) -> str:
    """Resolve a credential env var, preferring ~/.hermes/.env over os.environ.

    The user's config file is authoritative; stale env vars from parent
    processes (Codex CLI, test scripts) must not override deliberate .env
    changes. load_env() memoizes on mtime, so per-call reads cost a stat().
    An unresolved ``op://`` reference in .env yields to the already-resolved
    value from the active secret scope (set by apply_onepassword_secrets());
    otherwise every provider auth attempt would receive a URL instead of a key.
    """
    env_file = load_env()
    raw = env_file.get(key, "").strip()
    scoped_value = (_get_secret(key, "") or "").strip()
    if raw.startswith("op://") and scoped_value:
        return scoped_value
    return raw or scoped_value


# Providers already warned about env-key -> pool ingestion, once per process
# (#81952 expected-behavior #3).
_ENV_INGESTION_WARNED: Set[str] = set()


def _warn_env_ingestion_once(provider: str, env_var: str) -> None:
    """WARN once per process per provider when an env credential is ingested into a paid pool.

    Auto-ingesting OPENROUTER_API_KEY is what ARMS silent OpenRouter spend —
    every downstream auto-detect keys off the pool having credentials.
    Ingestion stays allowed (an exported key is arguable intent) but must
    never be silent.
    """
    if provider in _ENV_INGESTION_WARNED:
        return
    _ENV_INGESTION_WARNED.add(provider)
    logger.warning(
        "Ingested %s from environment into the %s credential pool — this "
        "enables %s spend. Remove the key or run "
        "hermes auth remove %s <n> to suppress.",
        env_var,
        provider,
        "OpenRouter" if provider == "openrouter" else provider,
        provider,
    )


def _env_payload(*, env_var: str, token: str, base_url: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "auth_type": AUTH_TYPE_API_KEY,
        "access_token": token,
        "base_url": base_url,
        "label": env_var,
    }
    try:
        from hermes_cli.env_loader import get_secret_source
        source_label = get_secret_source(env_var)
    except Exception:
        source_label = None
    secret_source = str(source_label).strip() if source_label else None
    if secret_source:
        payload["secret_source"] = secret_source
    return payload


# Region-specific endpoints inferred from the key itself.
_ENV_BASE_URL_RESOLVERS = {
    "kimi-coding": _resolve_kimi_base_url,
    "zai": _resolve_zai_base_url,
}


def _seed_from_env(provider: str, entries: List[PooledCredential]) -> Tuple[bool, Set[str]]:
    seed = _Seeder(provider, entries)
    # Copilot's singleton branch exchanges the raw ghu_ OAuth token for the
    # api token via `get_copilot_api_token`; the generic loop would re-read
    # COPILOT_GITHUB_TOKEN and overwrite it with the RAW token, causing 400s
    # ("not available for integrator copilot-language-server").
    if provider == "copilot":
        return seed.result

    if provider == "openrouter":
        token = get_env_prefer_dotenv("OPENROUTER_API_KEY")
        if token and seed.upsert(
            "env:OPENROUTER_API_KEY",
            _env_payload(env_var="OPENROUTER_API_KEY", token=token, base_url=OPENROUTER_BASE_URL),
        ):
            _warn_env_ingestion_once(provider, "OPENROUTER_API_KEY")
        return seed.result

    pconfig = PROVIDER_REGISTRY.get(provider)
    if not pconfig or pconfig.auth_type != AUTH_TYPE_API_KEY:
        return seed.result

    env_url = ""
    if pconfig.base_url_env_var:
        env_url = get_env_prefer_dotenv(pconfig.base_url_env_var).rstrip("/")

    env_vars = list(pconfig.api_key_env_vars)
    if provider == "anthropic":
        env_vars = ["ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]

    # Also seed any env-source pool entries already on disk that aren't in
    # the registry's tuple. A user can put `source: env:PROVIDER_API_KEY_2`
    # in auth.json expecting it to be picked up from the environment; the
    # registry only declares the primary env var, so this loop fills the
    # gap and keeps multi-key rotation working without per-provider code.
    for entry in entries:
        if entry.source.startswith("env:"):
            env_name = entry.source.split(":", 1)[1]
            if env_name and env_name not in env_vars:
                env_vars.append(env_name)

    resolve_base_url = _ENV_BASE_URL_RESOLVERS.get(provider)
    for env_var in env_vars:
        token = get_env_prefer_dotenv(env_var)
        if not token:
            continue
        base_url = env_url or pconfig.inference_base_url
        if resolve_base_url is not None:
            base_url = resolve_base_url(token, pconfig.inference_base_url, env_url)
        seed.upsert(f"env:{env_var}", _env_payload(env_var=env_var, token=token, base_url=base_url))
    return seed.result


def _prune_stale_seeded_entries(
    entries: List[PooledCredential],
    active_sources: Set[str],
    *,
    prune_env_sources: bool = True,
) -> bool:
    def _is_prunable(entry: PooledCredential) -> bool:
        # ``env:*`` entries are persisted references re-hydrated on every load.
        # A process that merely lacks the env var must NOT delete the on-disk
        # entry for every other process (#9331); prune only when explicitly
        # requested (an `hermes auth` command that confirmed the source is gone).
        if entry.source.startswith("env:"):
            return prune_env_sources
        # File-backed singletons and Hermes PKCE disappear when their backing file is gone.
        return is_borrowed_credential_source(entry.source, entry.provider) or entry.source == "hermes_pkce"

    retained = [
        entry
        for entry in entries
        if _is_manual_source(entry.source) or entry.source in active_sources or not _is_prunable(entry)
    ]
    if len(retained) == len(entries):
        return False
    entries[:] = retained
    return True


def _seed_custom_pool(pool_key: str, entries: List[PooledCredential]) -> Tuple[bool, Set[str]]:
    """Seed a custom endpoint pool from custom_providers config and model config."""
    seed = _Seeder(pool_key, entries)

    cp_config = _get_custom_provider_config(pool_key)
    if cp_config:
        api_key = str(cp_config.get("api_key") or "").strip()
        name = str(cp_config.get("name") or "").strip()
        if api_key:
            seed.upsert(f"config:{name}", {
                "auth_type": AUTH_TYPE_API_KEY,
                "access_token": api_key,
                "base_url": _norm_url(cp_config.get("base_url")),
                "label": name or f"config:{name}",
            })

    # Seed from model.api_key when model.provider=='custom' and model.base_url matches
    try:
        config = _load_config_safe()
        model_cfg = config.get("model") if config else None
        if isinstance(model_cfg, dict):
            model_provider = str(model_cfg.get("provider") or "").strip().lower()
            model_base_url = _norm_url(model_cfg.get("base_url"))
            model_api_key = next(
                (v.strip() for k in ("api_key", "api") for v in (model_cfg.get(k),) if isinstance(v, str) and v.strip()),
                "",
            )
            if model_provider == "custom" and model_base_url and model_api_key:
                # The pool may be keyed under the durable ``providers.<key>``
                # slug or legacy ``custom:<name>``; accept any candidate, or
                # seeding is skipped when the pool holds the other identity.
                # Check if this model's base_url matches our custom provider. See #100413.
                matched_keys = {
                    str(key).strip().lower() for key in custom_provider_pool_key_candidates(model_base_url)
                }
                if pool_key in matched_keys:
                    seed.upsert("model_config", {
                        "auth_type": AUTH_TYPE_API_KEY,
                        "access_token": model_api_key,
                        "base_url": model_base_url,
                        "label": "model_config",
                    })
    except Exception:
        pass

    return seed.result


def load_pool(provider: str) -> CredentialPool:
    provider = (provider or "").strip().lower()
    if provider in SINGLE_USE_REFRESH_POOL_PROVIDERS:
        # One-time heal for installs that forked this grant across profiles
        # before the clone-strip / root write-through existed (#100339).
        auth_mod.heal_forked_single_use_oauth_grants(provider)
    raw_entries = read_credential_pool(provider)
    disk_ids = {e.get("id") for e in raw_entries if isinstance(e, dict) and e.get("id")}
    changed = any(
        isinstance(payload, dict) and sanitize_borrowed_credential_payload(payload, provider) != payload
        for payload in raw_entries
    )
    entries = [PooledCredential.from_dict(provider, payload) for payload in raw_entries]
    raw_needs_auth_normalization = any(
        isinstance(payload, dict)
        and _normalize_pool_auth_type(
            provider, payload.get("access_token"), payload.get("auth_type", AUTH_TYPE_API_KEY),
        ) != payload.get("auth_type", AUTH_TYPE_API_KEY)
        for payload in raw_entries
    )
    if raw_needs_auth_normalization:
        # A profile may be reading this provider from the global-root fallback.
        # Keep that fallback read-only: only the owning store may rewrite these
        # rows; loading the default/root profile heals global rows.
        active_pool = _load_auth_store().get("credential_pool")
        active_entries = active_pool.get(provider) if isinstance(active_pool, dict) else None
        changed |= bool(active_entries)

    if provider.startswith(CUSTOM_POOL_PREFIX):
        custom_changed, custom_sources = _seed_custom_pool(provider, entries)
        changed |= custom_changed
        changed |= _prune_stale_seeded_entries(entries, custom_sources)
    else:
        singleton_changed, singleton_sources = _seed_from_singletons(provider, entries)
        env_changed, env_sources = _seed_from_env(provider, entries)
        changed |= singleton_changed or env_changed
        # ``load_pool()`` is a non-destructive read for env-seeded entries
        # (#9331); file-backed singletons still prune when their file is gone.
        borrowing_root_grant = (
            provider in SINGLE_USE_REFRESH_POOL_PROVIDERS
            and bool(disk_ids)
            and not _profile_owns_pool_provider(provider)
        )
        if borrowing_root_grant:
            # Rows read through the global-root fallback are seeded from the
            # ROOT's singleton files, which this profile cannot see; pruning
            # them would hide (and, via write-through, delete) the shared
            # grant. The root's own load_pool() prunes.
            borrowed = [e for e in entries if e.id in disk_ids]
            others = [e for e in entries if e.id not in disk_ids]
            changed |= _prune_stale_seeded_entries(
                others, singleton_sources | env_sources, prune_env_sources=False,
            )
            entries[:] = borrowed + others
        else:
            changed |= _prune_stale_seeded_entries(
                entries, singleton_sources | env_sources, prune_env_sources=False,
            )
        changed |= _normalize_pool_priorities(provider, entries)

    if changed:
        new_ids = {entry.id for entry in entries}
        persist_pool_entries(
            provider,
            [entry.to_dict() for entry in sorted(entries, key=lambda item: item.priority)],
            removed_ids=disk_ids - new_ids,
        )
    pool = CredentialPool(provider, entries)
    # Remember the root's borrowed rows so a later ``add_entry`` in this
    # profile leaves them out of the profile's own store (#100339).
    if provider in SINGLE_USE_REFRESH_POOL_PROVIDERS and not _profile_owns_pool_provider(provider):
        pool._borrowed_root_ids = set(disk_ids)
    return pool

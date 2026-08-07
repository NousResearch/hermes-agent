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
from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_constants import OPENROUTER_BASE_URL
from hermes_cli.config import load_env
from agent.secret_scope import get_secret as _get_secret
from agent.credential_persistence import (
    is_borrowed_credential_source,
    sanitize_borrowed_credential_payload,
)
import hermes_cli.auth as auth_mod
from hermes_cli.auth import (
    CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    PROVIDER_REGISTRY,
    _auth_store_lock,
    _codex_access_token_is_expiring,
    _decode_jwt_claims,
    _load_auth_store,
    _load_provider_state,
    _load_provider_state_with_source,
    _resolve_kimi_base_url,
    _resolve_zai_base_url,
    _save_auth_store,
    _save_provider_state,
    _store_provider_state,
    read_credential_pool,
    write_credential_pool,
)

logger = logging.getLogger(__name__)


def _load_config_safe() -> Optional[dict]:
    """Load config.yaml read-only, returning None on any error.

    Uses ``load_config_readonly()``: every consumer in this module only reads
    (``get_pool_strategy``, ``_iter_custom_providers``, the model-config seed),
    and the deepcopy that ``load_config()`` pays per call is what made
    credential-pool checks the dominant cost of ``model.options`` — the picker
    calls ``load_pool()`` once per provider row, each of which loaded (and
    deep-copied) the full config again.
    """
    try:
        from hermes_cli.config import load_config_readonly

        return load_config_readonly()
    except Exception:
        return None


# --- Status and type constants ---

STATUS_OK = "ok"
STATUS_EXHAUSTED = "exhausted"
# Terminal failure — the credential will never recover on its own.  Used for
# upstream-permanent OAuth states like ``token_invalidated`` / ``token_revoked``
# where retrying after a TTL cooldown is guaranteed to fail.  ``DEAD`` entries
# are excluded from rotation unconditionally and only clear when an explicit
# write-side sync (e.g. ``_save_codex_tokens`` after a fresh device-code
# login) rewrites the tokens.
STATUS_DEAD = "dead"

# OAuth error reasons that indicate the credential is permanently invalid
# server-side and cannot be recovered by retry/refresh.  Sourced from
# OpenAI Codex Responses API, Anthropic, xAI, and Google OAuth spec.
_TERMINAL_AUTH_REASONS = frozenset({
    "token_invalidated",   # OpenAI Codex: "Your authentication token has been invalidated."
    "token_revoked",        # OAuth 2.0 RFC 7009: token explicitly revoked
    "invalid_token",        # RFC 6750: bearer token is malformed/expired/revoked
    "invalid_grant",        # RFC 6749: refresh_token rejected during refresh
    "unauthorized_client",  # RFC 6749: client no longer authorized
    "refresh_token_reused", # Single-use refresh token consumed by another process
})

# How long a DEAD manual credential is preserved before being pruned.
# Manual entries (``manual:*``) are independent credentials with no singleton
# to re-seed from, so pruning them after a quiet window cleans up dead state
# without losing recoverability — the user always has the option to re-add
# via ``hermes auth add``.
#
# Singleton-seeded entries (``device_code``, ``claude_code``)
# are NOT pruned because ``_seed_from_singletons`` would just re-create them
# on the next ``load_pool()`` with the same stale singleton tokens, defeating
# the cleanup.  They remain in the pool marked DEAD until an explicit re-auth
# write-side sync (``_save_codex_tokens`` etc.) clears the status.
DEAD_MANUAL_PRUNE_TTL_SECONDS = 24 * 60 * 60  # 24 hours

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

# Cooldown before retrying an exhausted credential.
# Transient 401 auth failures cool down briefly so single-key setups can recover.
# 429 (rate-limited), 402 (billing/quota), and other failures cool down after 1 hour.
# Provider-supplied reset_at timestamps override these defaults.
EXHAUSTED_TTL_401_SECONDS = 5 * 60           # 5 minutes
EXHAUSTED_TTL_429_SECONDS = 60 * 60          # 1 hour
EXHAUSTED_TTL_DEFAULT_SECONDS = 60 * 60      # 1 hour
# When a pool has no other credential to rotate to (the offending key is the
# sole non-DEAD entry), a 1-hour bench means an hour of hard failures with
# nothing to fall back to. Throttles (429/403/5xx) are transient and reset in
# seconds, so a sole credential cools down briefly instead — same rationale as
# the short 401 cooldown above. Provider-supplied reset_at still overrides.
EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS = 60   # 1 minute

# ``FailoverReason.billing`` as a bare string. The pool stores classified
# failure semantics as plain text (it persists to JSON and must not import
# the classifier), so the value is duplicated here rather than referenced.
FAILURE_REASON_BILLING = "billing"

# Throttle window for the "no available entries" INFO line. Credential
# selection runs on a hot path (every model call, plus auxiliary tasks like
# compression/moa/titles), so when a pool is empty or fully exhausted the
# un-throttled log fires on *every* selection. On Windows several Hermes
# processes share one rotating log guarded by concurrent-log-handler's
# cross-process lock; that per-selection volume storms the lock
# (``RuntimeError: Cannot acquire lock after 20 attempts``), pegs a core, and
# stalls the asyncio event loop long enough to fail the Desktop backend
# readiness handshake ("Timed out connecting to Hermes backend after
# 15000ms"). Logging the condition at most once per window preserves the
# signal while removing the storm — same class of fix as the warn-once
# dedup in #58265.
NO_AVAILABLE_ENTRIES_LOG_THROTTLE_SECONDS = 60.0

# Pool key prefix for custom OpenAI-compatible endpoints.
# Custom endpoints all share provider='custom' but are keyed by their
# custom_providers name: 'custom:<normalized_name>'.
CUSTOM_POOL_PREFIX = "custom:"


# Fields that are only round-tripped through JSON — never used for logic as attributes.
_EXTRA_KEYS = frozenset({
    "token_type", "scope", "client_id", "portal_base_url", "obtained_at",
    "expires_in", "agent_key_id", "agent_key_expires_in", "agent_key_reused",
    "agent_key_obtained_at", "tls", "secret_source", "secret_fingerprint",
    # Classified failure semantics for the last exhaustion, as decided by
    # agent/error_classifier.py. The raw HTTP status is not enough to size a
    # cooldown: providers return 403 for both an edge throttle (transient,
    # seconds) and a spending/key limit (billing, needs a real fix). Persisted
    # with the entry so a restart doesn't downgrade a billing bench back to a
    # 60s transient cooldown.
    "failure_reason",
})


def _normalize_pool_auth_type(provider: str, token: Any, auth_type: Any) -> str:
    """Infer pool auth metadata for token formats with one unambiguous meaning."""
    if (
        provider == "anthropic"
        and isinstance(token, str)
        and token.startswith("sk-ant-oat")
    ):
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
    # Runtime-only owner for a singleton borrowed from another auth store.
    # Never hydrate or serialize this path: it is process-local routing state,
    # not part of the public credential-pool schema.
    source_store_path: Optional[Path] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
        self.auth_type = _normalize_pool_auth_type(
            self.provider,
            self.access_token,
            self.auth_type,
        )

    def __getattr__(self, name: str):
        if name in _EXTRA_KEYS:
            return self.extra.get(name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

    @classmethod
    def from_dict(cls, provider: str, payload: Dict[str, Any]) -> "PooledCredential":
        field_names = {
            f.name
            for f in fields(cls)
            if f.name not in {"provider", "source_store_path"}
        }
        data = {k: payload.get(k) for k in field_names if k in payload}
        # Rehydrated last_status_at may be an ISO string from to_dict() — normalize to float epoch
        if "last_status_at" in data and isinstance(data["last_status_at"], str):
            data["last_status_at"] = _parse_absolute_timestamp(data["last_status_at"])
        extra = {k: payload[k] for k in _EXTRA_KEYS if k in payload and payload[k] is not None}
        data["extra"] = extra
        data.setdefault("id", uuid.uuid4().hex[:6])
        data.setdefault("label", payload.get("source", provider))
        data.setdefault("auth_type", AUTH_TYPE_API_KEY)
        data.setdefault("priority", 0)
        data.setdefault("source", SOURCE_MANUAL)
        data.setdefault("access_token", "")
        return cls(provider=provider, **data)

    def to_dict(self) -> Dict[str, Any]:
        _ALWAYS_EMIT = {
            "last_status",
            "last_status_at",
            "last_error_code",
            "last_error_reason",
            "last_error_message",
            "last_error_reset_at",
        }
        result: Dict[str, Any] = {}
        for field_def in fields(self):
            if field_def.name in {"provider", "extra", "source_store_path"}:
                continue
            value = getattr(self, field_def.name)
            if value is not None or field_def.name in _ALWAYS_EMIT:
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
                        token,
                        scope=getattr(self, "scope", None),
                        expires_at=expires_at,
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


@dataclass(frozen=True)
class _TrustedCodexSourceOwner:
    """Pool-private provenance for a root-fallback Codex row."""

    source_path: Path
    owner_kind: str
    entry_id: str
    source: str


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


def _is_source_owned_elsewhere(entry: PooledCredential) -> bool:
    """Return whether an entry is a read-only alias owned by another store."""
    return entry.source_store_path is not None


def _exhausted_ttl(
    error_code: Optional[int],
    *,
    sole_credential: bool = False,
    failure_reason: Optional[str] = None,
) -> int:
    """Return cooldown seconds based on the HTTP status that caused exhaustion.

    When *sole_credential* is True the pool has no other entry to rotate to, so
    a long bench just blocks the only key. Transient throttles (429 and the
    catch-all default, which covers 403/5xx/unknown) are capped to a brief
    cooldown so the sole key can recover — mirroring the short 401 path. 401
    keeps its own (already short) TTL.

    *failure_reason* is the classified semantics from
    ``agent/error_classifier.py``. The raw status alone can't size the
    cooldown: an OpenRouter ``key limit exceeded`` and an xAI spending-limit
    block both arrive as **403** but classify as ``billing``, and a 60s retry
    on a spent account just re-fails every minute. Billing keeps the full
    bench regardless of status; 402 does too, since it is billing by
    definition even when nothing classified it.
    """
    if error_code == 401:
        return EXHAUSTED_TTL_401_SECONDS
    base = EXHAUSTED_TTL_429_SECONDS if error_code == 429 else EXHAUSTED_TTL_DEFAULT_SECONDS
    # Sole credential: shorten only TRANSIENT throttles (429 rate-limit, 403
    # edge-throttle, 5xx server, or unknown). Billing exhaustion — whether
    # classified as such or self-evident from a 402 — is a genuine depletion
    # where a quick retry can't help, so it keeps the full bench.
    is_billing = error_code == 402 or failure_reason == FAILURE_REASON_BILLING
    if sole_credential and not is_billing:
        return min(base, EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS)
    return base


def _parse_absolute_timestamp(value: Any) -> Optional[float]:
    """Best-effort parse for provider reset timestamps.

    Accepts epoch seconds, epoch milliseconds, and ISO-8601 strings.
    Returns seconds since epoch.
    """
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
        except ValueError:
            numeric = None
        if numeric is not None:
            return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _extract_retry_delay_seconds(message: str) -> Optional[float]:
    if not message:
        return None
    delay_match = re.search(r"quotaResetDelay[:\s\"]+(\d+(?:\.\d+)?)(ms|s)", message, re.IGNORECASE)
    if delay_match:
        value = float(delay_match.group(1))
        return value / 1000.0 if delay_match.group(2).lower() == "ms" else value
    sec_match = re.search(r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)", message, re.IGNORECASE)
    if sec_match:
        return float(sec_match.group(1))
    # "Resets in 4hr 5min" format used by OpenCode Go weekly usage limits
    hr_min_match = re.search(r"resets?\s+in\s+(\d+)\s*hr\s+(\d+)\s*min", message, re.IGNORECASE)
    if hr_min_match:
        return int(hr_min_match.group(1)) * 3600 + int(hr_min_match.group(2)) * 60
    hr_only_match = re.search(r"resets?\s+in\s+(\d+)\s*hr\b", message, re.IGNORECASE)
    if hr_only_match:
        return int(hr_only_match.group(1)) * 3600
    min_only_match = re.search(r"resets?\s+in\s+(\d+)\s*min\b", message, re.IGNORECASE)
    if min_only_match:
        return int(min_only_match.group(1)) * 60
    return None


def _normalize_error_context(error_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(error_context, dict):
        return {}
    normalized: Dict[str, Any] = {}
    reason = error_context.get("reason")
    if isinstance(reason, str) and reason.strip():
        normalized["reason"] = reason.strip()
    message = error_context.get("message")
    if isinstance(message, str) and message.strip():
        normalized["message"] = message.strip()
    reset_at = (
        error_context.get("reset_at")
        or error_context.get("resets_at")
        or error_context.get("retry_until")
    )
    parsed_reset_at = _parse_absolute_timestamp(reset_at)
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
    reset_at = _parse_absolute_timestamp(getattr(entry, "last_error_reset_at", None))
    if reset_at is not None:
        return reset_at
    if entry.last_status_at:
        return entry.last_status_at + _exhausted_ttl(
            entry.last_error_code,
            sole_credential=sole_credential,
            failure_reason=getattr(entry, "failure_reason", None),
        )
    return None


def _normalize_custom_pool_name(name: str) -> str:
    """Normalize a custom provider name for use as a pool key suffix."""
    return name.strip().lower().replace(" ", "-")


def _iter_custom_providers(config: Optional[dict] = None):
    """Yield (normalized_name, entry_dict) for each valid custom_providers entry."""
    if config is None:
        config = _load_config_safe()
    if config is None:
        return
    custom_providers = config.get("custom_providers")
    if not isinstance(custom_providers, list):
        # Fall back to the v12+ providers dict via the compatibility layer
        try:
            from hermes_cli.config import get_compatible_custom_providers

            custom_providers = get_compatible_custom_providers(config)
        except Exception:
            return
    if not custom_providers:
        return
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        yield _normalize_custom_pool_name(name), entry


def get_custom_provider_pool_key(base_url: Optional[str], provider_name: Optional[str] = None) -> Optional[str]:
    """Look up the custom_providers list in config.yaml and return 'custom:<name>' for a matching base_url.

    When provider_name is given, prefer matching by name first (solving the case where
    multiple custom providers share the same base_url but have different API keys).
    Falls back to base_url matching when no name match is found.

    Returns None if no match is found.
    """
    if not base_url:
        return None
    normalized_url = base_url.strip().rstrip("/")

    # When a provider name is given, try to match by name first.
    # This fixes the P1 bug where two custom providers sharing the same
    # base_url always resolve to the first one's credentials.
    if provider_name:
        normalized_name = _normalize_custom_pool_name(provider_name)
        for norm_name, entry in _iter_custom_providers():
            if norm_name == normalized_name:
                return f"{CUSTOM_POOL_PREFIX}{norm_name}"

    # Fall back to base_url matching (original behavior)
    for norm_name, entry in _iter_custom_providers():
        entry_url = str(entry.get("base_url") or "").strip().rstrip("/")
        if entry_url and entry_url == normalized_url:
            return f"{CUSTOM_POOL_PREFIX}{norm_name}"
    return None


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
    for norm_name, entry in _iter_custom_providers():
        if norm_name == suffix:
            return entry
    return None


def get_pool_strategy(provider: str) -> str:
    """Return the configured selection strategy for a provider."""
    config = _load_config_safe()
    if config is None:
        return STRATEGY_FILL_FIRST

    strategies = config.get("credential_pool_strategies")
    if not isinstance(strategies, dict):
        return STRATEGY_FILL_FIRST

    strategy = str(strategies.get(provider, "") or "").strip().lower()
    if strategy in SUPPORTED_POOL_STRATEGIES:
        return strategy
    return STRATEGY_FILL_FIRST


def credential_pool_matches_provider(
    pool_or_provider: Any,
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
) -> bool:
    """Return whether a pool belongs to the requested runtime provider.

    Named custom endpoints intentionally use two identities: the live agent is
    ``custom`` while its pool is keyed ``custom:<name>``. Accept that pair only
    when the runtime base URL resolves to the exact same custom pool key.
    Empty string identities fail closed. Legacy pool adapters without a
    ``provider`` attribute remain compatible; production pools are scoped.
    """
    raw_pool_provider = getattr(pool_or_provider, "provider", None)
    if raw_pool_provider is None:
        if isinstance(pool_or_provider, str):
            raw_pool_provider = pool_or_provider
        else:
            # Backward compatibility for lightweight/unscoped pool adapters.
            # Production CredentialPool instances always carry ``provider``;
            # old plugins and tests may expose only select()/has_credentials().
            return True
    pool_provider = str(raw_pool_provider or "").strip().lower()
    provider_norm = str(provider or "").strip().lower()
    if not pool_provider or not provider_norm:
        return False
    if pool_provider == provider_norm:
        return True
    if provider_norm != "custom" or not pool_provider.startswith(CUSTOM_POOL_PREFIX):
        return False
    try:
        matched_pool = get_custom_provider_pool_key(base_url or "")
    except Exception:
        return False
    return str(matched_pool or "").strip().lower() == pool_provider


DEFAULT_MAX_CONCURRENT_PER_CREDENTIAL = 1


def _write_through_provider_state_to_global_root(
    provider_id: str, state: Dict[str, Any]
) -> bool:
    """Persist a rotated OAuth ``state`` into the global-root auth.json.

    Observable write-through for the multi-profile rotation hazard
    (#48415 / #43589): nous, openai-codex, and xai-oauth rotate the
    refresh_token on refresh, so when a profile pool refresh rotates a grant
    it resolved from the root fallback, the rotated chain must land back in
    root. Otherwise root keeps a now-revoked refresh token and every other
    profile reading the stale root grant dies with ``refresh_token_reused`` /
    ``invalid_grant`` once its access token expires.

    Only updates ``providers.<provider_id>`` in the root store; never touches
    the profile store (the caller already saved that). Swallows all errors — a
    failed write-through degrades to the pre-existing behavior (root stale), it
    must never break the profile's own successful save. Mirrors
    ``hermes_cli.auth._write_through_xai_oauth_to_global_root`` (which covers
    the non-pool xAI refresh path) for the credential-pool refresh path.
    """
    try:
        global_path = auth_mod._global_auth_file_path()
    except Exception:
        return False
    if global_path is None:
        # Classic mode (profile == root); the profile save already hit root.
        return True
    # Seat belt: under pytest, refuse to write the real user's
    # ~/.hermes/auth.json even when HERMES_HOME points at a profile path
    # (mirrors the read-side guard in _load_global_auth_store). Uses the
    # unmodified HOME env, not Path.home() which fixtures may monkeypatch.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_env = os.environ.get("HOME", "")
        if real_home_env:
            real_root = Path(real_home_env) / ".hermes" / "auth.json"
            try:
                if global_path.resolve(strict=False) == real_root.resolve(strict=False):
                    return False
            except Exception:
                return False
    try:
        auth_mod._persist_provider_state_to_store(
            provider_id,
            state,
            global_path,
            set_active=False,
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive I/O boundary
        logger.debug(
            "%s pool refresh: write-through to global root failed: %s",
            provider_id,
            exc,
        )
        return False


class CredentialPool:
    def __init__(self, provider: str, entries: List[PooledCredential]):
        self.provider = provider
        self._entries = sorted(entries, key=lambda entry: entry.priority)
        self._current_id: Optional[str] = None
        self._strategy = get_pool_strategy(provider)
        # RLock: mutation primitives self-acquire this lock so refresh and
        # status-sync work performed outside the lock still serializes its
        # in-memory updates. Persistence is always drained after releasing it.
        self._lock = threading.RLock()
        # External provider state is serialized separately from in-memory pool
        # mutation. Lock order is always external_state -> pool (brief CAS only):
        # callers must acquire this lock without holding self._lock, may perform
        # provider/file I/O, and only then take self._lock to validate or commit.
        # The reverse order is forbidden so auth-store/file waits cannot form an
        # ABBA pair with pool persistence.
        self._external_state_lock = threading.RLock()
        self._persist_lock = threading.Lock()
        # Persist only entries changed by this pool instance. Borrowed rows are
        # snapshots of another auth store; rewriting every snapshot on an
        # unrelated local change can erase a newer owner cooldown/quarantine.
        self._dirty_entry_ids: Set[str] = set()
        self._dirty_entry_generations: Dict[str, int] = {}
        self._entry_mutation_generation = 0
        self._pending_removed_entries: List[PooledCredential] = []
        self._pending_removed_entry_generations: Dict[int, int] = {}
        self._trusted_codex_source_owners: Dict[int, _TrustedCodexSourceOwner] = {}
        self._source_status_reset_ids: Set[str] = set()
        self._active_leases: Dict[str, int] = {}
        self._max_concurrent = DEFAULT_MAX_CONCURRENT_PER_CREDENTIAL
        # Monotonic timestamp of the last "no available entries" log, used to
        # throttle that message so an empty/exhausted pool cannot storm the
        # shared rotating log (see NO_AVAILABLE_ENTRIES_LOG_THROTTLE_SECONDS).
        # Re-armed to None on every successful selection so a recover→re-exhaust
        # transition logs promptly instead of being swallowed by a stale window.
        self._last_no_entries_log_at: Optional[float] = None
        # #70401: consecutive mark_exhausted_and_rotate() calls whose supplied
        # credential identity matched no pool entry (OAuth wrappers whose
        # runtime key rotates, entries pruned by another process, ...).  These
        # rotations mark nothing exhausted, so without a cap the pool can
        # never converge to "no available entries" and the caller's 401 retry
        # loop runs unbounded and non-interruptible.  Reset whenever a real
        # entry is identified or an escape path returns None.
        self._unmatched_rotation_streak: int = 0

    @staticmethod
    def _same_store_path(left: Path, right: Path) -> bool:
        try:
            return left.expanduser().resolve() == right.expanduser().resolve()
        except (OSError, RuntimeError):
            return left.expanduser() == right.expanduser()

    def _trust_codex_source_owner(
        self,
        entry: PooledCredential,
        *,
        owner_kind: str,
    ) -> bool:
        """Register trusted root-fallback provenance issued during loading."""
        if self.provider != "openai-codex" or owner_kind not in {"singleton", "pool"}:
            return False
        source_path = entry.source_store_path
        if source_path is None:
            return False
        try:
            global_path = auth_mod._global_auth_file_path()
            active_path = auth_mod._auth_file_path()
        except Exception:
            return False
        if global_path is None:
            return False
        if not self._same_store_path(source_path, global_path):
            return False
        if self._same_store_path(source_path, active_path):
            return False
        self._trusted_codex_source_owners[id(entry)] = _TrustedCodexSourceOwner(
            source_path=source_path,
            owner_kind=owner_kind,
            entry_id=entry.id,
            source=entry.source,
        )
        return True

    def _trusted_codex_source_owner(
        self, entry: PooledCredential
    ) -> Optional[_TrustedCodexSourceOwner]:
        owner = self._trusted_codex_source_owners.get(id(entry))
        if owner is None:
            return None
        source_path = entry.source_store_path
        if (
            source_path is None
            or owner.entry_id != entry.id
            or owner.source != entry.source
            or not self._same_store_path(owner.source_path, source_path)
        ):
            return None
        try:
            global_path = auth_mod._global_auth_file_path()
            active_path = auth_mod._auth_file_path()
        except Exception:
            return None
        if global_path is None:
            return None
        if not self._same_store_path(owner.source_path, global_path):
            return None
        if self._same_store_path(owner.source_path, active_path):
            return None
        return owner

    def _is_trusted_codex_source_owned(self, entry: PooledCredential) -> bool:
        return self._trusted_codex_source_owner(entry) is not None

    @staticmethod
    def _entry_tokens_match(
        current: PooledCredential, expected: PooledCredential
    ) -> bool:
        return (
            current.id == expected.id
            and current.source == expected.source
            and current.access_token == expected.access_token
            and current.refresh_token == expected.refresh_token
        )

    def _forget_trusted_codex_source_owner(self, entry: PooledCredential) -> None:
        self._trusted_codex_source_owners.pop(id(entry), None)

    def _validate_source_owned_codex_entries(
        self,
        entry_ids: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Validate borrowed snapshots without holding the pool lock."""
        if self.provider != "openai-codex":
            return set()
        with self._lock:
            snapshots = [
                entry
                for entry in self._entries
                if (entry_ids is None or entry.id in entry_ids)
                and self._is_trusted_codex_source_owned(entry)
            ]
        failed: Set[str] = set()
        for entry in snapshots:
            if self._sync_source_owned_codex_entry(entry) is None:
                failed.add(entry.id)
        return failed

    def has_credentials(self) -> bool:
        with self._lock:
            return bool(self._entries)

    def has_available(self) -> bool:
        """True if at least one entry is not currently in exhaustion cooldown."""
        # ``_available_entries`` is not read-only: it prunes aged-out DEAD
        # manual entries (rebinding ``self._entries``) and persists.  It must
        # run under ``self._lock`` like every other caller (``select`` etc.),
        # otherwise a status probe here can race a concurrent ``select`` /
        # rotation and tear ``self._entries`` or double-write auth.json.
        failed_source_ids = self._validate_source_owned_codex_entries()
        self._sync_external_status_entries()
        with self._lock:
            available, _pending = self._available_entries(
                excluded_source_ids=failed_source_ids,
            )
            result = bool(available)
        self._persist_pending_changes()
        return result

    def next_available_at(self) -> Optional[float]:
        """Earliest epoch time (seconds) any entry re-enters rotation.

        Returns ``None`` when at least one entry is available right now, or
        when no exhausted entry carries a usable recovery time (empty pool,
        or only ``STATUS_DEAD`` entries, which never re-enter via TTL).
        Callers must treat ``None`` as "no wait information", not
        "unavailable".

        Like :meth:`has_available`, expired cooldowns are left uncleared
        (``clear_expired=False``); the only writes are the same
        re-auth/token sync paths ``has_available`` already performs — which
        is exactly why this must run under ``self._lock`` like every other
        ``_available_entries`` caller (see the comment on ``has_available``).
        """
        failed_source_ids = self._validate_source_owned_codex_entries()
        self._sync_external_status_entries()
        with self._lock:
            available, _pending = self._available_entries(
                excluded_source_ids=failed_source_ids,
            )
            if available:
                result = None
            else:
                # Mirror _available_entries: if the pool has no other credential
                # to rotate to, the sole entry's transient throttle cools down in
                # seconds — next_available_at must report that shorter window too,
                # or the fallback restore gate waits an hour for a 60s cooldown.
                sole_credential = sum(
                    1 for e in self._entries if e.last_status != STATUS_DEAD
                ) <= 1
                candidates: List[float] = []
                for entry in self._entries:
                    if entry.last_status != STATUS_EXHAUSTED:
                        continue
                    until = _exhausted_until(
                        entry,
                        sole_credential=sole_credential,
                    )
                    if until is not None:
                        candidates.append(until)
                result = min(candidates) if candidates else None
        self._persist_pending_changes()
        return result

    def entries(self) -> List[PooledCredential]:
        with self._lock:
            return list(self._entries)

    def _current_unlocked(self) -> Optional[PooledCredential]:
        if not self._current_id:
            return None
        return next((entry for entry in self._entries if entry.id == self._current_id), None)

    def current(self) -> Optional[PooledCredential]:
        with self._lock:
            current = self._current_unlocked()
        if current is not None and self._is_trusted_codex_source_owned(current):
            synced = self._sync_source_owned_codex_entry(current)
            if synced is None:
                return None
            with self._lock:
                latest = self._current_unlocked()
                if latest is None or not self._entry_tokens_match(latest, synced):
                    return None
                return latest
        with self._lock:
            return self._current_unlocked()

    def entry_id_for_api_key(self, api_key_hint: Any = None) -> Optional[str]:
        """Return the stable id for the runtime credential in use.

        Prefer the current selection when it still supplies ``api_key_hint``.
        If the cursor was cleared, fall back to an unambiguous key match.
        """
        with self._lock:
            current = self._current_unlocked()
            if current is not None and (
                api_key_hint is None
                or current.runtime_api_key == api_key_hint
            ):
                return current.id
            if api_key_hint is None:
                return None
            matches = [
                entry
                for entry in self._entries
                if entry.runtime_api_key == api_key_hint
            ]
            return matches[0].id if len(matches) == 1 else None

    def _record_entry_mutation_unlocked(
        self,
        entry_id: str,
        *,
        dirty: bool,
    ) -> int:
        """Version one entry mutation while ``self._lock`` is held."""
        self._entry_mutation_generation += 1
        generation = self._entry_mutation_generation
        self._dirty_entry_generations[entry_id] = generation
        self._source_status_reset_ids.discard(entry_id)
        if dirty:
            self._dirty_entry_ids.add(entry_id)
        else:
            self._dirty_entry_ids.discard(entry_id)
        return generation

    def _queue_removed_entry_unlocked(self, entry: PooledCredential) -> None:
        """Record a versioned removal while ``self._lock`` is held."""
        generation = self._record_entry_mutation_unlocked(entry.id, dirty=False)
        self._pending_removed_entries.append(entry)
        self._pending_removed_entry_generations[id(entry)] = generation

    def _dirty_generation_for_snapshot_unlocked(
        self,
        entry: PooledCredential,
    ) -> Optional[int]:
        current = next(
            (candidate for candidate in self._entries if candidate.id == entry.id),
            None,
        )
        if current != entry:
            return None
        return self._dirty_entry_generations.get(entry.id)

    def _clear_dirty_generation_unlocked(
        self,
        entry_id: str,
        generation: Optional[int],
    ) -> bool:
        if (
            generation is None
            or self._dirty_entry_generations.get(entry_id) != generation
        ):
            return False
        self._dirty_entry_ids.discard(entry_id)
        self._source_status_reset_ids.discard(entry_id)
        self._dirty_entry_generations.pop(entry_id, None)
        return True

    def _replace_entry(
        self,
        old: PooledCredential,
        new: PooledCredential,
        *,
        mark_dirty: bool = True,
        preserve_routing: bool = False,
    ) -> Optional[PooledCredential]:
        """Swap an entry only while its id and credential chain still match.

        Self-locking (RLock) so the deferred refresh path — which
        deliberately runs outside the pool lock — cannot tear
        ``self._entries`` against a concurrent select()/rotation.
        """
        with self._lock:
            for idx, entry in enumerate(self._entries):
                if entry.id == old.id:
                    if not self._entry_tokens_match(entry, old):
                        return entry
                    replacement = (
                        replace(
                            new,
                            priority=entry.priority,
                            request_count=entry.request_count,
                        )
                        if preserve_routing
                        else new
                    )
                    owner = self._trusted_codex_source_owner(entry)
                    self._entries[idx] = replacement
                    if owner is not None:
                        self._trusted_codex_source_owners.pop(id(entry), None)
                        if (
                            replacement.id == owner.entry_id
                            and replacement.source == owner.source
                            and replacement.source_store_path is not None
                            and self._same_store_path(
                                replacement.source_store_path,
                                owner.source_path,
                            )
                        ):
                            self._trusted_codex_source_owners[id(replacement)] = owner
                    if mark_dirty and entry != replacement:
                        self._record_entry_mutation_unlocked(
                            replacement.id,
                            dirty=True,
                        )
                    return replacement
            return None

    def _persist(
        self,
        *,
        removed_ids: Optional[List[str]] = None,
        removed_entries: Optional[List[PooledCredential]] = None,
    ) -> None:
        is_owned = getattr(self._lock, "_is_owned", None)
        if callable(is_owned) and is_owned():
            raise RuntimeError(
                "credential pool persistence cannot run while the pool lock is held"
            )
        # Auth transactions and file writes deliberately happen without the
        # pool lock. Source refresh takes auth locks first and then replaces an
        # in-memory row; holding the pool lock here would invert that order.
        with self._persist_lock:
            with self._lock:
                validate_ids = {
                    entry.id
                    for entry in self._entries
                    if self._is_trusted_codex_source_owned(entry)
                    and entry.id not in self._dirty_entry_ids
                }
            if validate_ids:
                self._validate_source_owned_codex_entries(validate_ids)

            with self._lock:
                pending_removed = list(self._pending_removed_entries)
                pending_removed_generations = {
                    id(entry): self._pending_removed_entry_generations.get(id(entry))
                    for entry in pending_removed
                }
                all_removed = pending_removed + list(removed_entries or [])
                entries_snapshot = list(self._entries)
                entries_by_id = {entry.id: entry for entry in entries_snapshot}
                dirty_entries = [
                    (
                        entries_by_id[entry_id],
                        self._dirty_entry_generations.get(entry_id),
                    )
                    for entry_id in self._dirty_entry_ids
                    if entry_id in entries_by_id
                ]
                orphan_dirty_generations = {
                    entry_id: self._dirty_entry_generations.get(entry_id)
                    for entry_id in self._dirty_entry_ids
                    if entry_id not in entries_by_id
                }
                source_dirty = [
                    entry
                    for entry, _generation in dirty_entries
                    if self._is_trusted_codex_source_owned(entry)
                ]
                local_dirty = any(
                    not self._is_trusted_codex_source_owned(entry)
                    for entry, _generation in dirty_entries
                )
                source_removed = [
                    entry
                    for entry in all_removed
                    if self._is_trusted_codex_source_owned(entry)
                ]
                local_removed_ids = list(removed_ids or [])
                local_removed_ids.extend(
                    entry.id
                    for entry in all_removed
                    if not self._is_trusted_codex_source_owned(entry)
                )
                local_entries = [
                    entry
                    for entry in entries_snapshot
                    if not self._is_trusted_codex_source_owned(entry)
                ]
                status_reset_ids = set(self._source_status_reset_ids)

            if local_dirty or local_removed_ids:
                write_credential_pool(
                    self.provider,
                    [entry.to_dict() for entry in local_entries],
                    removed_ids=local_removed_ids,
                )
            for entry in source_dirty:
                self._persist_source_owned_alias(
                    entry,
                    allow_status_reset=entry.id in status_reset_ids,
                )
            for entry in source_removed:
                self._remove_source_owned_alias(entry)

            with self._lock:
                for entry, generation in dirty_entries:
                    self._clear_dirty_generation_unlocked(entry.id, generation)
                current_ids = {entry.id for entry in self._entries}
                for entry_id, generation in orphan_dirty_generations.items():
                    if entry_id not in current_ids:
                        self._clear_dirty_generation_unlocked(entry_id, generation)
                retained_removals: List[PooledCredential] = []
                for entry in self._pending_removed_entries:
                    snapshot_generation = pending_removed_generations.get(id(entry))
                    current_generation = self._pending_removed_entry_generations.get(
                        id(entry)
                    )
                    if (
                        snapshot_generation is not None
                        and current_generation == snapshot_generation
                    ):
                        self._pending_removed_entry_generations.pop(id(entry), None)
                    else:
                        retained_removals.append(entry)
                self._pending_removed_entries = retained_removals
                for entry in all_removed:
                    self._forget_trusted_codex_source_owner(entry)
                pending_entry_ids = {
                    entry.id for entry in self._pending_removed_entries
                }
                for entry_id in list(self._dirty_entry_generations):
                    if (
                        entry_id not in self._dirty_entry_ids
                        and entry_id not in pending_entry_ids
                    ):
                        self._dirty_entry_generations.pop(entry_id, None)

    def _persist_pending_changes(self) -> None:
        with self._lock:
            pending = bool(
                getattr(self, "_dirty_entry_ids", set())
                or getattr(self, "_pending_removed_entries", [])
            )
        if pending:
            self._persist()

    def _is_terminal_auth_failure(
        self,
        status_code: Optional[int],
        normalized_error: Dict[str, Any],
    ) -> bool:
        """Detect upstream-permanent OAuth failures that won't recover on TTL.

        Only fires for 401 responses whose error code/reason matches a known
        terminal OAuth state (token_invalidated, token_revoked, invalid_grant,
        etc.).  Distinguishes permanent failures from transient ones like
        token_expired (refreshable) or generic 401 without a specific reason
        (could be a server-side glitch worth retrying).

        Returns False for non-401 status codes — 429 rate limits and 402
        billing failures are transient by nature and should keep TTL semantics.
        """
        if status_code != 401:
            return False
        reason = normalized_error.get("reason")
        if not isinstance(reason, str):
            return False
        return reason.strip().lower() in _TERMINAL_AUTH_REASONS

    def _mark_exhausted(
        self,
        entry: PooledCredential,
        status_code: Optional[int],
        error_context: Optional[Dict[str, Any]] = None,
        *,
        persist: bool = True,
        failure_reason: Optional[str] = None,
        source_validated: bool = False,
    ) -> PooledCredential:
        if (
            not source_validated
            and self._is_trusted_codex_source_owned(entry)
        ):
            synced = self._sync_source_owned_codex_entry(entry)
            if synced is None:
                return entry
            entry = synced
        normalized_error = _normalize_error_context(error_context)
        # Permanent OAuth failures (token_invalidated, token_revoked, etc.)
        # transition to STATUS_DEAD instead of STATUS_EXHAUSTED.  Without this,
        # a revoked credential gets a 1-hour TTL cooldown and then re-enters
        # rotation, failing immediately every hour until the user manually
        # removes it (issue #32849).  DEAD entries are excluded from rotation
        # unconditionally and only clear via an explicit re-auth write-side
        # sync (``_save_codex_tokens`` after a fresh device-code login).
        if self._is_terminal_auth_failure(status_code, normalized_error):
            terminal_status = STATUS_DEAD
        else:
            terminal_status = STATUS_EXHAUSTED
        # Carry the classifier's verdict onto the entry so the cooldown can be
        # sized by what actually failed, not just the HTTP status (a billing
        # 403 must not get the sole-credential transient cooldown). Absent a
        # classification, clear any stale verdict from a previous failure.
        updated_extra = dict(entry.extra)
        if failure_reason:
            updated_extra["failure_reason"] = failure_reason
        else:
            updated_extra.pop("failure_reason", None)
        updated = replace(
            entry,
            last_status=terminal_status,
            last_status_at=time.time(),
            last_error_code=status_code,
            last_error_reason=normalized_error.get("reason"),
            last_error_message=normalized_error.get("message"),
            last_error_reset_at=normalized_error.get("reset_at"),
            extra=updated_extra,
        )
        replaced = self._replace_entry(entry, updated)
        if replaced is None or replaced != updated:
            return replaced or entry
        if persist:
            self._persist()
        return replaced

    def _sync_anthropic_entry_from_credentials_file(
        self,
        entry: PooledCredential,
        *,
        persist: bool = True,
    ) -> PooledCredential:
        """Sync a claude_code pool entry from ~/.claude/.credentials.json if tokens differ.

        OAuth refresh tokens are single-use. When something external (e.g.
        Claude Code CLI, or another profile's pool) refreshes the token, it
        writes the new pair to ~/.claude/.credentials.json. The pool entry's
        refresh token becomes stale. This method detects that and syncs.
        """
        if self.provider != "anthropic" or entry.source != "claude_code":
            return entry
        try:
            from agent.anthropic_adapter import read_claude_code_credentials
            creds = read_claude_code_credentials()
            if not creds:
                return entry
            file_refresh = creds.get("refreshToken", "")
            file_access = creds.get("accessToken", "")
            file_expires = creds.get("expiresAt", 0)
            # Sync when either token changed.  Access tokens can be re-issued
            # without a new refresh token (silent re-issue path), so checking
            # only refresh_token misses that case and leaves a stale
            # access_token in the pool → 401 on every request until the pool
            # entry's exhausted TTL expires.
            entry_access = entry.access_token or ""
            entry_refresh = entry.refresh_token or ""
            if (file_access or file_refresh) and (
                (file_access and file_access != entry_access)
                or (file_refresh and file_refresh != entry_refresh)
            ):
                logger.debug(
                    "Pool entry %s: syncing tokens from credentials file (tokens changed)",
                    entry.id,
                )
                updated = replace(
                    entry,
                    access_token=file_access or entry.access_token,
                    refresh_token=file_refresh or entry.refresh_token,
                    expires_at_ms=file_expires or entry.expires_at_ms,
                    last_status=None,
                    last_status_at=None,
                    last_error_code=None,
                    last_error_reason=None,
                    last_error_message=None,
                    last_error_reset_at=None,
                    extra={
                        **entry.extra,
                        "credential_source": creds.get("source"),
                    },
                )
                current = self._replace_entry(
                    entry,
                    updated,
                    preserve_routing=True,
                )
                if persist:
                    self._persist_pending_changes()
                return current or entry
        except Exception as exc:
            logger.debug("Failed to sync from credentials file: %s", exc)
        return entry

    def _codex_entry_from_provider_state(
        self,
        entry: PooledCredential,
        state: Optional[Dict[str, Any]],
    ) -> PooledCredential:
        """Return ``entry`` updated from its singleton state, without I/O."""
        if (
            self.provider != "openai-codex"
            or entry.source != "device_code"
            or not isinstance(state, dict)
        ):
            return entry
        tokens = state.get("tokens")
        if not isinstance(tokens, dict):
            return entry
        store_access = tokens.get("access_token", "")
        store_refresh = tokens.get("refresh_token", "")
        entry_access = entry.access_token or ""
        entry_refresh = entry.refresh_token or ""
        should_adopt = bool(
            store_access
            and (
                store_access != entry_access
                or (store_refresh and store_refresh != entry_refresh)
            )
        )
        if store_refresh and store_refresh != entry_refresh and not store_access:
            logger.info(
                "Pool entry %s: auth.json has newer refresh_token but no "
                "access_token; adopting refresh_token to avoid replaying "
                "consumed token",
                entry.id,
            )
            should_adopt = True
        if not should_adopt:
            return entry

        logger.debug(
            "Pool entry %s: syncing Codex tokens from auth.json "
            "(refreshed by another process)",
            entry.id,
        )
        field_updates: Dict[str, Any] = {
            "access_token": store_access or entry.access_token,
            "refresh_token": store_refresh or entry.refresh_token,
            "last_status": None,
            "last_status_at": None,
            "last_error_code": None,
            "last_error_reason": None,
            "last_error_message": None,
            "last_error_reset_at": None,
        }
        if state.get("last_refresh"):
            field_updates["last_refresh"] = state["last_refresh"]
        return replace(entry, **field_updates)

    @staticmethod
    def _codex_provider_state_has_refresh_chain(
        state: Optional[Dict[str, Any]],
    ) -> bool:
        if not isinstance(state, dict):
            return False
        tokens = state.get("tokens")
        if not isinstance(tokens, dict):
            return False
        return bool(
            str(tokens.get("access_token") or "").strip()
            and str(tokens.get("refresh_token") or "").strip()
        )

    @staticmethod
    def _codex_exact_pool_alias(
        source_store: Dict[str, Any],
        entry: PooledCredential,
    ) -> Tuple[Optional[List[Any]], Optional[int], Optional[Dict[str, Any]]]:
        pool = source_store.get("credential_pool")
        persisted = pool.get("openai-codex") if isinstance(pool, dict) else None
        if not isinstance(persisted, list):
            return None, None, None
        for index, item in enumerate(persisted):
            if (
                isinstance(item, dict)
                and item.get("id") == entry.id
                and item.get("source") == entry.source
            ):
                return persisted, index, item
        return persisted, None, None

    @staticmethod
    def _codex_pool_alias_has_complete_credentials(
        payload: Optional[Dict[str, Any]],
    ) -> bool:
        if not isinstance(payload, dict):
            return False
        if not str(payload.get("access_token") or "").strip():
            return False
        if (
            payload.get("auth_type") == AUTH_TYPE_OAUTH
            or payload.get("source") in {"device_code", SOURCE_MANUAL_DEVICE_CODE}
        ):
            return bool(str(payload.get("refresh_token") or "").strip())
        return True

    @staticmethod
    def _codex_tokens_changed(
        left: PooledCredential,
        right: PooledCredential,
    ) -> bool:
        return (
            left.access_token != right.access_token
            or left.refresh_token != right.refresh_token
        )

    def _codex_alias_is_singleton(
        self,
        entry: PooledCredential,
        state: Optional[Dict[str, Any]],
        source_path: Path,
        *,
        owner_kind: Optional[str] = None,
    ) -> bool:
        owner = self._trusted_codex_source_owner(entry)
        trusted_kind = owner.owner_kind if owner is not None else owner_kind
        if trusted_kind is not None:
            return trusted_kind == "singleton"
        if (
            entry.source != "device_code"
            or not self._codex_provider_state_has_refresh_chain(state)
        ):
            return False
        assert isinstance(state, dict)
        alias = _codex_source_pool_alias(source_path, state)
        return bool(alias is not None and alias.get("id") == entry.id)

    def _codex_write_path_is_authorized(
        self,
        entry: PooledCredential,
        source_path: Path,
    ) -> bool:
        owner = self._trusted_codex_source_owner(entry)
        if owner is not None:
            return self._same_store_path(owner.source_path, source_path)
        try:
            active_path = auth_mod._auth_file_path()
        except Exception:
            return False
        return (
            entry.source_store_path is None
            and self._same_store_path(active_path, source_path)
        )

    def _drop_source_owned_entry(self, entry: PooledCredential) -> bool:
        with self._lock:
            current = next(
                (item for item in self._entries if item.id == entry.id),
                None,
            )
            if current is None:
                self._forget_trusted_codex_source_owner(entry)
                return True
            if not self._entry_tokens_match(current, entry):
                return False
            self._entries = [item for item in self._entries if item.id != entry.id]
            self._record_entry_mutation_unlocked(entry.id, dirty=False)
            if self._current_id == entry.id:
                self._current_id = None
            self._forget_trusted_codex_source_owner(current)
            return True

    def _revoke_missing_codex_source(
        self,
        entry: PooledCredential,
        source_path: Path,
        *,
        source_store: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Drop stale memory and remove an incomplete exact owner alias."""
        store = source_store if source_store is not None else _load_auth_store(source_path)
        persisted, index, item = self._codex_exact_pool_alias(store, entry)
        if (
            persisted is not None
            and index is not None
            and not self._codex_pool_alias_has_complete_credentials(item)
        ):
            persisted.pop(index)
            _save_auth_store(store, target_path=source_path)
        self._drop_source_owned_entry(entry)

    def _revoke_missing_codex_device_code_source(
        self,
        entry: PooledCredential,
        source_path: Path,
    ) -> None:
        """Compatibility wrapper for the pre-provenance singleton path."""
        self._revoke_missing_codex_source(entry, source_path)

    def _sync_source_owned_codex_entry(
        self,
        entry: PooledCredential,
    ) -> Optional[PooledCredential]:
        """Re-read one borrowed row by stable id, or revoke it if owner vanished."""
        owner = self._trusted_codex_source_owner(entry)
        if self.provider != "openai-codex" or owner is None:
            return entry
        try:
            with auth_mod._provider_state_transaction(
                "openai-codex",
                expected_source_path=owner.source_path,
            ) as (_active_store, state, source_path):
                if source_path is None:
                    self._drop_source_owned_entry(entry)
                    return None
                source_store = _load_auth_store(source_path)
                _persisted, _index, item = self._codex_exact_pool_alias(
                    source_store,
                    entry,
                )
                if not self._codex_pool_alias_has_complete_credentials(item):
                    self._revoke_missing_codex_source(
                        entry,
                        source_path,
                        source_store=source_store,
                    )
                    return None
                assert isinstance(item, dict)
                stored = replace(
                    PooledCredential.from_dict("openai-codex", item),
                    source_store_path=source_path,
                )
                if self._codex_alias_is_singleton(
                    stored,
                    state,
                    source_path,
                    owner_kind=owner.owner_kind,
                ):
                    if not self._codex_provider_state_has_refresh_chain(state):
                        self._revoke_missing_codex_source(
                            entry,
                            source_path,
                            source_store=source_store,
                        )
                        return None
                    synced = self._codex_entry_from_provider_state(stored, state)
                    if self._codex_tokens_changed(synced, stored):
                        self._persist_codex_device_code_refresh(synced, source_path)
                    stored = synced
                if stored != entry:
                    replaced = self._replace_entry(entry, stored, mark_dirty=False)
                    if replaced is None:
                        return None
                    if not self._entry_tokens_match(replaced, stored):
                        return None
                    return replaced
                return entry
        except Exception as exc:
            logger.debug("Failed to validate borrowed Codex entry: %s", exc)
            return None

    def _persist_source_owned_alias(
        self,
        entry: PooledCredential,
        *,
        allow_status_reset: bool = False,
    ) -> None:
        """Route one changed borrowed Codex row to its exact owning alias."""
        owner = self._trusted_codex_source_owner(entry)
        if self.provider != "openai-codex" or owner is None:
            return

        with auth_mod._provider_state_transaction(
            "openai-codex",
            expected_source_path=owner.source_path,
        ) as (_active_store, state, source_path):
            if source_path is None:
                self._drop_source_owned_entry(entry)
                return
            source_store = _load_auth_store(source_path)
            persisted, index, item = self._codex_exact_pool_alias(source_store, entry)
            if not self._codex_pool_alias_has_complete_credentials(item):
                self._revoke_missing_codex_source(
                    entry,
                    source_path,
                    source_store=source_store,
                )
                return
            assert isinstance(item, dict)
            stored = replace(
                PooledCredential.from_dict("openai-codex", item),
                source_store_path=source_path,
            )
            if self._codex_alias_is_singleton(
                stored,
                state,
                source_path,
                owner_kind=owner.owner_kind,
            ):
                if not self._codex_provider_state_has_refresh_chain(state):
                    self._revoke_missing_codex_source(
                        entry,
                        source_path,
                        source_store=source_store,
                    )
                    return
                synced = self._codex_entry_from_provider_state(stored, state)
                if self._codex_tokens_changed(synced, stored):
                    self._persist_codex_device_code_refresh(
                        synced,
                        source_path,
                        provenance_entry=entry,
                    )
                stored = synced
            if self._codex_tokens_changed(stored, entry):
                self._replace_entry(entry, stored, mark_dirty=False)
                return

            assert persisted is not None and index is not None and item is not None
            serialized = entry.to_dict()
            updated = dict(item)
            updated.update(serialized)
            if not allow_status_reset:
                updated = auth_mod._merge_disk_cooldown_state(
                    updated,
                    item,
                    "openai-codex",
                )
            if updated != item:
                persisted[index] = updated
                _save_auth_store(source_store, target_path=source_path)
            final_entry = replace(
                PooledCredential.from_dict("openai-codex", updated),
                source_store_path=source_path,
            )
            if final_entry != entry:
                self._replace_entry(entry, final_entry, mark_dirty=False)

    def _remove_source_owned_alias(self, entry: PooledCredential) -> None:
        owner = self._trusted_codex_source_owner(entry)
        if self.provider != "openai-codex" or owner is None:
            return
        with auth_mod._provider_state_transaction(
            "openai-codex",
            expected_source_path=owner.source_path,
        ) as (_active_store, state, source_path):
            if source_path is None:
                return
            source_store = _load_auth_store(source_path)
            persisted, index, item = self._codex_exact_pool_alias(source_store, entry)
            if persisted is None or index is None or not isinstance(item, dict):
                return
            stored = replace(
                PooledCredential.from_dict("openai-codex", item),
                source_store_path=source_path,
            )
            if self._codex_alias_is_singleton(
                stored,
                state,
                source_path,
                owner_kind=owner.owner_kind,
            ):
                refresh_token = str(item.get("refresh_token") or "").strip()
                persisted[:] = [
                    candidate
                    for candidate in persisted
                    if not (
                        isinstance(candidate, dict)
                        and candidate.get("source") == "device_code"
                        and (
                            candidate.get("id") == entry.id
                            or (
                                refresh_token
                                and str(
                                    candidate.get("refresh_token") or ""
                                ).strip()
                                == refresh_token
                            )
                        )
                    )
                ]
                providers = source_store.get("providers")
                persisted_state = (
                    providers.get("openai-codex")
                    if isinstance(providers, dict)
                    else None
                )
                if isinstance(persisted_state, dict):
                    tokens = persisted_state.get("tokens")
                    if isinstance(tokens, dict):
                        tokens.pop("access_token", None)
                        tokens.pop("refresh_token", None)
                        persisted_state["tokens"] = tokens
            else:
                persisted.pop(index)
            _save_auth_store(source_store, target_path=source_path)

    def _persist_codex_device_code_refresh(
        self,
        entry: PooledCredential,
        source_path: Path,
        *,
        provenance_entry: Optional[PooledCredential] = None,
    ) -> None:
        """Persist one singleton-owned Codex chain to its exact source store.

        The caller holds the active-profile -> source-store transaction. Only
        the matching ``device_code`` pool alias and singleton are updated;
        independent manual accounts and unrelated store data remain untouched.
        """
        if not self._codex_write_path_is_authorized(
            provenance_entry or entry,
            source_path,
        ):
            return
        with self._lock:
            dirty_generation = self._dirty_generation_for_snapshot_unlocked(entry)
        auth_store = _load_auth_store(source_path)
        changed = False

        providers = auth_store.get("providers")
        state = providers.get("openai-codex") if isinstance(providers, dict) else None
        if isinstance(state, dict):
            tokens = state.get("tokens")
            if isinstance(tokens, dict):
                next_tokens = dict(tokens)
                next_tokens["access_token"] = entry.access_token
                if entry.refresh_token:
                    next_tokens["refresh_token"] = entry.refresh_token
                if next_tokens != tokens:
                    state["tokens"] = next_tokens
                    changed = True
                if entry.last_refresh and state.get("last_refresh") != entry.last_refresh:
                    state["last_refresh"] = entry.last_refresh
                    changed = True

        pool = auth_store.get("credential_pool")
        entries = pool.get("openai-codex") if isinstance(pool, dict) else None
        if isinstance(entries, list):
            serialized = entry.to_dict()
            for persisted in entries:
                if not isinstance(persisted, dict):
                    continue
                if (
                    persisted.get("id") != entry.id
                    or persisted.get("source") != "device_code"
                ):
                    continue
                updated_persisted = dict(persisted)
                updated_persisted.update(serialized)
                if updated_persisted != persisted:
                    persisted.clear()
                    persisted.update(updated_persisted)
                    changed = True
                break

        if changed:
            _save_auth_store(auth_store, target_path=source_path)
        with self._lock:
            self._clear_dirty_generation_unlocked(entry.id, dirty_generation)

    def _persist_codex_exact_alias_refresh(
        self,
        entry: PooledCredential,
        source_path: Path,
    ) -> bool:
        """Persist a non-singleton Codex refresh to one exact owning row."""
        if not self._codex_write_path_is_authorized(entry, source_path):
            return False
        with self._lock:
            dirty_generation = self._dirty_generation_for_snapshot_unlocked(entry)
        auth_store = _load_auth_store(source_path)
        persisted, index, item = self._codex_exact_pool_alias(auth_store, entry)
        if not self._codex_pool_alias_has_complete_credentials(item):
            self._revoke_missing_codex_source(
                entry,
                source_path,
                source_store=auth_store,
            )
            return False
        assert persisted is not None and index is not None and item is not None
        updated = dict(item)
        updated.update(entry.to_dict())
        if updated != item:
            persisted[index] = updated
            _save_auth_store(auth_store, target_path=source_path)
        with self._lock:
            self._clear_dirty_generation_unlocked(entry.id, dirty_generation)
        return True

    def _quarantine_codex_exact_alias_refresh(
        self,
        entry: PooledCredential,
        source_path: Path,
    ) -> None:
        """Remove one terminal non-singleton chain without touching singleton."""
        if not self._codex_write_path_is_authorized(entry, source_path):
            return
        auth_store = _load_auth_store(source_path)
        persisted, index, _item = self._codex_exact_pool_alias(auth_store, entry)
        if persisted is not None and index is not None:
            persisted.pop(index)
            _save_auth_store(auth_store, target_path=source_path)

    def _quarantine_codex_device_code_refresh(
        self,
        entry: PooledCredential,
        source_path: Path,
        exc: Exception,
    ) -> Set[str]:
        """Clear every device alias on one terminal singleton chain."""
        if not self._codex_write_path_is_authorized(entry, source_path):
            return set()
        auth_store = _load_auth_store(source_path)
        removed_ids: Set[str] = {entry.id}
        providers = auth_store.get("providers")
        state = providers.get("openai-codex") if isinstance(providers, dict) else None
        if isinstance(state, dict):
            tokens = state.get("tokens")
            if isinstance(tokens, dict):
                store_refresh = str(tokens.get("refresh_token") or "").strip()
                entry_refresh = str(entry.refresh_token or "").strip()
                if not store_refresh or store_refresh == entry_refresh:
                    tokens.pop("access_token", None)
                    tokens.pop("refresh_token", None)
                    state["tokens"] = tokens
                    state["last_auth_error"] = {
                        "provider": "openai-codex",
                        "code": getattr(exc, "code", "unknown"),
                        "message": str(exc),
                        "reason": "credential_pool_refresh_failure",
                        "relogin_required": True,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }

        pool = auth_store.get("credential_pool")
        if isinstance(pool, dict) and isinstance(
            pool.get("openai-codex"), list
        ):
            entries = pool["openai-codex"]
            entry_refresh = str(entry.refresh_token or "").strip()
            removed_ids.update(
                str(persisted.get("id"))
                for persisted in entries
                if (
                    isinstance(persisted, dict)
                    and persisted.get("source") == "device_code"
                    and (
                        persisted.get("id") == entry.id
                        or (
                            entry_refresh
                            and str(persisted.get("refresh_token") or "").strip()
                            == entry_refresh
                        )
                    )
                    and persisted.get("id")
                )
            )
            pool["openai-codex"] = [
                persisted
                for persisted in entries
                if not (
                    isinstance(persisted, dict)
                    and persisted.get("source") == "device_code"
                    and (
                        persisted.get("id") == entry.id
                        or (
                            entry_refresh
                            and str(persisted.get("refresh_token") or "").strip()
                            == entry_refresh
                        )
                    )
                )
            ]
        _save_auth_store(auth_store, target_path=source_path)
        return removed_ids

    def _sync_codex_entry_from_auth_store(
        self, entry: PooledCredential
    ) -> PooledCredential:
        """Sync a singleton-seeded Codex entry from its owning auth store."""
        if self.provider != "openai-codex" or entry.source != "device_code":
            return entry
        owner = self._trusted_codex_source_owner(entry)
        try:
            with auth_mod._provider_state_transaction(
                "openai-codex",
                expected_source_path=(
                    owner.source_path
                    if owner is not None
                    else auth_mod._auth_file_path()
                ),
            ) as (
                _auth_store,
                state,
                source_path,
            ):
                source_path = source_path or auth_mod._auth_file_path()
                updated = self._codex_entry_from_provider_state(entry, state)
                if updated is entry:
                    return entry
                replaced = self._replace_entry(
                    entry,
                    updated,
                    preserve_routing=True,
                )
                if replaced is None or not self._entry_tokens_match(replaced, updated):
                    return replaced or entry
                self._persist_codex_device_code_refresh(replaced, source_path)
                return replaced
        except Exception as exc:
            logger.debug("Failed to sync Codex entry from auth.json: %s", exc)
            return entry

    def _sync_xai_oauth_entry_from_auth_store(
        self,
        entry: PooledCredential,
        *,
        persist: bool = True,
    ) -> PooledCredential:
        """Sync an xAI OAuth pool entry from auth.json if tokens differ.

        xAI OAuth refresh tokens are single-use.  When another Hermes process
        (or another profile sharing the same auth.json) refreshes the token,
        it writes the new pair to ``providers["xai-oauth"]["tokens"]`` under
        ``_auth_store_lock``.  Without this resync, our in-memory pool entry
        keeps the consumed refresh_token and the next ``_refresh_entry`` call
        would replay it and get a ``refresh_token_reused``-style 4xx.

        Only applies to entries seeded from the singleton (``device_code``);
        manually added entries are independent credentials with their own
        refresh-token lifecycle.
        """
        if self.provider != "xai-oauth" or entry.source != "device_code":
            return entry
        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                state = _load_provider_state(auth_store, "xai-oauth")
            if not isinstance(state, dict):
                return entry
            tokens = state.get("tokens")
            if not isinstance(tokens, dict):
                return entry
            store_access = tokens.get("access_token", "")
            store_refresh = tokens.get("refresh_token", "")
            entry_access = entry.access_token or ""
            entry_refresh = entry.refresh_token or ""
            if store_access and (
                store_access != entry_access
                or (store_refresh and store_refresh != entry_refresh)
            ):
                logger.debug(
                    "Pool entry %s: syncing xAI OAuth tokens from auth.json "
                    "(refreshed by another process)",
                    entry.id,
                )
                field_updates: Dict[str, Any] = {
                    "access_token": store_access,
                    "refresh_token": store_refresh or entry.refresh_token,
                    "last_status": None,
                    "last_status_at": None,
                    "last_error_code": None,
                    "last_error_reason": None,
                    "last_error_message": None,
                    "last_error_reset_at": None,
                }
                if state.get("last_refresh"):
                    field_updates["last_refresh"] = state["last_refresh"]
                updated = replace(entry, **field_updates)
                current = self._replace_entry(
                    entry,
                    updated,
                    preserve_routing=True,
                )
                if persist:
                    self._persist_pending_changes()
                return current or entry
        except Exception as exc:
            logger.debug("Failed to sync xAI OAuth entry from auth.json: %s", exc)
        return entry

    def _sync_xai_oauth_entry_from_pool_store(
        self, entry: PooledCredential
    ) -> PooledCredential:
        """Adopt a token pair rotated by another pool instance.

        Direct xAI integrations load a fresh ``CredentialPool`` for each
        request. Their in-memory locks therefore cannot protect xAI's
        single-use refresh token across concurrent requests or processes.
        This helper is called while the shared auth-store lock is held and
        re-reads the exact persisted row before a refresh POST is attempted.
        """
        if self.provider != "xai-oauth":
            return entry
        try:
            persisted = next(
                (
                    payload
                    for payload in read_credential_pool(self.provider)
                    if isinstance(payload, dict) and payload.get("id") == entry.id
                ),
                None,
            )
            if not isinstance(persisted, dict):
                return entry
            stored = PooledCredential.from_dict(self.provider, persisted)
            if (
                stored.access_token != entry.access_token
                or stored.refresh_token != entry.refresh_token
            ):
                logger.debug(
                    "Pool entry %s: adopting xAI OAuth tokens rotated by another pool instance",
                    entry.id,
                )
                current = self._replace_entry(
                    entry,
                    stored,
                    preserve_routing=True,
                )
                return current or entry
        except Exception as exc:
            logger.debug("Failed to sync xAI OAuth entry from credential pool: %s", exc)
        return entry

    def _sync_nous_entry_from_auth_store(
        self,
        entry: PooledCredential,
        *,
        persist: bool = True,
    ) -> PooledCredential:
        """Sync a Nous pool entry from auth.json if tokens differ.

        Nous OAuth refresh tokens are single-use.  When another process
        (e.g. a concurrent cron) refreshes the token via
        ``resolve_nous_runtime_credentials``, it writes fresh tokens to
        auth.json under ``_auth_store_lock``.  The pool entry's tokens
        become stale.  This method detects that and adopts the newer pair,
        avoiding a "refresh token reuse" revocation on the Nous Portal.
        """
        if self.provider != "nous" or entry.source != "device_code":
            return entry
        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                state = _load_provider_state(auth_store, "nous")
            if not state:
                return entry
            store_refresh = state.get("refresh_token", "")
            store_access = state.get("access_token", "")
            comparable_updates = {
                "access_token": store_access,
                "refresh_token": store_refresh,
                "expires_at": state.get("expires_at"),
                "agent_key": state.get("agent_key"),
                "agent_key_expires_at": state.get("agent_key_expires_at"),
                "inference_base_url": state.get("inference_base_url"),
            }
            should_sync = any(
                value not in (None, "") and getattr(entry, key, None) != value
                for key, value in comparable_updates.items()
            )
            if should_sync:
                logger.debug(
                    "Pool entry %s: syncing Nous state from auth.json",
                    entry.id,
                )
                field_updates: Dict[str, Any] = {
                    "last_status": None,
                    "last_status_at": None,
                    "last_error_code": None,
                    "last_error_reason": None,
                    "last_error_message": None,
                    "last_error_reset_at": None,
                }
                if store_access:
                    field_updates["access_token"] = store_access
                if store_refresh:
                    field_updates["refresh_token"] = store_refresh
                if state.get("expires_at"):
                    field_updates["expires_at"] = state["expires_at"]
                if state.get("agent_key"):
                    field_updates["agent_key"] = state["agent_key"]
                if state.get("agent_key_expires_at"):
                    field_updates["agent_key_expires_at"] = state["agent_key_expires_at"]
                if state.get("inference_base_url"):
                    field_updates["inference_base_url"] = state["inference_base_url"]
                extra_updates = dict(entry.extra)
                for extra_key in ("obtained_at", "expires_in", "agent_key_id",
                                  "agent_key_expires_in", "agent_key_reused",
                                  "agent_key_obtained_at"):
                    val = state.get(extra_key)
                    if val is not None:
                        extra_updates[extra_key] = val
                updated = replace(entry, extra=extra_updates, **field_updates)
                current = self._replace_entry(
                    entry,
                    updated,
                    preserve_routing=True,
                )
                if persist:
                    self._persist_pending_changes()
                return current or entry
        except Exception as exc:
            logger.debug("Failed to sync Nous entry from auth.json: %s", exc)
        return entry

    def _sync_device_code_entry_to_auth_store(self, entry: PooledCredential) -> bool:
        """Write refreshed pool entry tokens back to auth.json providers.

        After a pool-level refresh, the pool entry has fresh tokens but
        auth.json's ``providers.<id>`` still holds the pre-refresh state.
        On the next ``load_pool()``, ``_seed_from_singletons()`` reads that
        stale state and can overwrite the fresh pool entry — potentially
        re-seeding a consumed single-use refresh token.

        Applies to any OAuth provider whose singleton lives in auth.json
        (currently Nous, OpenAI Codex, and xAI Grok OAuth).

        ``set_active=False`` on every write: a pool sync-back is a
        token-rotation side effect, not the user choosing a provider.
        Using ``_save_provider_state`` (which sets ``active_provider``)
        here would mean every Nous/Codex/xAI refresh in a multi-provider
        setup silently flips the ``active_provider`` flag — the next
        ``hermes`` invocation that defaults to the active provider
        (e.g. setup wizard, ``hermes auth status``) would land on
        whatever provider happened to refresh last, not whatever the
        user actually chose.
        """
        # Only sync entries that were seeded *from* a singleton.  Manually
        # added pool entries (source="manual:*") are independent credentials
        # and must not write back to the singleton.  All singleton-seeded
        # device-code sources (nous, openai-codex, xAI) use ``device_code``.
        if entry.source != "device_code":
            return True
        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                _wt_provider_id = {
                    "nous": "nous",
                    "openai-codex": "openai-codex",
                    "xai-oauth": "xai-oauth",
                }.get(self.provider)
                # Resolve state and track which store it came from — the
                # source path tells us whether this profile genuinely owns
                # its provider block or is reading from the global root.
                # #74339: the old key-presence check decided write-through
                # on whether the profile had ``providers.<id>`` BEFORE the
                # save — correct for the first refresh but self-sealing
                # because ``_store_provider_state`` unconditionally creates
                # that key inside the same function.  Once the profile has
                # the key, every subsequent refresh silently disables the
                # root write-through and root keeps a revoked refresh token.
                #
                # Fix: use ``_load_provider_state_with_source`` to learn
                # where the state was resolved from.  When the grant was
                # resolved from the global root, write back *only* to root
                # and skip ``_store_provider_state`` for the profile so the
                # profile does not accrue a shadowing ``providers.<id>``
                # key that blocks both the root fallback and the write-through
                # on subsequent calls.
                if self.provider == "nous":
                    state, source_path = _load_provider_state_with_source(
                        auth_store, "nous"
                    )
                    if state is None:
                        return False
                elif self.provider == "openai-codex":
                    state, source_path = _load_provider_state_with_source(
                        auth_store, "openai-codex"
                    )
                    if not isinstance(state, dict):
                        return False
                elif self.provider == "xai-oauth":
                    state, source_path = _load_provider_state_with_source(
                        auth_store, "xai-oauth"
                    )
                    if not isinstance(state, dict):
                        return False
                else:
                    return False

                global_root = auth_mod._global_auth_file_path()
                is_from_root = bool(
                    source_path is not None
                    and global_root is not None
                    and auth_mod._same_path(source_path, global_root)
                )

                if self.provider == "nous":
                    state["access_token"] = entry.access_token
                    if entry.refresh_token:
                        state["refresh_token"] = entry.refresh_token
                    if entry.expires_at:
                        state["expires_at"] = entry.expires_at
                    if entry.agent_key:
                        state["agent_key"] = entry.agent_key
                    if entry.agent_key_expires_at:
                        state["agent_key_expires_at"] = entry.agent_key_expires_at
                    for extra_key in ("obtained_at", "expires_in", "agent_key_id",
                                      "agent_key_expires_in", "agent_key_reused",
                                      "agent_key_obtained_at"):
                        val = entry.extra.get(extra_key)
                        if val is not None:
                            state[extra_key] = val
                    if entry.inference_base_url:
                        state["inference_base_url"] = entry.inference_base_url

                elif self.provider == "openai-codex":
                    tokens = state.get("tokens")
                    if not isinstance(tokens, dict):
                        return False
                    tokens["access_token"] = entry.access_token
                    if entry.refresh_token:
                        tokens["refresh_token"] = entry.refresh_token
                    if entry.last_refresh:
                        state["last_refresh"] = entry.last_refresh

                elif self.provider == "xai-oauth":
                    tokens = state.get("tokens")
                    if not isinstance(tokens, dict):
                        return False
                    tokens["access_token"] = entry.access_token
                    if entry.refresh_token:
                        tokens["refresh_token"] = entry.refresh_token
                    if entry.last_refresh:
                        state["last_refresh"] = entry.last_refresh

                if is_from_root and _wt_provider_id:
                    # Grant was resolved from root — write back to root
                    # only.  Do NOT call _store_provider_state on the
                    # profile auth_store (it would create a shadowing
                    # providers.<id> key that disables write-through on
                    # the next refresh — #74339).
                    # _load_provider_state has root fallback, so the
                    # profile can always read fresh tokens from root
                    # without needing its own providers block.
                    return _write_through_provider_state_to_global_root(
                        _wt_provider_id, state
                    )
                else:
                    # Profile genuinely owns this provider — write to
                    # the profile store as normal.
                    _store_provider_state(
                        auth_store, self.provider, state, set_active=False
                    )
                    _save_auth_store(auth_store)
                    return True
        except Exception as exc:
            logger.debug("Failed to sync %s pool entry back to auth store: %s", self.provider, exc)
            return False

    def _refresh_entry(
        self,
        entry: PooledCredential,
        *,
        force: bool,
    ) -> Optional[PooledCredential]:
        """Serialize provider refresh and reject superseded token lineages."""
        self._assert_external_state_lock_order()

        def refresh_current() -> Optional[PooledCredential]:
            with self._lock:
                current = next(
                    (item for item in self._entries if item.id == entry.id),
                    None,
                )
            if current is None:
                return None
            if (
                current.source != entry.source
                or not self._entry_tokens_match(current, entry)
            ):
                return current
            return self._refresh_entry_serialized(current, force=force)

        source_owned = (
            self.provider == "anthropic" and entry.source == "claude_code"
        ) or (
            self.provider in {"openai-codex", "xai-oauth"}
        ) or (
            self.provider == "nous"
            and entry.source == "device_code"
        )
        if source_owned:
            refreshed = refresh_current()
        else:
            with self._external_state_lock:
                refreshed = refresh_current()

        # Source transactions release their auth/provider locks before pool
        # persistence.  Normal persistence acquires _persist_lock first and may
        # then acquire the auth-store lock, so draining here keeps one order.
        self._persist_pending_changes()
        if refreshed is None:
            return None
        current = self._revalidate_refreshed_entry(refreshed)
        self._persist_pending_changes()
        return current

    def _refresh_entry_serialized(
        self,
        entry: PooledCredential,
        *,
        force: bool,
    ) -> Optional[PooledCredential]:
        if entry.auth_type != AUTH_TYPE_OAUTH or not entry.refresh_token:
            if force:
                self._mark_exhausted(entry, None)
            return None

        if self.provider == "anthropic" and entry.source == "claude_code":
            return self._refresh_anthropic_source_entry(entry, force=force)

        # Codex and xAI OAuth refresh tokens are single-use.  The
        # sync→POST→write-back sequence below must run atomically across Hermes
        # processes: otherwise two processes can both adopt the same on-disk
        # token, both POST it, and the loser gets ``refresh_token_reused``.
        # Serialize the whole sequence through the shared cross-process
        # auth-store flock (the same lock and extended-timeout pattern used by
        # resolve_codex_runtime_credentials()).  When a waiter finally acquires
        # the lock, the in-lock re-sync below picks up the rotated token the
        # winner persisted and skips the POST.
        codex_owner = self._trusted_codex_source_owner(entry)
        if self.provider == "openai-codex" and codex_owner is not None:
            with auth_mod._provider_state_transaction(
                "openai-codex",
                timeout_seconds=self._single_use_refresh_lock_timeout(),
                expected_source_path=codex_owner.source_path,
            ) as (_auth_store, state, source_path):
                if source_path is None:
                    self._drop_source_owned_entry(entry)
                    return None
                source_store = _load_auth_store(source_path)
                _persisted, _index, item = self._codex_exact_pool_alias(
                    source_store,
                    entry,
                )
                if not self._codex_pool_alias_has_complete_credentials(item):
                    self._revoke_missing_codex_source(
                        entry,
                        source_path,
                        source_store=source_store,
                    )
                    return None
                assert isinstance(item, dict)
                stored = replace(
                    PooledCredential.from_dict("openai-codex", item),
                    source_store_path=source_path,
                )
                singleton_owned = self._codex_alias_is_singleton(
                    stored,
                    state,
                    source_path,
                    owner_kind=codex_owner.owner_kind,
                )
                if singleton_owned:
                    if not self._codex_provider_state_has_refresh_chain(state):
                        self._revoke_missing_codex_source(
                            entry,
                            source_path,
                            source_store=source_store,
                        )
                        return None
                    stored = self._codex_entry_from_provider_state(stored, state)
                if self._codex_tokens_changed(stored, entry):
                    # A waiter that observed the winner's rotated exact row
                    # adopts it even for force=True; replay would consume a
                    # single-use refresh token twice.
                    replaced = self._replace_entry(
                        entry,
                        stored,
                        mark_dirty=False,
                        preserve_routing=True,
                    )
                    if replaced is None or not self._entry_tokens_match(replaced, stored):
                        return None
                    if singleton_owned:
                        self._persist_codex_device_code_refresh(replaced, source_path)
                    return replaced
                replaced = self._replace_entry(
                    entry,
                    stored,
                    mark_dirty=False,
                    preserve_routing=True,
                )
                if replaced is None or not self._entry_tokens_match(replaced, stored):
                    return None
                return self._refresh_entry_impl(
                    replaced,
                    force=force,
                    codex_source_path=source_path,
                    codex_singleton_owned=singleton_owned,
                )
        if self.provider == "openai-codex" and entry.source == "device_code":
            with auth_mod._provider_state_transaction(
                "openai-codex",
                timeout_seconds=self._single_use_refresh_lock_timeout(),
                expected_source_path=auth_mod._auth_file_path(),
            ) as (_auth_store, state, source_path):
                source_path = source_path or auth_mod._auth_file_path()
                if not self._codex_provider_state_has_refresh_chain(state):
                    self._revoke_missing_codex_device_code_source(
                        entry,
                        source_path,
                    )
                    return None
                synced = self._codex_entry_from_provider_state(entry, state)
                if synced is not entry:
                    # A waiter that observed the winner's rotated chain adopts
                    # it even when its caller requested force=True. Replaying
                    # the consumed refresh token would invalidate the chain.
                    replaced = self._replace_entry(
                        entry,
                        synced,
                        preserve_routing=True,
                    )
                    if replaced is None or not self._entry_tokens_match(
                        replaced,
                        synced,
                    ):
                        return replaced
                    self._persist_codex_device_code_refresh(
                        replaced,
                        source_path,
                    )
                    return replaced
                return self._refresh_entry_impl(
                    entry,
                    force=force,
                    codex_source_path=source_path,
                    codex_singleton_owned=True,
                )
        if self.provider == "openai-codex":
            # Manual Codex accounts are profile-owned and independent of the
            # singleton, but their own single-use refresh still needs the
            # active profile's pool-store lock.
            with _auth_store_lock(
                timeout_seconds=self._single_use_refresh_lock_timeout()
            ):
                return self._refresh_entry_impl(entry, force=force)
        if self.provider == "xai-oauth":
            if entry.source != "device_code":
                with _auth_store_lock(
                    timeout_seconds=self._single_use_refresh_lock_timeout()
                ):
                    return self._refresh_entry_impl(entry, force=force)
            return self._refresh_xai_source_entry(entry, force=force)
        if self.provider == "nous" and entry.source == "device_code":
            # The real resolver owns the complete provider-state transaction.
            # Do not nest a second source lock around its network operation.
            return self._refresh_entry_impl(entry, force=force)
        return self._refresh_entry_impl(entry, force=force)

    def _mark_source_persistence_dead(
        self,
        entry: PooledCredential,
        *,
        persist: bool = True,
    ) -> None:
        updated = replace(
            entry,
            last_status=STATUS_DEAD,
            last_status_at=time.time(),
            last_error_reason="source_persistence_failed",
            last_error_message=(
                "Refreshed credentials were not persisted to their owning source."
            ),
        )
        self._replace_entry(
            entry,
            updated,
            preserve_routing=True,
            mark_dirty=persist,
        )

    def _quarantine_terminal_xai_lineage(
        self,
        entry: PooledCredential,
        source_path: Optional[Path],
        exc: Exception,
    ) -> None:
        """Remove one terminal singleton lineage without pool-lock I/O."""
        if source_path is not None:
            try:
                with auth_mod._provider_state_transaction(
                    "xai-oauth",
                    timeout_seconds=self._single_use_refresh_lock_timeout(),
                    expected_source_path=source_path,
                ) as (_auth_store, state, locked_source_path):
                    tokens = state.get("tokens") if isinstance(state, dict) else None
                    if (
                        locked_source_path is not None
                        and isinstance(tokens, dict)
                        and str(tokens.get("refresh_token") or "").strip()
                        == str(entry.refresh_token or "").strip()
                    ):
                        source_store = _load_auth_store(Path(locked_source_path))
                        providers = source_store.get("providers")
                        if isinstance(providers, dict):
                            providers.pop("xai-oauth", None)
                        pool = source_store.get("credential_pool")
                        if isinstance(pool, dict) and isinstance(
                            pool.get("xai-oauth"), list
                        ):
                            pool["xai-oauth"] = [
                                item
                                for item in pool["xai-oauth"]
                                if not (
                                    isinstance(item, dict)
                                    and item.get("source") == "device_code"
                                )
                            ]
                        _save_auth_store(
                            source_store,
                            target_path=Path(locked_source_path),
                        )
            except Exception as clear_exc:
                logger.debug(
                    "Failed to clear terminal xAI OAuth state after %s: %s",
                    exc,
                    clear_exc,
                )

        singleton_sources = {"device_code"}
        with self._lock:
            removed_entries = [
                item for item in self._entries if item.source in singleton_sources
            ]
            for removed in removed_entries:
                self._queue_removed_entry_unlocked(removed)
            self._entries = [
                item for item in self._entries if item.source not in singleton_sources
            ]
            if self._current_id == entry.id:
                self._current_id = None

    def _anthropic_entry_from_locked_credentials(
        self,
        entry: PooledCredential,
        creds: Dict[str, Any],
    ) -> PooledCredential:
        access_token = str(creds.get("accessToken", "") or "")
        refresh_token = str(creds.get("refreshToken", "") or "")
        changed = (
            access_token != entry.access_token
            or refresh_token != (entry.refresh_token or "")
        )
        return replace(
            entry,
            access_token=access_token or entry.access_token,
            refresh_token=refresh_token or entry.refresh_token,
            expires_at_ms=creds.get("expiresAt", 0) or entry.expires_at_ms,
            last_status=STATUS_OK if changed else entry.last_status,
            last_status_at=None if changed else entry.last_status_at,
            last_error_code=None if changed else entry.last_error_code,
            last_error_reason=None if changed else entry.last_error_reason,
            last_error_message=None if changed else entry.last_error_message,
            last_error_reset_at=None if changed else entry.last_error_reset_at,
            extra={
                **entry.extra,
                "credential_source": creds.get("source"),
            },
        )

    def _refresh_anthropic_source_entry(
        self,
        entry: PooledCredential,
        *,
        force: bool,
    ) -> Optional[PooledCredential]:
        from agent.anthropic_adapter import (
            _refresh_claude_code_source_credentials,
        )

        del force  # A source-owned explicit refresh always reaches this method.
        observed = {
            "accessToken": entry.access_token,
            "refreshToken": entry.refresh_token or "",
            "expiresAt": entry.expires_at_ms or 0,
            "source": entry.extra.get("credential_source")
            or "claude_code_credentials_file",
        }
        try:
            winner = _refresh_claude_code_source_credentials(observed)
        except auth_mod.SourceCredentialPersistenceError:
            self._mark_source_persistence_dead(entry, persist=False)
            return None
        if not winner:
            self._mark_source_persistence_dead(entry, persist=False)
            return None
        current = self._current_refresh_candidate(entry) or entry
        updated = self._anthropic_entry_from_locked_credentials(
            current,
            winner,
        )
        return self._replace_entry(
            current,
            updated,
            preserve_routing=True,
        )

    def _xai_entry_from_locked_state(
        self,
        entry: PooledCredential,
        state: Dict[str, Any],
        source_path: Path,
    ) -> Optional[PooledCredential]:
        tokens = state.get("tokens")
        if not isinstance(tokens, dict):
            return None
        access_token = str(tokens.get("access_token", "") or "")
        refresh_token = str(tokens.get("refresh_token", "") or "")
        if not access_token or not refresh_token:
            return None
        changed = (
            access_token != entry.access_token
            or refresh_token != (entry.refresh_token or "")
        )
        return replace(
            entry,
            access_token=access_token,
            refresh_token=refresh_token,
            last_refresh=state.get("last_refresh") or entry.last_refresh,
            source_store_path=source_path,
            last_status=None if changed else entry.last_status,
            last_status_at=None if changed else entry.last_status_at,
            last_error_code=None if changed else entry.last_error_code,
            last_error_reason=None if changed else entry.last_error_reason,
            last_error_message=None if changed else entry.last_error_message,
            last_error_reset_at=None if changed else entry.last_error_reset_at,
        )

    def _refresh_xai_source_entry(
        self,
        entry: PooledCredential,
        *,
        force: bool,
    ) -> Optional[PooledCredential]:
        observed_source_path, observed_refresh_fingerprint = (
            auth_mod._observe_provider_refresh_source("xai-oauth")
        )
        expected_source_path = observed_source_path
        winner: Optional[PooledCredential] = None
        refresh_failed = False
        for _attempt in range(3):
            with auth_mod._provider_state_transaction(
                "xai-oauth",
                timeout_seconds=self._single_use_refresh_lock_timeout(),
                expected_source_path=expected_source_path,
            ) as (transaction_store, state, source_path):
                if source_path is None or state is None:
                    expected_source_path = None
                    continue
                resolved_state, resolved_source_path = (
                    auth_mod._load_provider_state_with_source(
                        transaction_store,
                        "xai-oauth",
                    )
                )
                if (
                    resolved_state is not None
                    and resolved_source_path is not None
                    and not auth_mod._same_path(resolved_source_path, Path(source_path))
                ):
                    state = resolved_state
                    source_path = resolved_source_path
                if auth_mod._source_state_is_reserved(state):
                    refresh_failed = True
                    break
                authoritative = self._xai_entry_from_locked_state(
                    entry,
                    state,
                    Path(source_path),
                )
                if authoritative is None:
                    refresh_failed = True
                    break
                current_refresh_fingerprint = auth_mod._refresh_lineage_fingerprint(
                    authoritative.refresh_token
                )
                owner_changed = bool(
                    observed_source_path is not None
                    and not auth_mod._same_path(observed_source_path, Path(source_path))
                )
                lineage_changed = bool(
                    observed_refresh_fingerprint is not None
                    and current_refresh_fingerprint != observed_refresh_fingerprint
                )
                if (
                    (
                        owner_changed
                        or lineage_changed
                        or not self._entry_tokens_match(authoritative, entry)
                    )
                    and not auth_mod._xai_access_token_is_expiring(
                        authoritative.access_token,
                        0,
                    )
                ):
                    winner = authoritative
                    break
                if not force and not self._entry_needs_refresh(authoritative):
                    winner = authoritative
                    break
                discovery = dict(state.get("discovery") or {})
                token_endpoint = str(
                    discovery.get("token_endpoint", "") or ""
                ).strip()
                if not token_endpoint:
                    token_endpoint = auth_mod._xai_oauth_discovery(
                        auth_mod.env_float("HERMES_XAI_REFRESH_TIMEOUT_SECONDS", 20)
                    )["token_endpoint"]
                consumed_refresh_token = authoritative.refresh_token or ""
                reservation = auth_mod._reserve_provider_refresh_source(
                    "xai-oauth",
                    state,
                    source_path,
                    expected_refresh_token=consumed_refresh_token,
                )
                try:
                    refreshed = auth_mod.refresh_xai_oauth_pure(
                        authoritative.access_token,
                        consumed_refresh_token,
                        token_endpoint=token_endpoint,
                        timeout_seconds=auth_mod.env_float(
                            "HERMES_XAI_REFRESH_TIMEOUT_SECONDS",
                            20,
                        ),
                    )
                except Exception:
                    refresh_failed = True
                    break
                updated_state = auth_mod._xai_oauth_state_with_refreshed_tokens(
                    state,
                    refreshed,
                    token_endpoint=token_endpoint,
                )
                try:
                    auth_mod._finalize_provider_refresh_reservation(
                        reservation,
                        updated_state,
                    )
                except auth_mod.SourceCredentialLineageChanged:
                    continue
                except OSError:
                    refresh_failed = True
                    break
                winner = replace(
                    entry,
                    access_token=refreshed["access_token"],
                    refresh_token=refreshed["refresh_token"],
                    last_refresh=refreshed.get("last_refresh"),
                    source_store_path=Path(source_path),
                    last_status=STATUS_OK,
                    last_status_at=None,
                    last_error_code=None,
                    last_error_reason=None,
                    last_error_message=None,
                    last_error_reset_at=None,
                )
                break
        if refresh_failed:
            self._mark_source_persistence_dead(entry, persist=False)
            return None
        if winner is None:
            return None
        return self._replace_entry(
            entry,
            winner,
            preserve_routing=True,
        )

    def _current_refresh_candidate(
        self,
        expected: PooledCredential,
    ) -> Optional[PooledCredential]:
        """Return the current row only when its refresh lineage still matches."""
        with self._lock:
            current = next(
                (item for item in self._entries if item.id == expected.id),
                None,
            )
        if current is None:
            return None
        if (
            current.source != expected.source
            or not self._entry_tokens_match(current, expected)
        ):
            return None
        return current

    def _single_use_refresh_lock_timeout(self) -> float:
        """Lock timeout for single-use-refresh-token providers.

        Covers the configured refresh POST timeout plus a margin so a slow
        token endpoint cannot make the flock give up before the refresh
        resolves.  Reads the provider's ``HERMES_*_REFRESH_TIMEOUT_SECONDS``
        override.
        """
        env_var = (
            "HERMES_CODEX_REFRESH_TIMEOUT_SECONDS"
            if self.provider == "openai-codex"
            else "HERMES_XAI_REFRESH_TIMEOUT_SECONDS"
        )
        refresh_timeout_seconds = auth_mod.env_float(env_var, 20)
        return max(
            float(auth_mod.AUTH_LOCK_TIMEOUT_SECONDS),
            float(refresh_timeout_seconds) + 5.0,
        )

    def _commit_refreshed_entry(
        self,
        expected: PooledCredential,
        refreshed: PooledCredential,
        *,
        codex_source_path: Optional[Path] = None,
        codex_singleton_owned: bool = False,
    ) -> Optional[PooledCredential]:
        """Commit refreshed tokens only while their source lineage is current."""
        if (
            self.provider == "anthropic" and expected.source == "claude_code"
        ) or (
            self.provider == "xai-oauth" and expected.source == "device_code"
        ):
            # These rotating owners must use their dedicated pre-POST source
            # transaction. Refuse the legacy post-POST source commit path.
            return None

        committed = self._replace_entry(
            expected,
            refreshed,
            preserve_routing=True,
        )
        if committed is None:
            return None
        if not self._entry_tokens_match(committed, refreshed):
            return committed

        if self.provider == "openai-codex" and codex_source_path is not None:
            with self._external_state_lock:
                current = self._current_refresh_candidate(committed)
                if current is None:
                    return None
                if codex_singleton_owned:
                    self._persist_codex_device_code_refresh(
                        current,
                        codex_source_path,
                    )
                elif not self._persist_codex_exact_alias_refresh(
                    current,
                    codex_source_path,
                ):
                    return None
                return current

        with self._external_state_lock:
            return self._current_refresh_candidate(committed)

    def _revalidate_refreshed_entry(
        self,
        refreshed: PooledCredential,
    ) -> Optional[PooledCredential]:
        """Return only a current row that still matches its owning source."""

        if self.provider == "anthropic" and refreshed.source == "claude_code":
            from agent.anthropic_adapter import claude_code_credentials_transaction

            with claude_code_credentials_transaction(
                timeout_seconds=self._single_use_refresh_lock_timeout(),
            ) as (_source_path, creds):
                if not creds:
                    return None
                source_snapshot = dict(creds)
            with self._external_state_lock:
                with self._lock:
                    current = next(
                        (
                            item
                            for item in self._entries
                            if item.id == refreshed.id
                        ),
                        None,
                    )
                if current is None or current.source != refreshed.source:
                    return None
                authoritative = self._anthropic_entry_from_locked_credentials(
                    current,
                    source_snapshot,
                )
                if not self._entry_tokens_match(authoritative, current):
                    return self._replace_entry(
                        current,
                        authoritative,
                        preserve_routing=True,
                    )
                return current

        if self.provider == "xai-oauth" and refreshed.source == "device_code":
            expected_source_path = (
                refreshed.source_store_path or auth_mod._auth_file_path()
            )
            with auth_mod._provider_state_transaction(
                "xai-oauth",
                timeout_seconds=self._single_use_refresh_lock_timeout(),
                expected_source_path=expected_source_path,
            ) as (_auth_store, state, source_path):
                if state is None or source_path is None:
                    return None
                source_snapshot = dict(state)
                snapshot_path = Path(source_path)
            with self._external_state_lock:
                with self._lock:
                    current = next(
                        (
                            item
                            for item in self._entries
                            if item.id == refreshed.id
                        ),
                        None,
                    )
                if current is None or current.source != refreshed.source:
                    return None
                authoritative = self._xai_entry_from_locked_state(
                    current,
                    source_snapshot,
                    snapshot_path,
                )
                if authoritative is None:
                    return None
                if not self._entry_tokens_match(authoritative, current):
                    return self._replace_entry(
                        current,
                        authoritative,
                        preserve_routing=True,
                    )
                return current

        if self.provider == "nous" and refreshed.source == "device_code":
            auth_store = _load_auth_store()
            _state, source_path = _load_provider_state_with_source(
                auth_store,
                self.provider,
            )
            with auth_mod._provider_state_transaction(
                self.provider,
                timeout_seconds=self._single_use_refresh_lock_timeout(),
                expected_source_path=source_path,
            ) as (_auth_store, state, _locked_source_path):
                if not isinstance(state, dict) or not state.get("access_token"):
                    return None
                source_snapshot = dict(state)
            with self._external_state_lock:
                with self._lock:
                    current = next(
                        (
                            item
                            for item in self._entries
                            if item.id == refreshed.id
                        ),
                        None,
                    )
                if current is None or current.source != refreshed.source:
                    return None
                authoritative = replace(
                    current,
                    access_token=str(source_snapshot.get("access_token") or ""),
                    refresh_token=str(source_snapshot.get("refresh_token") or ""),
                    expires_at=source_snapshot.get("expires_at"),
                    agent_key=source_snapshot.get("agent_key"),
                    agent_key_expires_at=source_snapshot.get("agent_key_expires_at"),
                    inference_base_url=source_snapshot.get("inference_base_url"),
                )
                if not self._entry_tokens_match(authoritative, current):
                    return self._replace_entry(
                        current,
                        authoritative,
                        preserve_routing=True,
                    )
                return current

        with self._external_state_lock:
            with self._lock:
                return next(
                    (
                        item
                        for item in self._entries
                        if item.id == refreshed.id
                        and item.source == refreshed.source
                        and self._entry_tokens_match(item, refreshed)
                    ),
                    None,
                )

    def _refresh_entry_impl(
        self,
        entry: PooledCredential,
        *,
        force: bool,
        codex_source_path: Optional[Path] = None,
        codex_singleton_owned: bool = False,
    ) -> Optional[PooledCredential]:
        current = self._current_refresh_candidate(entry)
        if current is None:
            return None
        entry = current
        try:
            if self.provider == "anthropic":
                from agent.anthropic_adapter import refresh_anthropic_oauth_pure

                refreshed = refresh_anthropic_oauth_pure(
                    entry.refresh_token,
                    use_json=entry.source.endswith("hermes_pkce"),
                )
                updated = replace(
                    entry,
                    access_token=refreshed["access_token"],
                    refresh_token=refreshed["refresh_token"],
                    expires_at_ms=refreshed["expires_at_ms"],
                )
            elif self.provider == "openai-codex":
                # Adopt fresher tokens from auth.json before spending the
                # refresh_token — single-use tokens consumed by another Hermes
                # process sharing the same auth.json singleton would otherwise
                # trigger ``refresh_token_reused`` on the next POST.
                if codex_source_path is None:
                    synced = self._sync_codex_entry_from_auth_store(entry)
                    if synced is not entry:
                        entry = synced
                current = self._current_refresh_candidate(entry)
                if current is None:
                    return None
                entry = current
                refreshed = auth_mod.refresh_codex_oauth_pure(
                    entry.access_token,
                    entry.refresh_token,
                )
                updated = replace(
                    entry,
                    access_token=refreshed["access_token"],
                    refresh_token=refreshed["refresh_token"],
                    last_refresh=refreshed.get("last_refresh"),
                )
            elif self.provider == "xai-oauth":
                # Adopt fresher tokens from auth.json before spending the
                # refresh_token — single-use tokens consumed by another
                # process (or another profile sharing the singleton) would
                # otherwise trigger ``refresh_token_reused`` on the next
                # POST.  Only meaningful for singleton-seeded entries.
                synced = self._sync_xai_oauth_entry_from_auth_store(
                    entry,
                    persist=False,
                )
                if synced is not entry:
                    entry = synced
                current = self._current_refresh_candidate(entry)
                if current is None:
                    return None
                entry = current
                refreshed = auth_mod.refresh_xai_oauth_pure(
                    entry.access_token,
                    entry.refresh_token,
                )
                updated = replace(
                    entry,
                    access_token=refreshed["access_token"],
                    refresh_token=refreshed["refresh_token"],
                    last_refresh=refreshed.get("last_refresh"),
                )
            elif self.provider == "nous":
                observed_refresh_token = entry.refresh_token
                synced = self._sync_nous_entry_from_auth_store(
                    entry,
                    persist=False,
                )
                if synced is not entry:
                    entry = synced
                if (
                    entry.refresh_token != observed_refresh_token
                    and not self._entry_needs_refresh(entry)
                ):
                    return entry
                current = self._current_refresh_candidate(entry)
                if current is None:
                    return None
                entry = current
                auth_mod.resolve_nous_runtime_credentials(
                    force_refresh=force,
                )
                updated = self._sync_nous_entry_from_auth_store(
                    entry,
                    persist=False,
                )
            else:
                return entry
        except Exception as exc:
            logger.debug("Credential refresh failed for %s/%s: %s", self.provider, entry.id, exc)
            if (
                self.provider == "nous"
                and isinstance(exc, auth_mod.SourceCredentialPersistenceError)
            ):
                # The exact owner is already durably reserved. Keep this pool
                # instance unavailable without creating a borrower-local copy.
                self._mark_source_persistence_dead(entry, persist=False)
                return None
            # For anthropic claude_code entries: the refresh token may have been
            # consumed by another process. Check if ~/.claude/.credentials.json
            # has a newer token pair and retry once.
            if self.provider == "anthropic" and entry.source == "claude_code":
                synced = self._sync_anthropic_entry_from_credentials_file(
                    entry,
                    persist=False,
                )
                if synced.refresh_token != entry.refresh_token:
                    logger.debug("Retrying refresh with synced token from credentials file")
                    try:
                        from agent.anthropic_adapter import refresh_anthropic_oauth_pure
                        refreshed = refresh_anthropic_oauth_pure(
                            synced.refresh_token,
                            use_json=synced.source.endswith("hermes_pkce"),
                        )
                        updated = replace(
                            synced,
                            access_token=refreshed["access_token"],
                            refresh_token=refreshed["refresh_token"],
                            expires_at_ms=refreshed["expires_at_ms"],
                            last_status=STATUS_OK,
                            last_status_at=None,
                            last_error_code=None,
                        )
                        return self._commit_refreshed_entry(synced, updated)
                    except Exception as retry_exc:
                        logger.debug("Retry refresh also failed: %s", retry_exc)
                elif not self._entry_needs_refresh(synced):
                    # Credentials file had a valid (non-expired) token — use it directly
                    logger.debug("Credentials file has valid token, using without refresh")
                    return synced
            # For xai-oauth: same race as nous — another process may have
            # consumed the refresh token between our proactive sync and the
            # HTTP call.  Re-check auth.json and adopt the fresh tokens if
            # they have rotated since.  Only meaningful for singleton-seeded
            # (device_code) entries; manual entries don't share
            # state with the singleton.
            if self.provider == "xai-oauth":
                synced = self._sync_xai_oauth_entry_from_auth_store(
                    entry,
                    persist=False,
                )
                if synced.refresh_token != entry.refresh_token:
                    logger.debug(
                        "xAI OAuth refresh failed but auth.json has newer tokens — adopting"
                    )
                    updated = replace(
                        synced,
                        last_status=STATUS_OK,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                    committed = self._replace_entry(
                        synced,
                        updated,
                        preserve_routing=True,
                    )
                    if committed is None:
                        return None
                    if not self._entry_tokens_match(committed, updated):
                        return committed
                    return committed
                # Terminal error: auth.json has no newer tokens — the stored
                # refresh_token is dead.  Clear it from auth.json so the next
                # session does not re-seed the same revoked credentials, and
                # remove all singleton-seeded xAI entries from the in-memory
                # pool. Mirrors the Nous quarantine path above.
                if auth_mod._is_terminal_xai_oauth_refresh_error(exc):
                    logger.debug(
                        "xAI OAuth refresh token is terminally invalid; clearing local token state"
                    )
                    try:
                        with _auth_store_lock():
                            auth_store = _load_auth_store()
                            state = _load_provider_state(auth_store, "xai-oauth") or {}
                            if isinstance(state, dict):
                                tokens = state.get("tokens") or {}
                                if isinstance(tokens, dict):
                                    store_refresh = str(tokens.get("refresh_token") or "").strip()
                                    entry_refresh = str(entry.refresh_token or "").strip()
                                    if not store_refresh or store_refresh == entry_refresh:
                                        tokens.pop("access_token", None)
                                        tokens.pop("refresh_token", None)
                                        state["tokens"] = tokens
                                        state["last_auth_error"] = {
                                            "provider": "xai-oauth",
                                            "code": getattr(exc, "code", "unknown"),
                                            "message": str(exc),
                                            "reason": "credential_pool_refresh_failure",
                                            "relogin_required": True,
                                            "at": datetime.now(timezone.utc).isoformat(),
                                        }
                                        _save_provider_state(auth_store, "xai-oauth", state)
                                        _save_auth_store(auth_store)
                    except Exception as clear_exc:
                        logger.debug(
                            "Failed to clear terminal xAI OAuth state: %s", clear_exc
                        )
                    # Read-modify-write of self._entries: must be atomic.
                    # This runs on the DEFERRED refresh path (outside the
                    # pool lock), so take it here. self._lock is an RLock,
                    # so the still-locked callers re-enter safely.
                    with self._lock:
                        removed_entries = [
                            item for item in self._entries
                            if item.source == "device_code"
                        ]
                        for removed in removed_entries:
                            self._queue_removed_entry_unlocked(removed)
                        self._entries = [
                            item for item in self._entries
                            if item.source != "device_code"
                        ]
                        if self._current_id == entry.id:
                            self._current_id = None
                    return None
            # For openai-codex: same race as xAI/nous — another Hermes process
            # may have consumed the refresh token between our proactive sync
            # and the HTTP call.  Re-check auth.json and adopt the fresh tokens
            # if they have rotated since.
            if self.provider == "openai-codex":
                if codex_source_path is not None:
                    source_store = _load_auth_store(codex_source_path)
                    if codex_singleton_owned:
                        providers = source_store.get("providers")
                        state = (
                            providers.get("openai-codex")
                            if isinstance(providers, dict)
                            else None
                        )
                        synced = self._codex_entry_from_provider_state(entry, state)
                    else:
                        _persisted, _index, item = self._codex_exact_pool_alias(
                            source_store,
                            entry,
                        )
                        if self._codex_pool_alias_has_complete_credentials(item):
                            assert isinstance(item, dict)
                            synced = replace(
                                PooledCredential.from_dict("openai-codex", item),
                                source_store_path=codex_source_path,
                            )
                        else:
                            self._revoke_missing_codex_source(
                                entry,
                                codex_source_path,
                                source_store=source_store,
                            )
                            return None
                else:
                    synced = self._sync_codex_entry_from_auth_store(entry)
                if synced.refresh_token != entry.refresh_token:
                    logger.debug(
                        "Codex OAuth refresh failed but auth.json has newer tokens — adopting"
                    )
                    updated = replace(
                        synced,
                        last_status=STATUS_OK,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                    committed = self._replace_entry(
                        synced,
                        updated,
                        preserve_routing=True,
                    )
                    if committed is None:
                        return None
                    if not self._entry_tokens_match(committed, updated):
                        return committed
                    if codex_source_path is not None:
                        if codex_singleton_owned:
                            self._persist_codex_device_code_refresh(
                                committed, codex_source_path
                            )
                        else:
                            self._persist_codex_exact_alias_refresh(
                                committed, codex_source_path
                            )
                    return committed
                # Terminal error: auth.json has no newer tokens — the stored
                # refresh_token is dead.  Clear it from auth.json so the next
                # session does not re-seed the same revoked credentials, and
                # remove all singleton-seeded (device_code) entries from the
                # in-memory pool.  Mirrors the xAI and Nous quarantine paths.
                if auth_mod._is_terminal_codex_oauth_refresh_error(exc):
                    logger.debug(
                        "Codex OAuth refresh token is terminally invalid; clearing local token state"
                    )
                    removed_ids = {entry.id}
                    try:
                        if codex_source_path is not None:
                            if codex_singleton_owned:
                                removed_ids = (
                                    self._quarantine_codex_device_code_refresh(
                                        entry, codex_source_path, exc
                                    )
                                )
                            else:
                                self._quarantine_codex_exact_alias_refresh(
                                    entry, codex_source_path
                                )
                                removed_ids = {entry.id}
                        else:
                            removed_ids = {entry.id}
                            with _auth_store_lock():
                                auth_store = _load_auth_store()
                                state = (
                                    _load_provider_state(auth_store, "openai-codex")
                                    or {}
                                )
                                if isinstance(state, dict):
                                    tokens = state.get("tokens") or {}
                                    if isinstance(tokens, dict):
                                        store_refresh = str(
                                            tokens.get("refresh_token") or ""
                                        ).strip()
                                        entry_refresh = str(
                                            entry.refresh_token or ""
                                        ).strip()
                                        if (
                                            not store_refresh
                                            or store_refresh == entry_refresh
                                        ):
                                            tokens.pop("access_token", None)
                                            tokens.pop("refresh_token", None)
                                            state["tokens"] = tokens
                                            state["last_auth_error"] = {
                                                "provider": "openai-codex",
                                                "code": getattr(
                                                    exc, "code", "unknown"
                                                ),
                                                "message": str(exc),
                                                "reason": (
                                                    "credential_pool_refresh_failure"
                                                ),
                                                "relogin_required": True,
                                                "at": datetime.now(
                                                    timezone.utc
                                                ).isoformat(),
                                            }
                                            _save_provider_state(
                                                auth_store,
                                                "openai-codex",
                                                state,
                                            )
                                            _save_auth_store(auth_store)
                    except Exception as clear_exc:
                        logger.debug(
                            "Failed to clear terminal Codex OAuth state: %s", clear_exc
                        )
                    # Read-modify-write of self._entries: must be atomic.
                    # This runs on the DEFERRED refresh path (outside the
                    # pool lock), so take it here. self._lock is an RLock,
                    # so the still-locked callers re-enter safely.
                    with self._lock:
                        removed_entries = [
                            item for item in self._entries if item.id in removed_ids
                        ]
                        for removed in removed_entries:
                            self._queue_removed_entry_unlocked(removed)
                        self._entries = [
                            item for item in self._entries if item.id not in removed_ids
                        ]
                        if self._current_id in removed_ids:
                            self._current_id = None
                    return None
            # For nous: another process may have consumed the refresh token
            # between our proactive sync and the HTTP call.  Re-sync from
            # auth.json and adopt the fresh tokens if available.
            if self.provider == "nous":
                synced = self._sync_nous_entry_from_auth_store(
                    entry,
                    persist=False,
                )
                if synced.refresh_token != entry.refresh_token:
                    logger.debug("Nous refresh failed but auth.json has newer tokens — adopting")
                    updated = replace(
                        synced,
                        last_status=STATUS_OK,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                    committed = self._replace_entry(
                        synced,
                        updated,
                        preserve_routing=True,
                    )
                    if committed is None:
                        return None
                    if not self._entry_tokens_match(committed, updated):
                        return committed
                    return committed
                if auth_mod._is_terminal_nous_refresh_error(exc):
                    logger.debug("Nous refresh token is terminally invalid; reconciling source state")
                    reserved_owner = False
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
                            reserved_owner = auth_mod._source_state_is_reserved(state)
                            store_refresh = str(state.get("refresh_token") or "").strip()
                            entry_refresh = str(entry.refresh_token or "").strip()
                            if (
                                not reserved_owner
                                and (not store_refresh or store_refresh == entry_refresh)
                            ):
                                auth_mod._quarantine_nous_oauth_state(
                                    state,
                                    exc,
                                    reason="credential_pool_refresh_failure",
                                )
                                auth_mod._quarantine_nous_pool_entries(
                                    auth_store,
                                    exc,
                                    reason="credential_pool_refresh_failure",
                                )
                                _save_provider_state(auth_store, "nous", state)
                                _save_auth_store(auth_store)
                    except Exception as clear_exc:
                        logger.debug("Failed to clear terminal Nous OAuth state: %s", clear_exc)

                    if reserved_owner:
                        self._mark_source_persistence_dead(entry, persist=False)
                        return None

                    singleton_sources = {
                        auth_mod.NOUS_DEVICE_CODE_SOURCE,
                        f"manual:{auth_mod.NOUS_DEVICE_CODE_SOURCE}",
                    }
                    # Atomic read-modify-write; see the note above.
                    with self._lock:
                        removed_entries = [
                            item for item in self._entries
                            if item.source in singleton_sources
                        ]
                        for removed in removed_entries:
                            self._queue_removed_entry_unlocked(removed)
                        self._entries = [
                            item for item in self._entries
                            if item.source not in singleton_sources
                        ]
                        if self._current_id == entry.id:
                            self._current_id = None
                    return None
            if self.provider == "openai-codex" and codex_source_path is not None:
                exhausted = self._mark_exhausted(entry, None, persist=False)
                if codex_singleton_owned:
                    self._persist_codex_device_code_refresh(
                        exhausted, codex_source_path
                    )
                else:
                    self._persist_codex_exact_alias_refresh(
                        exhausted, codex_source_path
                    )
            else:
                self._mark_exhausted(entry, None, persist=False)
            return None

        updated = replace(
            updated,
            last_status=STATUS_OK,
            last_status_at=None,
            last_error_code=None,
            last_error_reason=None,
            last_error_message=None,
            last_error_reset_at=None,
        )
        return self._commit_refreshed_entry(
            entry,
            updated,
            codex_source_path=codex_source_path,
            codex_singleton_owned=codex_singleton_owned,
        )

    def _codex_quota_restored_upstream(self, entry: PooledCredential) -> bool:
        """Live-check whether an exhausted Codex entry's quota reset early.

        A Codex 429 persists a ``last_error_reset_at`` that can be days in
        the future (weekly windows), but the upstream window can reopen
        before then — the user redeems a banked rate-limit reset via the
        Codex CLI / ChatGPT UI, upgrades their plan, or OpenAI resets the
        window.  Without this check the pool keeps the credential frozen
        until the stale timestamp elapses even though the account is
        usable (issue #43747).

        Only fires for openai-codex entries frozen by a 429/quota-shaped
        error.  The underlying probe is throttled per token (5 min) so this
        is safe on the hot selection path.
        """
        if self.provider != "openai-codex" or entry.last_status != STATUS_EXHAUSTED:
            return False
        if not auth_mod._is_codex_rate_limit_shaped(
            entry.last_error_code,
            entry.last_error_reason,
            entry.last_error_message,
        ):
            return False
        token = entry.access_token or ""
        if not token:
            return False
        try:
            return bool(
                auth_mod._probe_codex_quota_restored(
                    token,
                    base_url=entry.base_url,
                )
            )
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
            return _codex_access_token_is_expiring(
                entry.access_token,
                CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
            )
        if self.provider == "xai-oauth":
            return auth_mod._xai_access_token_is_expiring(
                entry.access_token,
                auth_mod._xai_proactive_refresh_skew_seconds(entry.access_token),
            )
        if self.provider == "nous":
            # Nous refresh can require network access and should happen when
            # runtime credentials are actually resolved, not merely when the pool
            # is enumerated for listing, migration, or selection.
            return False
        return False

    def _assert_external_state_lock_order(self) -> None:
        """Reject external synchronization while the pool lock is held."""
        is_owned = getattr(self._lock, "_is_owned", None)
        if callable(is_owned) and is_owned():
            raise RuntimeError(
                "External credential state cannot be synchronized while the "
                "credential-pool lock is held"
            )

    def _sync_external_status_entries(self, *, probe_quota: bool = False) -> None:
        """Refresh external status sources without holding the pool lock."""
        self._assert_external_state_lock_order()
        self._sync_external_status_entries_locked(probe_quota=probe_quota)

    def _sync_external_status_entries_locked(
        self,
        *,
        probe_quota: bool = False,
    ) -> None:
        with self._lock:
            status_entries = [
                entry
                for entry in self._entries
                if entry.last_status in {STATUS_EXHAUSTED, STATUS_DEAD}
            ]

        for entry in status_entries:
            if self.provider == "anthropic" and entry.source == "claude_code":
                self._sync_anthropic_entry_from_credentials_file(
                    entry,
                    persist=False,
                )
            elif self.provider == "nous" and entry.source == "device_code":
                self._sync_nous_entry_from_auth_store(entry, persist=False)
            elif (
                self.provider == "openai-codex"
                and entry.source == "device_code"
                and entry.source_store_path is None
            ):
                self._sync_codex_entry_from_auth_store(entry)
            elif self.provider == "xai-oauth" and entry.source == "device_code":
                self._sync_xai_oauth_entry_from_auth_store(entry, persist=False)

        if not probe_quota:
            return
        with self._lock:
            quota_entries = [
                entry
                for entry in self._entries
                if entry.last_status == STATUS_EXHAUSTED
            ]
        for entry in quota_entries:
            if not self._codex_quota_restored_upstream(entry):
                continue
            cleared = replace(
                entry,
                last_status=STATUS_OK,
                last_status_at=None,
                last_error_code=None,
                last_error_reason=None,
                last_error_message=None,
                last_error_reset_at=None,
            )
            self._replace_entry(entry, cleared)

    def _converge_empty_xai_source_selection(self) -> bool:
        """Adopt a durable xAI singleton winner after an in-flight reservation.

        A pool loaded while another process owns the source refresh transaction
        sees the deliberately secret-free reservation row and therefore has no
        selectable entry. Wait for that exact source transaction, snapshot its
        finalized row while still locked, then hydrate this pool only after the
        source lock is released. A stranded reservation remains fail-closed.
        """
        if self.provider != "xai-oauth":
            return False

        try:
            with auth_mod._provider_state_transaction(
                "xai-oauth",
                timeout_seconds=self._single_use_refresh_lock_timeout(),
            ) as (_auth_store, state, source_path):
                if (
                    source_path is None
                    or not isinstance(state, dict)
                    or auth_mod._source_state_is_reserved(state)
                ):
                    return False
                tokens = state.get("tokens")
                if not isinstance(tokens, dict):
                    return False
                access_token = str(tokens.get("access_token") or "").strip()
                if not access_token:
                    return False

                source_path = Path(source_path)
                source_store = _load_auth_store(source_path)
                source_pool = source_store.get("credential_pool")
                source_rows = (
                    source_pool.get("xai-oauth")
                    if isinstance(source_pool, dict)
                    else None
                )
                candidates = [
                    PooledCredential.from_dict("xai-oauth", row)
                    for row in source_rows or []
                    if isinstance(row, dict)
                ]
                _upsert_entry(
                    candidates,
                    "xai-oauth",
                    "device_code",
                    {
                        "source": "device_code",
                        "auth_type": AUTH_TYPE_OAUTH,
                        "access_token": access_token,
                        "refresh_token": tokens.get("refresh_token"),
                        "base_url": auth_mod.DEFAULT_XAI_OAUTH_BASE_URL,
                        "last_refresh": state.get("last_refresh"),
                        "label": label_from_token(access_token, "device_code"),
                    },
                )
                winner = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.source == "device_code"
                    ),
                    None,
                )
                if winner is None:
                    return False
                winner = replace(winner, source_store_path=source_path)
        except Exception as exc:
            logger.debug("Failed to converge empty xAI OAuth pool: %s", exc)
            return False

        # The source transaction is intentionally over before pool mutation.
        # Another selector may already have converged this instance; preserve
        # its complete entry instead of replacing a newer in-memory snapshot.
        with self._external_state_lock:
            with self._lock:
                if any(
                    entry.source == "device_code" and entry.access_token
                    for entry in self._entries
                ):
                    return True
                stale_ids = {
                    entry.id
                    for entry in self._entries
                    if entry.source == "device_code"
                }
                self._entries = [
                    entry
                    for entry in self._entries
                    if entry.source != "device_code"
                ]
                self._entries.append(winner)
                if self._current_id in stale_ids:
                    self._current_id = None
        return True

    def select(self) -> Optional[PooledCredential]:
        failed_source_ids = self._validate_source_owned_codex_entries()
        self._sync_external_status_entries(probe_quota=True)
        entry, pending_refresh = self._select_under_lock(failed_source_ids)
        self._persist_pending_changes()
        if (
            entry is None
            and not pending_refresh
            and self._converge_empty_xai_source_selection()
        ):
            failed_source_ids = self._validate_source_owned_codex_entries()
            self._sync_external_status_entries(probe_quota=True)
            entry, pending_refresh = self._select_under_lock(failed_source_ids)
            self._persist_pending_changes()
        if pending_refresh:
            self._refresh_pending_entries(pending_refresh)
        if entry is not None:
            self._unmatched_rotation_streak = 0
            return entry
        # If no entry was available but we just refreshed some, re-select
        # now that the refreshed entries are back in the pool.
        if pending_refresh:
            failed_source_ids = self._validate_source_owned_codex_entries()
            self._sync_external_status_entries(probe_quota=True)
            entry, _ = self._select_under_lock(failed_source_ids)
            self._persist_pending_changes()
            if entry is not None:
                self._unmatched_rotation_streak = 0
        return entry

    def _select_under_lock(
        self,
        excluded_source_ids: Optional[Set[str]] = None,
    ) -> Tuple[Optional[PooledCredential], List[tuple]]:
        """Run selection under the lock, returning entry + pending refreshes."""
        with self._lock:
            return self._select_unlocked(
                excluded_source_ids=excluded_source_ids,
            )

    def _refresh_pending_entries(self, pending: List[tuple]) -> None:
        """Refresh deferred OAuth entries outside the lock.

        Each entry is refreshed under the cross-process ``_auth_store_lock``
        (which can block for 20+ seconds) and then merged into the pool.
        On failure the entry is silently skipped.
        """
        for entry, _sync_fn in pending:
            # _refresh_entry merges the refreshed entry into the pool
            # internally. Its mutation primitives (_replace_entry, _persist)
            # are self-locking, and the quarantine paths inside
            # _refresh_entry_impl take self._lock explicitly around their
            # read-modify-write of self._entries — required because this
            # call site runs OUTSIDE the pool lock.
            self._refresh_entry(entry, force=False)

    def _available_entries(
        self,
        *,
        clear_expired: bool = False,
        refresh: bool = False,
        excluded_source_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[PooledCredential], List[tuple]]:
        """Return (available, pending_refresh) for entries not in cooldown.

        When *clear_expired* is True, entries whose cooldown has elapsed are
        reset to STATUS_OK and persisted. When *refresh* is True, entries that
        need a token refresh are returned to the caller for refresh after the
        lock is released.
        """
        now = time.time()
        entries_to_prune: List[str] = []
        available: List[PooledCredential] = []
        # Refresh can perform network and auth-store I/O for every OAuth
        # provider, so collect candidates here and execute them after unlock.
        pending_refresh: List[tuple] = []  # (entry, sync_entry_fn)
        # DEAD entries never re-enter rotation, so if at most one non-DEAD entry
        # exists there is nothing to rotate to: an exhausted sole credential
        # should cool down briefly rather than bench the only key for an hour.
        sole_credential = sum(
            1 for e in self._entries if e.last_status != STATUS_DEAD
        ) <= 1
        for entry in list(self._entries):
            if excluded_source_ids and entry.id in excluded_source_ids:
                continue
            # Borrowed credentials persist as metadata-only references and are
            # hydrated from their live source on load.  A stale duplicate row
            # can remain unhydrated; never lease or select it as an empty key.
            if entry.auth_type == AUTH_TYPE_API_KEY and not entry.runtime_api_key:
                continue
            if entry.last_status == STATUS_DEAD:
                # Manual DEAD credentials get pruned after a 24h quiet window
                # so the pool doesn't accumulate dead entries forever.  The
                # user can always re-add via ``hermes auth add``.  Singleton-
                # seeded DEAD entries are kept so the audit trail (label,
                # last_error_reason, timestamps) stays visible — pruning them
                # would just be undone by ``_seed_from_singletons`` on the
                # next load anyway.
                if _is_manual_source(entry.source):
                    dead_at = entry.last_status_at or 0
                    if dead_at and now - dead_at > DEAD_MANUAL_PRUNE_TTL_SECONDS:
                        _label = entry.label or entry.id[:8]
                        logger.warning(
                            "credential pool: pruning DEAD manual entry %s "
                            "(reason=%s, age=%.1fh) — re-add via `hermes auth add %s`",
                            _label,
                            entry.last_error_reason or "unknown",
                            (now - dead_at) / 3600.0,
                            self.provider,
                        )
                        # Mark for removal after the loop completes; we can't
                        # mutate self._entries while iterating.
                        entries_to_prune.append(entry.id)
                # Permanently failed credentials never re-enter rotation via
                # TTL.  They only clear when a write-side re-auth sync rewrites
                # the tokens (e.g. ``_save_codex_tokens`` after a fresh
                # device-code login).  The auth.json-sync paths below handle
                # the re-auth case for OAuth singletons.
                continue
            if entry.last_status == STATUS_EXHAUSTED:
                exhausted_until = _exhausted_until(entry, sole_credential=sole_credential)
                if exhausted_until is not None and now < exhausted_until:
                    continue
                if clear_expired:
                    cleared = replace(
                        entry,
                        last_status=STATUS_OK,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                    self._replace_entry(entry, cleared)
                    entry = cleared
            if refresh and self._entry_needs_refresh(entry):
                pending_refresh.append((entry, None))
                continue
            available.append(entry)
        if entries_to_prune:
            pruned_ids = set(entries_to_prune)
            pruned_entries = [
                entry for entry in self._entries if entry.id in pruned_ids
            ]
            self._entries = [e for e in self._entries if e.id not in pruned_ids]
        else:
            pruned_entries = []
        if pruned_entries:
            for pruned_entry in pruned_entries:
                self._queue_removed_entry_unlocked(pruned_entry)
        return available, pending_refresh

    def _log_no_available_entries(self) -> None:
        """Emit the empty-pool INFO line at most once per throttle window.

        Called on every selection while the pool is empty/exhausted. Without
        throttling this storms the Windows cross-process log lock and stalls the
        event loop (see NO_AVAILABLE_ENTRIES_LOG_THROTTLE_SECONDS).
        """
        now = time.monotonic()
        last = self._last_no_entries_log_at
        if last is not None and (now - last) < NO_AVAILABLE_ENTRIES_LOG_THROTTLE_SECONDS:
            return
        self._last_no_entries_log_at = now
        logger.info("credential pool: no available entries (all exhausted or empty)")

    def _select_unlocked(
        self,
        *,
        refresh: bool = True,
        excluded_source_ids: Optional[Set[str]] = None,
    ) -> Tuple[Optional[PooledCredential], List[tuple]]:
        """Select the best available credential entry.

        Returns ``(entry, pending_refresh)`` where *pending_refresh* contains
        single-use-token entries that must be refreshed outside the lock.
        """
        available, pending_refresh = self._available_entries(
            clear_expired=True,
            refresh=refresh,
            excluded_source_ids=excluded_source_ids,
        )
        if pending_refresh:
            # Do not select, rotate, or increment counters from a partial view
            # that excludes refreshable entries. The caller refreshes outside
            # the lock and performs one selection against the current pool.
            return None, pending_refresh
        if not available:
            self._current_id = None
            self._log_no_available_entries()
            return None, pending_refresh

        # A successful selection means the pool recovered; re-arm the throttle
        # so a later re-exhaustion logs immediately rather than being silenced
        # by a window opened during the previous empty stretch.
        self._last_no_entries_log_at = None

        if self._strategy == STRATEGY_RANDOM:
            entry = random.choice(available)
            self._current_id = entry.id
            return entry, pending_refresh

        if self._strategy == STRATEGY_LEAST_USED and len(available) > 1:
            entry = min(available, key=lambda e: e.request_count)
            # Increment usage counter so subsequent selections distribute load
            updated = replace(entry, request_count=entry.request_count + 1)
            self._replace_entry(entry, updated)
            self._current_id = entry.id
            return updated, pending_refresh

        if self._strategy == STRATEGY_ROUND_ROBIN and len(available) > 1:
            entry = available[0]
            previous_by_id = {candidate.id: candidate for candidate in self._entries}
            previous_owners = {
                candidate.id: self._trusted_codex_source_owner(candidate)
                for candidate in self._entries
            }
            rotated = [candidate for candidate in self._entries if candidate.id != entry.id]
            rotated.append(replace(entry, priority=len(self._entries) - 1))
            self._entries = [replace(candidate, priority=idx) for idx, candidate in enumerate(rotated)]
            for previous in previous_by_id.values():
                self._trusted_codex_source_owners.pop(id(previous), None)
            for candidate in self._entries:
                owner = previous_owners.get(candidate.id)
                previous = previous_by_id.get(candidate.id)
                if (
                    owner is not None
                    and previous is not None
                    and self._entry_tokens_match(candidate, previous)
                    and candidate.source_store_path is not None
                    and self._same_store_path(
                        candidate.source_store_path,
                        owner.source_path,
                    )
                ):
                    self._trusted_codex_source_owners[id(candidate)] = owner
            for candidate in self._entries:
                if previous_by_id.get(candidate.id) != candidate:
                    self._record_entry_mutation_unlocked(
                        candidate.id,
                        dirty=True,
                    )
            self._current_id = entry.id
            return self._current_unlocked() or entry, pending_refresh

        entry = available[0]
        self._current_id = entry.id
        return entry, pending_refresh

    def peek(self) -> Optional[PooledCredential]:
        # Single lock acquisition for the whole read; call the unlocked
        # helpers so we don't re-enter the non-reentrant ``self._lock``.
        failed_source_ids = self._validate_source_owned_codex_entries()
        self._sync_external_status_entries()
        with self._lock:
            available, _pending = self._available_entries(
                excluded_source_ids=failed_source_ids,
            )
            current = self._current_unlocked()
            if current is not None:
                for entry in available:
                    if entry.id == current.id:
                        result = entry
                        break
                else:
                    result = available[0] if available else None
            else:
                result = available[0] if available else None
        self._persist_pending_changes()
        return result

    def mark_exhausted_and_rotate(
        self,
        *,
        status_code: Optional[int],
        error_context: Optional[Dict[str, Any]] = None,
        api_key_hint: Optional[str] = None,
        credential_id: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> Optional[PooledCredential]:
        failed_source_ids = self._validate_source_owned_codex_entries()
        self._sync_external_status_entries(probe_quota=True)
        try:
            return self._mark_exhausted_and_rotate_locked(
                status_code=status_code,
                error_context=error_context,
                api_key_hint=api_key_hint,
                credential_id=credential_id,
                failure_reason=failure_reason,
                failed_source_ids=failed_source_ids,
            )
        finally:
            self._persist_pending_changes()

    def _mark_exhausted_and_rotate_locked(
        self,
        *,
        status_code: Optional[int],
        error_context: Optional[Dict[str, Any]],
        api_key_hint: Optional[str],
        credential_id: Optional[str],
        failure_reason: Optional[str],
        failed_source_ids: Set[str],
    ) -> Optional[PooledCredential]:
        with self._lock:
            if credential_id and credential_id in failed_source_ids:
                self._unmatched_rotation_streak = 0
                if self._current_id == credential_id:
                    self._current_id = None
                return None
            entry = None
            identity_supplied = bool(credential_id or api_key_hint)
            if credential_id and credential_id not in failed_source_ids:
                entry = next(
                    (e for e in self._entries if e.id == credential_id),
                    None,
                )
            if entry is None and api_key_hint:
                # Prefer the specific entry whose API key matches the one that
                # actually failed.  When this pool was freshly loaded from disk
                # (another process already rotated), current() is None and
                # _select_unlocked() would return the NEXT key — the wrong one.
                entry = next(
                    (
                        e
                        for e in self._entries
                        if e.id not in failed_source_ids
                        and e.runtime_api_key == api_key_hint
                    ),
                    None,
                )
            if entry is None and identity_supplied:
                # The failed credential is identifiable but matches no entry
                # (rotated away, or a wrapper whose runtime key differs).
                # Falling through to current()/_select_unlocked() would mark an
                # innocent healthy key exhausted for the full cooldown TTL.
                #
                # #70401: this branch must still be BOUNDED. With OAuth-token
                # auth the upstream 401's key hint never matches any entry's
                # ``runtime_api_key``, so every retry lands here, nothing is
                # ever marked exhausted, and the pool can never reach the
                # "no available entries" state — the caller retries the same
                # dead token forever (~6/sec, starving the event loop so chat
                # interrupts are never processed). The single-entry case
                # below already escapes; multi-entry pools could still
                # ping-pong A→B→A indefinitely without marking anything.
                # Cap consecutive no-mark rotations at one full lap of the
                # available entries: past that, every candidate has been
                # handed back at least once without recovery, so stop
                # guessing and surface the error (no cooldown is written for
                # anybody — healthy keys stay available for the next turn).
                self._unmatched_rotation_streak += 1
                available_count, _ = self._available_entries(
                    excluded_source_ids=failed_source_ids,
                )
                available_count = len(available_count)
                if self._unmatched_rotation_streak > max(available_count, 1):
                    logger.warning(
                        "credential pool: failed credential identity matched no "
                        "%s entry for %d consecutive rotations (pool size %d) — "
                        "surfacing the error instead of rotating again",
                        self.provider,
                        self._unmatched_rotation_streak,
                        available_count,
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
                next_entry, _pending = self._select_unlocked(
                    refresh=False,
                    excluded_source_ids=failed_source_ids,
                )
                avail, _ = self._available_entries(
                    excluded_source_ids=failed_source_ids,
                )
                if (
                    next_entry is not None
                    and len(avail) == 1
                    and not (
                        credential_id in failed_source_ids
                        and next_entry.id != credential_id
                    )
                ):
                    # A single-entry pool cannot rotate. Returning its only
                    # entry reports a successful recovery without changing
                    # the credential, so the caller retries the same 401
                    # indefinitely. Let fallback/error propagation proceed.
                    self._unmatched_rotation_streak = 0
                    self._current_id = None
                    return None
                return next_entry
            # A real entry was identified — any prior unmatched-rotation
            # streak is stale (this mark WILL advance pool state).
            self._unmatched_rotation_streak = 0
            if entry is None:
                current = self._current_unlocked()
                if current is not None and current.id in failed_source_ids:
                    current = None
                entry = current or self._select_unlocked(
                    refresh=False,
                    excluded_source_ids=failed_source_ids,
                )[0]
            if entry is None:
                return None
            _label = entry.label or entry.id[:8]
            self._mark_exhausted(
                entry,
                status_code,
                error_context,
                persist=False,
                failure_reason=failure_reason,
                source_validated=True,
            )
            # A 402/429/401 is an API-key–level failure: the account is out of
            # balance, rate-limited, or its key is rejected.  The same key can
            # back more than one pool entry (e.g. an explicit pool entry plus a
            # ``model_config`` entry auto-seeded from ``model.api_key`` — both
            # carry the identical ``runtime_api_key``).  Marking only the first
            # match leaves the sibling entries OK, so ``_select_unlocked()``
            # keeps handing back the same depleted key and rotation never
            # converges — the caller ``continue``s forever until the client
            # disconnects (a ~2.5min hang with no error surfaced to the user).
            # Mark every entry sharing the failed key so the pool can reach the
            # "no available entries" state and let the error propagate.
            failed_runtime_key = getattr(entry, "runtime_api_key", None)
            if identity_supplied and failed_runtime_key:
                for sibling in self._entries:
                    if sibling.id == entry.id:
                        continue
                    if sibling.runtime_api_key == failed_runtime_key:
                        self._mark_exhausted(
                            sibling,
                            status_code,
                            error_context,
                            persist=False,
                            failure_reason=failure_reason,
                            source_validated=True,
                        )
            # Re-read the updated entry to log the correct terminal state.
            updated_entry = next(
                (e for e in self._entries if e.id == entry.id), entry,
            )
            if updated_entry.last_status == STATUS_DEAD:
                logger.warning(
                    "credential pool: marking %s DEAD (status=%s, reason=%s) — "
                    "permanently failed, will NOT re-enter rotation until re-auth",
                    _label, status_code, updated_entry.last_error_reason or "unknown",
                )
            else:
                logger.info(
                    "credential pool: marking %s exhausted (status=%s), rotating",
                    _label, status_code,
                )
            self._current_id = None
            next_entry, _pending = self._select_unlocked(
                refresh=False,
                excluded_source_ids=failed_source_ids,
            )
            if next_entry:
                _next_label = next_entry.label or next_entry.id[:8]
                logger.info("credential pool: rotated to %s", _next_label)
            return next_entry

    def acquire_lease(self, credential_id: Optional[str] = None) -> Optional[str]:
        """Acquire a soft lease on a credential.

        If a specific credential_id is provided, lease that entry directly.
        Otherwise prefer the least-leased available credential, using priority as
        a stable tie-breaker. When every credential is already at the soft cap,
        still return the least-leased one instead of blocking.
        """
        entry_ids = {credential_id} if credential_id is not None else None
        failed_source_ids = self._validate_source_owned_codex_entries(entry_ids)
        self._sync_external_status_entries(probe_quota=True)
        if failed_source_ids:
            chosen_id, pending_refresh = self._acquire_lease_under_lock(
                credential_id,
                excluded_source_ids=failed_source_ids,
            )
        else:
            chosen_id, pending_refresh = self._acquire_lease_under_lock(
                credential_id,
            )
        self._persist_pending_changes()
        if pending_refresh:
            self._refresh_pending_entries(pending_refresh)
            # The first pass never leases from a partial view when refreshes
            # are pending. Re-read and choose once from the current pool.
            failed_source_ids = self._validate_source_owned_codex_entries(
                entry_ids,
            )
            self._sync_external_status_entries(probe_quota=True)
            if failed_source_ids:
                chosen_id, _ = self._acquire_lease_under_lock(
                    credential_id,
                    excluded_source_ids=failed_source_ids,
                )
            else:
                chosen_id, _ = self._acquire_lease_under_lock(
                    credential_id,
                )
            self._persist_pending_changes()
        return chosen_id

    def _acquire_lease_under_lock(
        self,
        credential_id: Optional[str],
        *,
        excluded_source_ids: Optional[Set[str]] = None,
    ) -> Tuple[Optional[str], List[tuple]]:
        """Run lease acquisition under the lock, returning id + pending refreshes."""
        with self._lock:
            if credential_id:
                if excluded_source_ids and credential_id in excluded_source_ids:
                    return None, []
                candidate = next(
                    (
                        entry
                        for entry in self._entries
                        if entry.id == credential_id
                    ),
                    None,
                )
                if candidate is None:
                    return None, []
                self._active_leases[credential_id] = self._active_leases.get(credential_id, 0) + 1
                self._current_id = credential_id
                return credential_id, []

            if excluded_source_ids:
                available, pending_refresh = self._available_entries(
                    clear_expired=True,
                    refresh=True,
                    excluded_source_ids=excluded_source_ids,
                )
            else:
                available, pending_refresh = self._available_entries(
                    clear_expired=True,
                    refresh=True,
                )
            if pending_refresh:
                return None, pending_refresh
            if not available:
                return None, pending_refresh

            below_cap = [
                entry for entry in available
                if self._active_leases.get(entry.id, 0) < self._max_concurrent
            ]
            candidates = below_cap if below_cap else available
            chosen = min(
                candidates,
                key=lambda entry: (self._active_leases.get(entry.id, 0), entry.priority),
            )
            self._active_leases[chosen.id] = self._active_leases.get(chosen.id, 0) + 1
            self._current_id = chosen.id
            return chosen.id, pending_refresh

    def release_lease(self, credential_id: str) -> None:
        """Release a previously acquired credential lease."""
        with self._lock:
            count = self._active_leases.get(credential_id, 0)
            if count <= 1:
                self._active_leases.pop(credential_id, None)
            else:
                self._active_leases[credential_id] = count - 1

    def try_refresh_current(self) -> Optional[PooledCredential]:
        with self._lock:
            entry = self._current_unlocked()
        if entry is None:
            return None
        refreshed = self._refresh_entry(entry, force=True)
        if refreshed is not None:
            with self._lock:
                self._current_id = refreshed.id
        return refreshed

    def try_refresh_matching(
        self,
        api_key_hint: Optional[str] = None,
        credential_id: Optional[str] = None,
    ) -> Optional[PooledCredential]:
        """Force-refresh the entry that supplied the failed request.

        Direct provider integrations may reload the pool after a request has
        already failed, so they cannot rely on ``current_id`` identifying the
        issuing credential. With no hint, select an entry without first doing
        the normal proactive refresh; the forced refresh below must consume a
        rotating refresh token exactly once.
        """
        failed_source_ids = self._validate_source_owned_codex_entries()
        self._sync_external_status_entries(probe_quota=True)
        try:
            with self._lock:
                if credential_id and credential_id in failed_source_ids:
                    if self._current_id == credential_id:
                        self._current_id = None
                    return None
                entry = None
                if credential_id and credential_id not in failed_source_ids:
                    entry = next(
                        (
                            candidate
                            for candidate in self._entries
                            if candidate.id == credential_id
                        ),
                        None,
                    )
                if entry is None:
                    if api_key_hint:
                        entry = next(
                            (
                                candidate
                                for candidate in self._entries
                                if candidate.id not in failed_source_ids
                                and candidate.runtime_api_key == api_key_hint
                            ),
                            None,
                        )
                    else:
                        current = self._current_unlocked()
                        if current is not None and current.id in failed_source_ids:
                            current = None
                        entry = current or self._select_unlocked(
                            refresh=False,
                            excluded_source_ids=failed_source_ids,
                        )[0]
                if entry is None:
                    return None
                self._current_id = entry.id
        finally:
            self._persist_pending_changes()
        refreshed = self._refresh_entry(entry, force=True)
        if refreshed is not None:
            with self._lock:
                self._current_id = refreshed.id
        return refreshed

    def reset_statuses(self) -> int:
        failed_source_ids = self._validate_source_owned_codex_entries()
        with self._lock:
            count = 0
            reset_ids: Set[str] = set()
            for entry in list(self._entries):
                if entry.id in failed_source_ids:
                    continue
                if entry.last_status or entry.last_status_at or entry.last_error_code:
                    updated = replace(
                        entry,
                        last_status=None,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                    self._replace_entry(entry, updated)
                    reset_ids.add(entry.id)
                    count += 1
            if count:
                self._source_status_reset_ids.update(
                    entry.id
                    for entry in self._entries
                    if entry.id in reset_ids
                    and self._is_trusted_codex_source_owned(entry)
                )
        self._persist_pending_changes()
        return count

    def remove_index(self, index: int) -> Optional[PooledCredential]:
        self._validate_source_owned_codex_entries()
        with self._external_state_lock:
            with self._lock:
                if index < 1 or index > len(self._entries):
                    return None
                removed = self._entries.pop(index - 1)
                for new_priority, entry in enumerate(list(self._entries)):
                    if entry.priority != new_priority:
                        self._replace_entry(
                            entry,
                            replace(entry, priority=new_priority),
                        )
                self._queue_removed_entry_unlocked(removed)
                if self._current_id == removed.id:
                    self._current_id = None
        self._persist_pending_changes()
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
        self._validate_source_owned_codex_entries()
        with self._external_state_lock:
            with self._lock:
                entry = replace(entry, priority=_next_priority(self._entries))
                self._entries.append(entry)
                self._record_entry_mutation_unlocked(entry.id, dirty=True)
        self._persist_pending_changes()
        return entry


def _upsert_entry(entries: List[PooledCredential], provider: str, source: str, payload: Dict[str, Any]) -> bool:
    matching_indices = []
    for idx, entry in enumerate(entries):
        if entry.source == source:
            matching_indices.append(idx)

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
    field_updates = {}
    extra_updates = {}
    _field_names = {f.name for f in fields(existing)}
    payload_fingerprint = sanitize_borrowed_credential_payload(
        {**payload, "source": source},
        provider,
    ).get("secret_fingerprint")
    same_redacted_lineage = bool(
        provider == "anthropic"
        and source == "claude_code"
        and payload_fingerprint
        and payload_fingerprint == existing.extra.get("secret_fingerprint")
    )
    token_changed = (
        "access_token" in payload
        and payload["access_token"] is not None
        and payload["access_token"] != existing.access_token
        and not same_redacted_lineage
    )
    for key, value in payload.items():
        if key in {"id", "priority"} or value is None:
            continue
        if key == "label" and existing.label:
            continue
        if key in _field_names:
            if getattr(existing, key) != value:
                field_updates[key] = value
        elif key in _EXTRA_KEYS:
            if existing.extra.get(key) != value:
                extra_updates[key] = value
    # When the credential token itself changes (key rotation), clear any
    # exhaustion/error state — the old status is stale for the new key. Claude
    # Code secrets are redacted on disk, so their persisted fingerprint, not the
    # empty runtime value loaded from that row, determines whether its lineage changed.
    if token_changed and existing.last_status is not None:
        field_updates["last_status"] = None
        field_updates["last_status_at"] = None
        field_updates["last_error_code"] = None
        field_updates["last_error_reason"] = None
        field_updates["last_error_message"] = None
        field_updates["last_error_reset_at"] = None
    if field_updates or extra_updates:
        if extra_updates:
            field_updates["extra"] = {**existing.extra, **extra_updates}
        updated = replace(existing, **field_updates)
        entries[existing_idx] = updated
        # Runtime-only borrowed secret updates should refresh the in-memory
        # entry without forcing auth.json churn when the disk-safe payload is
        # unchanged (for example env keys with the same fingerprint).
        return bool(duplicate_indices) or existing.to_dict() != updated.to_dict()
    return bool(duplicate_indices)


def _normalize_pool_priorities(provider: str, entries: List[PooledCredential]) -> bool:
    if provider != "anthropic":
        return False

    source_rank = {
        "env:ANTHROPIC_TOKEN": 0,
        "env:CLAUDE_CODE_OAUTH_TOKEN": 1,
        "hermes_pkce": 2,
        "claude_code": 3,
        "env:ANTHROPIC_API_KEY": 4,
    }
    manual_entries = sorted(
        (entry for entry in entries if _is_manual_source(entry.source)),
        key=lambda entry: entry.priority,
    )
    seeded_entries = sorted(
        (entry for entry in entries if not _is_manual_source(entry.source)),
        key=lambda entry: (
            source_rank.get(entry.source, len(source_rank)),
            entry.priority,
            entry.label,
        ),
    )

    ordered = [*manual_entries, *seeded_entries]
    id_to_idx = {entry.id: idx for idx, entry in enumerate(entries)}
    changed = False
    for new_priority, entry in enumerate(ordered):
        if entry.priority != new_priority:
            entries[id_to_idx[entry.id]] = replace(entry, priority=new_priority)
            changed = True
    return changed


def _codex_source_pool_alias(
    source_path: Path,
    state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the owning Codex ``device_code`` alias for a singleton state."""
    source_store = auth_mod._load_auth_store(source_path)
    pool = source_store.get("credential_pool")
    persisted = pool.get("openai-codex") if isinstance(pool, dict) else None
    if not isinstance(persisted, list):
        return None

    aliases = [
        item
        for item in persisted
        if isinstance(item, dict) and item.get("source") == "device_code"
    ]
    if not aliases:
        return None

    tokens = state.get("tokens")
    if isinstance(tokens, dict):
        access_token = str(tokens.get("access_token") or "")
        refresh_token = str(tokens.get("refresh_token") or "")
        for alias in aliases:
            if (
                str(alias.get("access_token") or "") == access_token
                and str(alias.get("refresh_token") or "") == refresh_token
            ):
                return dict(alias)
    return dict(aliases[0])


def _loaded_codex_source_owner_kind(entry: PooledCredential) -> str:
    """Classify trusted fallback ownership from the hydrated row snapshot.

    Codex singleton aliases use the canonical ``device_code`` source. Explicit
    independent accounts are written as ``manual:device_code``. Binding owner
    kind to that trusted row category avoids a later unlocked provider reread;
    an uncertain canonical row therefore fails closed as singleton-owned.
    """
    return "singleton" if entry.source == "device_code" else "pool"


def _seed_from_singletons(
    provider: str,
    entries: List[PooledCredential],
) -> Tuple[bool, Set[str]]:
    changed = False
    active_sources: Set[str] = set()
    auth_store = _load_auth_store()
    active_pool = auth_store.get("credential_pool")
    local_entries = (
        active_pool.get(provider) if isinstance(active_pool, dict) else None
    )
    local_entry_ids = {
        str(entry.get("id"))
        for entry in local_entries or []
        if isinstance(entry, dict) and entry.get("id")
    }

    # Shared suppression gate — used at every upsert site so
    # `hermes auth remove <provider> <N>` is stable across all source types.
    try:
        from hermes_cli.auth import is_source_suppressed as _is_suppressed
    except ImportError:
        def _is_suppressed(_p, _s):  # type: ignore[misc]
            return False

    if provider == "anthropic":
        # Only auto-discover external credentials (Claude Code, Hermes PKCE)
        # when the user has explicitly configured anthropic as their provider.
        # Without this gate, auxiliary client fallback chains silently read
        # ~/.claude/.credentials.json without user consent.  See PR #4210.
        try:
            from hermes_cli.auth import is_provider_explicitly_configured
            if not is_provider_explicitly_configured("anthropic"):
                return changed, active_sources
        except ImportError:
            pass

        # API-key vs OAuth is a user-visible choice at `hermes setup` ("Claude
        # Pro/Max subscription" vs "Anthropic API key").  The signal that the
        # user picked the API-key path is: ANTHROPIC_API_KEY set in the env,
        # AND no OAuth env vars set — `save_anthropic_api_key()` writes the
        # API key and zeros ANTHROPIC_TOKEN; `save_anthropic_oauth_token()`
        # does the inverse.  When that signal is present we MUST NOT seed
        # autodiscovered OAuth tokens (~/.claude/.credentials.json from the
        # Claude Code CLI, hermes_pkce creds from a previous OAuth login)
        # into the anthropic pool — otherwise rotation on a 401/429 silently
        # flips the session onto an OAuth credential, which forces the Claude
        # Code identity injection, `mcp_` tool-name rewrite, and claude-cli
        # User-Agent header (`agent/anthropic_adapter.py:2128`).  Users who
        # explicitly opted into the API-key path are explicitly opting OUT of
        # that masquerade.  Prefer ~/.hermes/.env over os.environ for the
        # same reason `_seed_from_env` does — that's the authoritative file
        # that `hermes setup` writes.
        _env_file = load_env()

        def _env_val(key: str) -> str:
            return (_env_file.get(key) or _get_secret(key, "") or "").strip()

        anthropic_api_key = _env_val("ANTHROPIC_API_KEY")
        anthropic_oauth_env = (
            _env_val("ANTHROPIC_TOKEN") or _env_val("CLAUDE_CODE_OAUTH_TOKEN")
        )
        api_key_path_explicit = bool(anthropic_api_key and not anthropic_oauth_env)

        if api_key_path_explicit:
            # Prune any stale autodiscovered OAuth entries that may have been
            # seeded into the on-disk pool during a previous OAuth session.
            # Without this, switching OAuth -> API key at setup leaves the
            # OAuth entries dormant in auth.json forever and rotation on a
            # transient 401 could revive them.
            retained = [
                entry for entry in entries
                if entry.source not in {"hermes_pkce", "claude_code"}
            ]
            if len(retained) != len(entries):
                entries[:] = retained
                changed = True
            return changed, active_sources

        from agent.anthropic_adapter import read_claude_code_credentials, read_hermes_oauth_credentials

        hermes_creds = read_hermes_oauth_credentials()
        claude_creds = read_claude_code_credentials()
        if not claude_creds or not claude_creds.get("accessToken"):
            retained = [entry for entry in entries if entry.source != "claude_code"]
            if len(retained) != len(entries):
                entries[:] = retained
                changed = True

        for source_name, creds in (
            ("hermes_pkce", hermes_creds),
            ("claude_code", claude_creds),
        ):
            if creds and creds.get("accessToken"):
                if _is_suppressed(provider, source_name):
                    continue
                active_sources.add(source_name)
                changed |= _upsert_entry(
                    entries,
                    provider,
                    source_name,
                    {
                        "source": source_name,
                        "auth_type": AUTH_TYPE_OAUTH,
                        "access_token": creds.get("accessToken", ""),
                        "refresh_token": creds.get("refreshToken"),
                        "expires_at_ms": creds.get("expiresAt"),
                        "label": label_from_token(creds.get("accessToken", ""), source_name),
                        **(
                            {"credential_source": creds.get("source")}
                            if source_name == "claude_code"
                            else {}
                        ),
                    },
                )

    elif provider == "nous":
        state = _load_provider_state(auth_store, "nous")
        has_runtime_material = bool(
            isinstance(state, dict)
            and (
                str(state.get("access_token") or "").strip()
                or str(state.get("agent_key") or "").strip()
            )
        )
        if state and not has_runtime_material:
            retained = [
                entry for entry in entries
                if entry.source not in {"device_code", "manual:device_code"}
            ]
            if len(retained) != len(entries):
                entries[:] = retained
                changed = True
        if state and has_runtime_material and not _is_suppressed(provider, "device_code"):
            active_sources.add("device_code")
            # Prefer a user-supplied label embedded in the singleton state
            # (set by persist_nous_credentials(label=...) when the user ran
            # `hermes auth add nous --label <name>`).  Fall back to the
            # auto-derived token fingerprint for logins that didn't supply one.
            custom_label = str(state.get("label") or "").strip()
            seeded_label = custom_label or label_from_token(
                state.get("access_token", ""), "device_code"
            )
            changed |= _upsert_entry(
                entries,
                provider,
                "device_code",
                {
                    "source": "device_code",
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
                    # Carry the refresh timestamps into the pool so
                    # freshness-sensitive consumers (self-heal hooks, pool
                    # pruning by age) can distinguish just-refreshed credentials
                    # from stale ones.  Without these, fresh device_code
                    # entries get obtained_at=None and look older than they
                    # are (#15099).
                    "obtained_at": state.get("obtained_at"),
                    "expires_in": state.get("expires_in"),
                    "agent_key_id": state.get("agent_key_id"),
                    "agent_key_expires_in": state.get("agent_key_expires_in"),
                    "agent_key_reused": state.get("agent_key_reused"),
                    "agent_key_obtained_at": state.get("agent_key_obtained_at"),
                    "tls": state.get("tls") if isinstance(state.get("tls"), dict) else None,
                    "label": seeded_label,
                },
            )

    elif provider == "copilot":
        # Copilot tokens are resolved dynamically via `gh auth token` or
        # env vars (COPILOT_GITHUB_TOKEN / GH_TOKEN).  They don't live in
        # the auth store or credential pool, so we resolve them here.
        try:
            from hermes_cli.copilot_auth import (
                COPILOT_ENV_VARS,
                resolve_copilot_token,
                get_copilot_api_token,
            )
            # All-sources suppression gate BEFORE any work — including the
            # `gh auth token` subprocess spawn.  resolve_copilot_token()
            # shells out (~30ms), and the exchange retries 3x with backoff
            # (~35s worst case); a user who suppressed every copilot source
            # (hermes auth remove copilot gh_cli) must not pay either on
            # every pool load (model picker open, /model, agent startup).
            # Enumerating the full source space here matches what
            # credential_sources._remove_copilot_gh suppresses, so an
            # all-suppressed check is stable.
            copilot_sources = ["gh_cli"] + [f"env:{v}" for v in COPILOT_ENV_VARS]
            if all(_is_suppressed(provider, s) for s in copilot_sources):
                return changed, active_sources
            token, source = resolve_copilot_token()
            if token:
                # ``resolve_copilot_token`` returns exactly "gh auth token"
                # for the CLI path; env-sourced tokens return the var name.
                # Match exactly — a substring test classifies GH_TOKEN and
                # GITHUB_TOKEN as gh_cli, silently bypassing a user's
                # per-env-var suppression.
                source_name = "gh_cli" if source == "gh auth token" else f"env:{source}"
                # Per-source suppression gate (a user may suppress only the
                # gh CLI path and keep an env var, or vice versa) BEFORE the
                # network exchange.  The exchange retries 3x with 10s
                # timeouts and 4.5s total backoff (~35s worst case), so a
                # source the user already suppressed
                # must not burn that dead time just to have the entry
                # discarded afterwards.  Same early-gate pattern every other
                # singleton branch uses.
                if _is_suppressed(provider, source_name):
                    return changed, active_sources
                api_token, enterprise_base_url = get_copilot_api_token(token)
                # Observability: get_copilot_api_token falls back to returning
                # the RAW token when the exchange fails. A raw ~40-char token
                # sent to the Copilot API is routed to the fallback
                # "copilot-language-server" integrator, whose allowlist omits
                # enterprise-only models (claude-opus-4.8) → HTTP 400 on every
                # turn. exchange_copilot_token now retries + reuses a persisted
                # JWT, so this should be rare; surface it at WARNING so a
                # recurrence is visible in logs instead of failing silently.
                if api_token == token and not enterprise_base_url:
                    logger.warning(
                        "Copilot token exchange degraded to RAW token (exchange "
                        "unavailable); enterprise-only models may 400 with "
                        "model_not_available_for_integrator until exchange recovers."
                    )
                active_sources.add(source_name)
                pconfig = PROVIDER_REGISTRY.get(provider)
                # Use enterprise base URL from token exchange if available,
                # otherwise fall back to the provider's default.
                effective_base_url = enterprise_base_url or (
                    pconfig.inference_base_url if pconfig else ""
                )
                changed |= _upsert_entry(
                    entries,
                    provider,
                    source_name,
                    {
                        "source": source_name,
                        "auth_type": AUTH_TYPE_API_KEY,
                        "access_token": api_token,
                        "base_url": effective_base_url,
                        "label": source,
                    },
                )
        except Exception as exc:
            logger.debug("Copilot token seed failed: %s", exc)

    elif provider == "qwen-oauth":
        # Qwen OAuth tokens live in ~/.qwen/oauth_creds.json, written by
        # the Qwen CLI (`qwen auth qwen-oauth`).  They aren't in the
        # Hermes auth store or env vars, so resolve them here.
        # Use refresh_if_expiring=False to avoid network calls during
        # pool loading / provider discovery.
        try:
            from hermes_cli.auth import resolve_qwen_runtime_credentials
            creds = resolve_qwen_runtime_credentials(refresh_if_expiring=False)
            token = creds.get("api_key", "")
            if token:
                source_name = creds.get("source", "qwen-cli")
                if not _is_suppressed(provider, source_name):
                    active_sources.add(source_name)
                    changed |= _upsert_entry(
                        entries,
                        provider,
                        source_name,
                        {
                            "source": source_name,
                            "auth_type": AUTH_TYPE_OAUTH,
                            "access_token": token,
                            "expires_at_ms": creds.get("expires_at_ms"),
                            "base_url": creds.get("base_url", ""),
                            "label": creds.get("auth_file", source_name),
                        },
                    )
        except Exception as exc:
            logger.debug("Qwen OAuth token seed failed: %s", exc)

    elif provider == "minimax-oauth":
        # MiniMax OAuth tokens live in ~/.hermes/auth.json providers.minimax-oauth.
        # Seed the pool so `/auth list` reflects the logged-in state and the
        # standard `hermes auth remove minimax-oauth <N>` flow works.
        # Use refresh_if_expiring=False equivalent: resolve_minimax_oauth_runtime_credentials
        # always refreshes on expiry, so instead read raw state here to avoid
        # surprise network calls during provider discovery.
        try:
            from hermes_cli.auth import get_provider_auth_state
            state = get_provider_auth_state("minimax-oauth")
            if state and state.get("access_token"):
                source_name = "oauth"
                if not _is_suppressed(provider, source_name):
                    active_sources.add(source_name)
                    expires_at_ms = None
                    try:
                        from datetime import datetime as _dt
                        raw = state.get("expires_at", "")
                        if raw:
                            expires_at_ms = int(_dt.fromisoformat(raw).timestamp() * 1000)
                    except Exception:
                        expires_at_ms = None
                    base_url = str(state.get("inference_base_url", "") or "").rstrip("/")
                    changed |= _upsert_entry(
                        entries,
                        provider,
                        source_name,
                        {
                            "source": source_name,
                            "auth_type": AUTH_TYPE_OAUTH,
                            "access_token": state["access_token"],
                            "refresh_token": state.get("refresh_token"),
                            "expires_at_ms": expires_at_ms,
                            "base_url": base_url,
                            "label": state.get("label", "") or label_from_token(
                                state.get("access_token", ""), source_name
                            ),
                        },
                    )
        except Exception as exc:
            logger.debug("MiniMax OAuth token seed failed: %s", exc)

    elif provider == "openai-codex":
        # Respect user suppression — `hermes auth remove openai-codex` marks
        # the device_code source as suppressed so it won't be re-seeded from
        # the Hermes auth store.  Without this gate the removal is instantly
        # undone on the next load_pool() call.
        if _is_suppressed(provider, "device_code"):
            return changed, active_sources

        state, source_path = auth_mod._load_provider_state_with_source(
            auth_store,
            "openai-codex",
        )
        tokens = state.get("tokens") if isinstance(state, dict) else None
        # Hermes owns its own Codex auth state — we do NOT auto-import from
        # ~/.codex/auth.json at pool-load time.  OAuth refresh tokens are
        # single-use, so sharing them with Codex CLI / VS Code causes
        # refresh_token_reused race failures.  Users who want to adopt
        # existing Codex CLI credentials get a one-time, explicit prompt
        # via `hermes auth openai-codex`.
        if (
            isinstance(state, dict)
            and isinstance(tokens, dict)
            and tokens.get("access_token")
        ):
            active_sources.add("device_code")
            custom_label = str(state.get("label") or "").strip()
            payload = {
                "source": "device_code",
                "auth_type": AUTH_TYPE_OAUTH,
                "access_token": tokens.get("access_token", ""),
                "refresh_token": tokens.get("refresh_token"),
                "base_url": "https://chatgpt.com/backend-api/codex",
                "last_refresh": state.get("last_refresh"),
                "label": custom_label
                or label_from_token(tokens.get("access_token", ""), "device_code"),
            }
            active_path = auth_mod._auth_file_path()
            borrowed = bool(
                source_path is not None
                and not auth_mod._same_path(source_path, active_path)
            )
            if borrowed and source_path is not None:
                alias = _codex_source_pool_alias(source_path, state)
                if alias is not None:
                    token_changed = (
                        alias.get("access_token") != payload["access_token"]
                        or alias.get("refresh_token") != payload["refresh_token"]
                    )
                    alias_payload = dict(alias)
                    alias_payload.update(
                        {
                            key: value
                            for key, value in payload.items()
                            if key != "label"
                        }
                    )
                    if not alias_payload.get("label"):
                        alias_payload["label"] = payload["label"]
                    if token_changed:
                        for status_field in (
                            "last_status",
                            "last_status_at",
                            "last_error_code",
                            "last_error_reason",
                            "last_error_message",
                            "last_error_reset_at",
                        ):
                            alias_payload[status_field] = None
                    payload = alias_payload

                local_shadow_present = any(
                    entry.source == "device_code" and entry.id in local_entry_ids
                    for entry in entries
                )
                entries[:] = [
                    entry for entry in entries if entry.source != "device_code"
                ]
                payload.setdefault("id", uuid.uuid4().hex[:6])
                payload.setdefault("priority", _next_priority(entries))
                borrowed_entry = PooledCredential.from_dict(provider, payload)
                entries.append(
                    replace(borrowed_entry, source_store_path=source_path)
                )
                # Adding a runtime-only borrowed alias is not a profile write.
                # Only pre-existing copied profile rows require cleanup.
                changed |= local_shadow_present
            else:
                changed |= _upsert_entry(
                    entries,
                    provider,
                    "device_code",
                    payload,
                )

    elif provider == "xai-oauth":
        # When the user logs in via ``hermes model`` -> xAI Grok OAuth,
        # tokens are written to the auth.json singleton
        # (``providers["xai-oauth"]``).  Surface them in the pool too so
        # ``hermes auth list`` reflects the logged-in state and so the pool
        # is the single source of truth for refresh during runtime resolution.
        state = _load_provider_state(auth_store, "xai-oauth")
        tokens = state.get("tokens") if isinstance(state, dict) else None
        if isinstance(tokens, dict) and tokens.get("access_token"):
            # Device code is the only supported xAI OAuth flow; the singleton is
            # always surfaced as ``device_code`` (consistent with nous/codex).
            source = "device_code"
            if _is_suppressed(provider, source):
                return changed, active_sources
            active_sources.add(source)
            from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL

            base_url = DEFAULT_XAI_OAUTH_BASE_URL
            changed |= _upsert_entry(
                entries,
                provider,
                source,
                {
                    "source": source,
                    "auth_type": AUTH_TYPE_OAUTH,
                    "access_token": tokens.get("access_token", ""),
                    "refresh_token": tokens.get("refresh_token"),
                    "base_url": base_url,
                    "last_refresh": state.get("last_refresh"),
                    "label": label_from_token(tokens.get("access_token", ""), source),
                },
            )

    return changed, active_sources


# Prefer ~/.hermes/.env over os.environ — the user's config file is the
# authoritative source for Hermes credentials. Stale env vars from parent
# processes (Codex CLI, test scripts, etc.) should not override deliberate
# changes to the .env file. load_env() memoizes on the .env mtime, so
# per-call reads (pool seeding, per-turn credential refresh) cost a stat()
# when the file is unchanged.
def get_env_prefer_dotenv(key: str) -> str:
    env_file = load_env()
    raw = env_file.get(key, "").strip()
    scoped_value = (_get_secret(key, "") or "").strip()
    # If .env contains an unresolved op:// reference, prefer the
    # already-resolved value supplied by the active secret scope (or by
    # os.environ in legacy single-profile mode), set by
    # load_hermes_dotenv() -> apply_onepassword_secrets()).  The raw
    # "op://Vault/Item/field" string would otherwise win and every
    # provider auth attempt would receive a URL instead of a key.  This
    # happens during a partial migration, or when the user wrote op://
    # references straight into .env rather than the secrets.onepassword
    # config block.  For every non-op:// value the original
    # .env-takes-precedence behaviour is preserved unchanged.
    if raw.startswith("op://") and scoped_value:
        return scoped_value
    return raw or scoped_value


def _seed_from_env(provider: str, entries: List[PooledCredential]) -> Tuple[bool, Set[str]]:
    changed = False
    active_sources: Set[str] = set()

    # Copilot has its own dedicated seeding branch (see `_seed_credentials`
    # for provider == "copilot") which exchanges the raw ghu_ OAuth token
    # for the ~437-char api token via `get_copilot_api_token`. If we let
    # the generic env-var loop below run for copilot, it re-reads
    # COPILOT_GITHUB_TOKEN from .env and shoves the RAW 40-char token in
    # as `access_token`, overwriting the correctly-exchanged token. That
    # bypasses the Copilot token exchange entirely and causes 400s with
    # "not available for integrator copilot-language-server" (the server's
    # fallback integrator when it receives a raw OAuth token instead of
    # an api token). Skip the generic loop here — the copilot-specific
    # branch is authoritative.
    if provider == "copilot":
        return False, active_sources

    # The .env-preferring resolution lives at module level
    # (``get_env_prefer_dotenv``) so the pool seeder and the per-turn
    # credential refresh share one implementation.
    _get_env_prefer_dotenv = get_env_prefer_dotenv

    # Honour user suppression — `hermes auth remove <provider> <N>` for an
    # env-seeded credential marks the env:<VAR> source as suppressed so it
    # won't be re-seeded from the user's shell environment or ~/.hermes/.env.
    # Without this gate the removal is silently undone on the next
    # load_pool() call whenever the var is still exported by the shell.
    try:
        from hermes_cli.auth import is_source_suppressed as _is_source_suppressed
    except ImportError:
        def _is_source_suppressed(_p, _s):  # type: ignore[misc]
            return False

    def _secret_source_for_env(env_var: str) -> Optional[str]:
        try:
            from hermes_cli.env_loader import get_secret_source
            source_label = get_secret_source(env_var)
        except Exception:
            source_label = None
        return str(source_label).strip() if source_label else None

    def _env_payload(
        *,
        source: str,
        env_var: str,
        token: str,
        base_url: str,
        auth_type: str = AUTH_TYPE_API_KEY,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "source": source,
            "auth_type": auth_type,
            "access_token": token,
            "base_url": base_url,
            "label": env_var,
        }
        secret_source = _secret_source_for_env(env_var)
        if secret_source:
            payload["secret_source"] = secret_source
        return payload

    if provider == "openrouter":
        # Prefer ~/.hermes/.env over os.environ
        token = _get_env_prefer_dotenv("OPENROUTER_API_KEY")
        if token:
            source = "env:OPENROUTER_API_KEY"
            if _is_source_suppressed(provider, source):
                return changed, active_sources
            active_sources.add(source)
            changed |= _upsert_entry(
                entries,
                provider,
                source,
                _env_payload(
                    source=source,
                    env_var="OPENROUTER_API_KEY",
                    token=token,
                    base_url=OPENROUTER_BASE_URL,
                ),
            )
        return changed, active_sources

    pconfig = PROVIDER_REGISTRY.get(provider)
    if not pconfig or pconfig.auth_type != AUTH_TYPE_API_KEY:
        return changed, active_sources

    env_url = ""
    if pconfig.base_url_env_var:
        env_url = _get_env_prefer_dotenv(pconfig.base_url_env_var).rstrip("/")

    env_vars = list(pconfig.api_key_env_vars)
    if provider == "anthropic":
        env_vars = [
            "ANTHROPIC_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "ANTHROPIC_API_KEY",
        ]

    for env_var in env_vars:
        # Prefer ~/.hermes/.env over os.environ
        token = _get_env_prefer_dotenv(env_var)
        if not token:
            continue
        source = f"env:{env_var}"
        if _is_source_suppressed(provider, source):
            continue
        active_sources.add(source)
        base_url = env_url or pconfig.inference_base_url
        if provider == "kimi-coding":
            base_url = _resolve_kimi_base_url(token, pconfig.inference_base_url, env_url)
        elif provider == "zai":
            base_url = _resolve_zai_base_url(token, pconfig.inference_base_url, env_url)
        changed |= _upsert_entry(
            entries,
            provider,
            source,
            _env_payload(
                source=source,
                env_var=env_var,
                token=token,
                base_url=base_url,
            ),
        )
    return changed, active_sources


def _prune_stale_seeded_entries(
    entries: List[PooledCredential],
    active_sources: Set[str],
    *,
    prune_env_sources: bool = True,
) -> bool:
    def _is_prunable(entry: PooledCredential) -> bool:
        # ``env:*`` entries are persisted references that get re-hydrated from
        # the environment on every load. A process that merely lacks the env
        # var this call must NOT delete the on-disk entry for every other
        # process — that destructive read is the bug behind #9331. Only prune
        # an env source when ``prune_env_sources`` is explicitly requested
        # (e.g. an `hermes auth` command that confirmed the source is gone).
        if entry.source.startswith("env:"):
            return prune_env_sources
        # Codex may retain independent device-code rows on chains other than
        # the current singleton. Terminal quarantine removes only aliases that
        # share the rejected singleton chain; absence of singleton state alone
        # is not authority to delete the unrelated rows that remain in pool.
        if entry.provider == "openai-codex" and entry.source == "device_code":
            return False
        # File-backed singletons (device-code OAuth, claude_code) and Hermes
        # PKCE should disappear from the pool when their backing file is gone.
        return (
            is_borrowed_credential_source(entry.source, entry.provider)
            or entry.source == "hermes_pkce"
        )

    retained = [
        entry
        for entry in entries
        if _is_manual_source(entry.source)
        or entry.source in active_sources
        or not _is_prunable(entry)
    ]
    if len(retained) == len(entries):
        return False
    entries[:] = retained
    return True


def _seed_custom_pool(pool_key: str, entries: List[PooledCredential]) -> Tuple[bool, Set[str]]:
    """Seed a custom endpoint pool from custom_providers config and model config."""
    changed = False
    active_sources: Set[str] = set()

    # Shared suppression gate — same pattern as _seed_from_env/_seed_from_singletons.
    try:
        from hermes_cli.auth import is_source_suppressed as _is_suppressed
    except ImportError:
        def _is_suppressed(_p, _s):  # type: ignore[misc]
            return False

    # Seed from the custom_providers config entry's api_key field
    cp_config = _get_custom_provider_config(pool_key)
    if cp_config:
        api_key = str(cp_config.get("api_key") or "").strip()
        base_url = str(cp_config.get("base_url") or "").strip().rstrip("/")
        name = str(cp_config.get("name") or "").strip()
        if api_key:
            source = f"config:{name}"
            if not _is_suppressed(pool_key, source):
                active_sources.add(source)
                changed |= _upsert_entry(
                    entries,
                    pool_key,
                    source,
                    {
                        "source": source,
                        "auth_type": AUTH_TYPE_API_KEY,
                        "access_token": api_key,
                        "base_url": base_url,
                        "label": name or source,
                    },
                )

    # Seed from model.api_key if model.provider=='custom' and model.base_url matches
    try:
        config = _load_config_safe()
        model_cfg = config.get("model") if config else None
        if isinstance(model_cfg, dict):
            model_provider = str(model_cfg.get("provider") or "").strip().lower()
            model_base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
            model_api_key = ""
            for k in ("api_key", "api"):
                v = model_cfg.get(k)
                if isinstance(v, str) and v.strip():
                    model_api_key = v.strip()
                    break
            if model_provider == "custom" and model_base_url and model_api_key:
                # Check if this model's base_url matches our custom provider
                matched_key = get_custom_provider_pool_key(model_base_url)
                if matched_key == pool_key:
                    source = "model_config"
                    if not _is_suppressed(pool_key, source):
                        active_sources.add(source)
                        changed |= _upsert_entry(
                            entries,
                            pool_key,
                            source,
                            {
                                "source": source,
                                "auth_type": AUTH_TYPE_API_KEY,
                                "access_token": model_api_key,
                                "base_url": model_base_url,
                                "label": "model_config",
                            },
                        )
    except Exception:
        pass

    return changed, active_sources


def load_pool(provider: str) -> CredentialPool:
    provider = (provider or "").strip().lower()
    active_store = _load_auth_store()
    active_pool = active_store.get("credential_pool")
    active_entries = (
        active_pool.get(provider) if isinstance(active_pool, dict) else None
    )
    local_raw_entries = list(active_entries) if isinstance(active_entries, list) else []
    local_entry_ids = {
        str(entry.get("id"))
        for entry in local_raw_entries
        if isinstance(entry, dict) and entry.get("id")
    }
    raw_entries = read_credential_pool(provider)
    disk_ids = {
        str(entry.get("id"))
        for entry in raw_entries
        if isinstance(entry, dict) and entry.get("id")
    }
    raw_needs_sanitization = any(
        isinstance(payload, dict)
        and sanitize_borrowed_credential_payload(payload, provider) != payload
        for payload in raw_entries
    )
    entries = []
    for payload in raw_entries:
        entry = PooledCredential.from_dict(provider, payload)
        if provider == "openai-codex":
            source_path = auth_mod._credential_pool_row_source_path(payload)
            if source_path is not None:
                entry = replace(entry, source_store_path=source_path)
        entries.append(entry)
    raw_needs_auth_normalization = any(
        isinstance(payload, dict)
        and _normalize_pool_auth_type(
            provider,
            payload.get("access_token"),
            payload.get("auth_type", AUTH_TYPE_API_KEY),
        ) != payload.get("auth_type", AUTH_TYPE_API_KEY)
        for payload in raw_entries
    )
    if raw_needs_auth_normalization:
        # A profile may be reading this provider from the global-root fallback.
        # Keep that fallback read-only: only the store that owns these rows may
        # rewrite them. Loading the default/root profile will heal global rows.
        raw_needs_auth_normalization = bool(active_entries)

    if provider.startswith(CUSTOM_POOL_PREFIX):
        # Custom endpoint pool — seed from custom_providers config and model config
        custom_changed, custom_sources = _seed_custom_pool(provider, entries)
        changed = raw_needs_sanitization or raw_needs_auth_normalization or custom_changed
        changed |= _prune_stale_seeded_entries(entries, custom_sources)
    else:
        singleton_changed, singleton_sources = _seed_from_singletons(provider, entries)
        env_changed, env_sources = _seed_from_env(provider, entries)
        changed = (
            raw_needs_sanitization
            or raw_needs_auth_normalization
            or singleton_changed
            or env_changed
        )
        # ``load_pool()`` is a non-destructive read for env-seeded entries: a
        # process missing a provider env var must not delete the persisted
        # pool entry for every other process (#9331). File-backed singletons
        # still prune when their backing file is gone.
        changed |= _prune_stale_seeded_entries(
            entries,
            singleton_sources | env_sources,
            prune_env_sources=False,
        )
        changed |= _normalize_pool_priorities(provider, entries)

    if changed:
        persisted_entries = [
            entry
            for entry in entries
            if not _is_source_owned_elsewhere(entry)
        ]
        new_ids = {entry.id for entry in persisted_entries}
        persisted_disk_ids = (
            local_entry_ids
            if any(_is_source_owned_elsewhere(entry) for entry in entries)
            else disk_ids
        )
        write_credential_pool(
            provider,
            [
                entry.to_dict()
                for entry in sorted(
                    persisted_entries,
                    key=lambda item: item.priority,
                )
            ],
            removed_ids=persisted_disk_ids - new_ids,
        )
    pool = CredentialPool(provider, entries)
    if provider == "openai-codex":
        for entry in pool.entries():
            if entry.source_store_path is not None:
                pool._trust_codex_source_owner(
                    entry,
                    owner_kind=_loaded_codex_source_owner_kind(entry),
                )
    return pool

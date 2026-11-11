"""Plugin-provider refresh support for the credential pool (#116408).

A model-provider plugin makes its pooled OAuth rows refreshable by shipping
``ProviderProfile.refresh_credential(entry) -> Mapping | None``. The pool owns
the locking, persistence and failure classification around that hook; the
plugin owns only the token POST. This sibling module keeps that logic out of
``agent/credential_pool.py`` (near the size cap).

Contract (documented in website/docs/developer-guide/model-provider-plugin.md):

* the hook returns a mapping of rotated values — dataclass field names
  (``access_token``, ``refresh_token``, ``expires_at_ms`` …) replace the row's
  fields, every other key (``expires_in``, ``token_type``, ``scope`` — the
  natural token-endpoint shape) lands in ``entry.extra``; ``None``/empty = could not
  rotate and the pool benches the row like a failed refresh POST;
* raising ``AuthError(..., relogin_required=True)`` (or a grant-dead OAuth code)
  is terminal: the row goes DEAD with a WARNING naming ``hermes auth add``;
  any other exception is transient and only benches the row.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, fields, replace
from urllib.parse import urlsplit
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple

from hermes_cli.auth import _OAUTH_GRANT_DEAD_CODES
from hermes_cli.auth_constants import AuthError

if TYPE_CHECKING:  # pragma: no cover
    from agent.credential_pool import CredentialPool, PooledCredential

logger = logging.getLogger(__name__)

_EXPIRY_SKEW_MS = 120_000


def plugin_row_is_expiring(entry: PooledCredential) -> bool:
    """An expiry-stamped OAuth row (a plugin's, or Anthropic's) is due within the skew of ``expires_at_ms``.

    The token endpoint's ``expires_in`` is the only clock: many providers issue opaque bearers (Google's
    ``ya29.*``) with no JWT ``exp`` to decode, so without this a long session sends the dead bearer.
    """
    return entry.expires_at_ms is not None and int(entry.expires_at_ms) <= int(time.time() * 1000) + _EXPIRY_SKEW_MS


def apply_plugin_refresh_result(entry: PooledCredential, result: Any) -> PooledCredential:
    """Merge a ``refresh_credential`` return value into *entry*.

    Field names go through ``dataclasses.replace``; everything else is merged into ``extra``
    (mirroring ``PooledCredential.from_dict``). Before this split a token-endpoint-shaped mapping
    with ``expires_in`` raised ``TypeError`` inside ``replace`` and the pool benched a row whose
    single-use refresh token the server had already rotated — the login was lost.
    """
    if not result:
        return entry
    mapping: Mapping[str, Any] = dict(result)
    field_names = {f.name for f in fields(type(entry))} - {"provider", "extra"}
    field_updates = {k: v for k, v in mapping.items() if k in field_names}
    extra_updates = {k: v for k, v in mapping.items() if k not in field_names and k != "provider"}
    if extra_updates:
        field_updates["extra"] = {**entry.extra, **extra_updates}
    return replace(entry, **field_updates) if field_updates else entry


def is_terminal_plugin_refresh_error(exc: BaseException) -> bool:
    """True when retrying the same plugin refresh token cannot succeed.

    Plugins have no per-provider code table, so the predicate is the structural one every built-in
    flow shares: a structured ``AuthError`` that asks for a re-login, or one carrying a grant-dead
    OAuth code. Transport errors, 429/5xx-shaped ``RuntimeError`` and plain ``AuthError`` without
    that signal stay transient (EXHAUSTED for one cooldown).
    """
    if not isinstance(exc, AuthError):
        return False
    return bool(exc.relogin_required) or (exc.code or "") in _OAUTH_GRANT_DEAD_CODES


def recover_failed_plugin_refresh(
    pool: CredentialPool, entry: PooledCredential, exc: Exception,
) -> tuple[bool, Optional[PooledCredential]]:
    """Recovery for a plugin hook that raised: adopt a peer's rotation, or quarantine a dead grant.

    Returns ``(handled, result)``; ``handled=False`` means the caller should bench the row as a
    transient failure. A peer process may have rotated the single-use token between our in-lock
    sync and the hook's POST — adopt that pair before classifying the failure.
    """
    from agent.credential_pool import _MARK_OK

    synced = pool._sync_entry_from_pool_store(entry)
    if synced.refresh_token != entry.refresh_token and (synced.access_token or "").strip():
        logger.debug("%s refresh failed but the pool store has newer tokens — adopting", pool.provider)
        return True, pool._adopt(synced, **_MARK_OK)
    if is_terminal_plugin_refresh_error(exc):
        # WARNING, not debug: this is the moment a login is lost. Benching for a TTL would replay
        # the dead token every cooldown at DEBUG with no trace for the user.
        logger.warning(
            "%s refresh token for %s is terminally invalid (%s); the credential leaves rotation. "
            "Re-run 'hermes auth add %s' to sign in again.",
            pool.provider, entry.label or entry.id[:8], exc, pool.provider,
        )
        pool._mark_dead_refresh_grant(entry, exc)
        return True, None
    return False, None


# --- Ephemeral runtime credentials (``prepare_pool_entry`` / ``refresh_pool_runtime_credentials``) ---


@dataclass(frozen=True, eq=False)
class RuntimeOverride:
    """A derived key/endpoint bound to the durable grant it was minted from; never persisted."""

    access_token: str = field(repr=False)
    source_value: str = field(repr=False)
    api_key: str = field(repr=False)
    base_url: Optional[str] = None


def durable_runtime_key(entry: PooledCredential) -> str:
    """The row's own runtime key, ignoring any derived override."""
    if entry.provider == "nous":
        # Nous stores the runtime inference credential in agent_key for
        # compatibility. It must be a NAS invoke JWT.
        import hermes_cli.auth as auth_mod
        for token, expires_at in (
            (entry.agent_key, entry.agent_key_expires_at),
            (entry.access_token, entry.expires_at),
        ):
            if (
                isinstance(token, str)
                and token.strip()
                and auth_mod._nous_invoke_jwt_is_usable(
                    token, scope=getattr(entry, "scope", None), expires_at=expires_at,
                )
            ):
                return token.strip()
        return ""
    return str(entry.access_token or "")


def active_runtime_override(entry: Any) -> Optional[RuntimeOverride]:
    """The row's override while its durable grant and source key are unchanged, else None."""
    override = getattr(entry, "__dict__", {}).get("_runtime_override")
    if not isinstance(override, RuntimeOverride):
        return None
    if override.access_token != entry.access_token or override.source_value != durable_runtime_key(entry):
        return None
    return override


def _install(entry: Any, override: Optional[RuntimeOverride]) -> None:
    object.__setattr__(entry, "_runtime_override", override)


def exact_https_origin(url: Any) -> Optional[str]:
    """``https://host`` for a complete host-only HTTPS URL; None for any path, query, port, userinfo."""
    if not isinstance(url, str) or not url or url != url.strip() or "\\" in url:
        return None
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    host = parts.hostname or ""
    if (parts.scheme != "https" or not host or port not in (None, 443) or parts.username is not None
            or parts.password is not None or parts.path or parts.query or parts.fragment or url.endswith(("?", "#"))
            or host.endswith(".") or host != parts.netloc.lower().removesuffix(":443")):
        return None
    return f"https://{host}"


def _allowed_origins(profile: Any) -> set[str]:
    raw = getattr(profile, "credential_pool_base_urls", ()) or ()
    if isinstance(raw, str):
        return set()
    return {origin for origin in map(exact_https_origin, raw) if origin}


def _overrides(profile: Any, name: str) -> bool:
    from providers.base import ProviderProfile
    hook = getattr(type(profile), name, None)
    return hook is not None and hook is not getattr(ProviderProfile, name, None)


def _tentative_override(entry: Any, result: Any, *, profile: Any, source_base_url: Any,
                        typed_only: bool = False) -> Optional[RuntimeOverride]:
    """Validate a hook result completely before anything is installed on the row."""
    from hermes_cli.auth import has_usable_secret
    from providers.base import PoolRuntimeCredential

    if isinstance(result, PoolRuntimeCredential):
        api_key, base_url = result.api_key, result.base_url
    elif isinstance(result, str) and not typed_only:
        api_key, base_url = result, None  # legacy: key only, the row keeps its endpoint
    else:
        return None
    if not has_usable_secret(api_key):
        return None
    if base_url is not None:
        allowed = _allowed_origins(profile)
        origin = exact_https_origin(base_url)
        if origin is None or origin not in allowed or exact_https_origin(source_base_url) not in allowed:
            return None
        base_url = origin
    return RuntimeOverride(entry.access_token, durable_runtime_key(entry), api_key, base_url)


def _serves_source(entry: Any, override: Optional[RuntimeOverride], profile: Any, source_base_url: Any) -> bool:
    """Adopting the row from ``source_base_url`` must not move the session to a foreign endpoint."""
    if source_base_url is None:
        return True
    if override is not None and override.base_url is not None:
        return exact_https_origin(source_base_url) in _allowed_origins(profile)
    from agent.credential_pool import credential_pool_entry_serves_endpoint
    return credential_pool_entry_serves_endpoint(entry, source_base_url)


def _profile(provider: str) -> Any:
    from providers import get_provider_profile
    return get_provider_profile(provider)


def prepare_runtime_entry(pool: CredentialPool, entry: Any, *, source_base_url: Any = None) -> Any:
    """The live row ready for dispatch from ``source_base_url``, or None when it must not be adopted."""
    from hermes_cli.auth import has_usable_secret

    if entry is None:
        return None
    profile = _profile(pool.provider)
    with pool._lock:
        live = pool._find(lambda e: e.id == getattr(entry, "id", None)) or entry
        override = active_runtime_override(live)
        if override is not None:
            return live if _serves_source(live, override, profile, source_base_url) else None
        if hasattr(live, "__dict__"):
            _install(live, None)  # a populated but invalidated override never bypasses preparation
        grant = getattr(live, "access_token", None)
        pooled = any(e is live for e in pool._entries)
    if pool.provider == "nous" or profile is None or not _overrides(profile, "prepare_pool_entry"):
        usable = has_usable_secret(getattr(live, "runtime_api_key", ""))
        return live if usable and _serves_source(live, None, profile, source_base_url) else None
    source_value = durable_runtime_key(live)
    try:
        result = profile.prepare_pool_entry(live, source_value)
    except Exception as exc:  # health: allow BLE001 -- plugin hook boundary; exception text may carry secrets
        logger.debug("%s could not prepare pool row %s (%s)", pool.provider, live.id, type(exc).__name__)
        return None
    origin = live.runtime_base_url if source_base_url is None else source_base_url
    tentative = _tentative_override(live, result, profile=profile, source_base_url=origin)
    if tentative is None or not _serves_source(live, tentative, profile, source_base_url):
        logger.info("%s pool row %s prepared runtime credential refused", pool.provider, live.id)
        return None
    with pool._lock:
        # The hook ran unlocked: a concurrent renewal, grant rotation or removal wins over this result.
        if pooled and not any(e is live for e in pool._entries):
            return None
        if getattr(live, "access_token", None) != grant or durable_runtime_key(live) != source_value:
            return None
        if hasattr(live, "__dict__") and live.__dict__.get("_runtime_override") is not None:
            current = active_runtime_override(live)
            return live if current is not None and _serves_source(live, current, profile, source_base_url) else None
        # Typed results declare derived credentials even when a faulty hook returns its input.
        # Preserve that provenance so explicit-key dispatch cannot mistake them for legacy identity strings.
        if tentative.api_key != source_value or tentative.base_url is not None or not isinstance(result, str):
            _install(live, tentative)
    return live


def _scoped_row(pool: CredentialPool, credential_id: Optional[str], api_key_hint: Optional[str]) -> Any:
    if not credential_id or not api_key_hint:
        return None
    with pool._lock:
        row = pool._find(lambda e: e.id == credential_id)
        if row is None or row.runtime_api_key != api_key_hint:
            return None
        return row


def refresh_runtime_matching(pool: CredentialPool, *, credential_id: Optional[str], api_key_hint: Optional[str],
                             reason: str) -> Any:
    """Re-mint the dispatched row's runtime credential; None declines without benching the row."""
    from agent.credential_auth_recovery import pool_runtime_refresh_profile

    row = _scoped_row(pool, credential_id, api_key_hint)
    profile = pool_runtime_refresh_profile(pool) if row is not None else None
    if profile is None:
        return None
    grant, source_base_url = row.access_token, row.runtime_base_url
    try:
        result = profile.refresh_pool_runtime_credentials(row, reason=reason)
    except Exception as exc:  # health: allow BLE001 -- plugin hook boundary; exception text may carry secrets
        logger.debug("%s runtime renewal for row %s failed (%s)", pool.provider, row.id, type(exc).__name__)
        return None
    tentative = _tentative_override(row, result, profile=profile, source_base_url=source_base_url, typed_only=True)
    with pool._lock:
        live = pool._find(lambda e: e.id == row.id)
        if tentative is None or live is not row or row.access_token != grant:
            return None
        _install(row, tentative)
        pool._current_id = row.id
    return row


class PoolRuntimeCredentialMixin:
    """CredentialPool surface for ephemeral, route-checked runtime credentials."""

    def prepare_runtime_entry(self, entry: Any, *, source_base_url: Any = None) -> Any:
        prepared = prepare_runtime_entry(self, entry, source_base_url=source_base_url)
        if prepared is not None and active_runtime_override(prepared) is not None:
            from agent.auxiliary_pool_recovery import remember
            remember(self)
        return prepared

    def try_refresh_runtime_matching(self, *, credential_id: Optional[str] = None,
                                     api_key_hint: Optional[str] = None, reason: str = "unauthorized") -> Any:
        refreshed = refresh_runtime_matching(self, credential_id=credential_id, api_key_hint=api_key_hint, reason=reason)
        if refreshed is not None:
            from agent.auxiliary_pool_recovery import remember
            remember(self)
        return refreshed

    def supports_runtime_credential_refresh(self) -> bool:
        from agent.credential_auth_recovery import pool_runtime_refresh_profile
        return pool_runtime_refresh_profile(self) is not None

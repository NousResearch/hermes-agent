"""Plugin-provider refresh support for the credential pool (#116408).

A model-provider plugin makes its pooled OAuth rows refreshable by shipping
``ProviderProfile.refresh_credential(entry) -> Mapping | None``. The pool owns
the locking, persistence and failure classification around that hook; the
plugin owns only the token POST. This sibling module keeps that logic out of
``agent/credential_pool.py`` (near the size cap).

Contract (documented in website/docs/developer-guide/model-provider-plugin.md):

* the hook returns a mapping of rotated values — refreshable dataclass field names
  (``access_token``, ``refresh_token``, ``expires_at_ms`` …) replace the row's
  fields, pool-owned field names are refused with a warning, every non-field key
  (``expires_in``, ``token_type``, ``scope`` — the natural token-endpoint shape)
  lands in ``entry.extra``; ``None``/empty or a result with no credential field = could not
  rotate and the pool benches the row like a failed refresh POST;
* raising ``AuthError(..., relogin_required=True)`` (or a grant-dead OAuth code)
  is terminal: the row goes DEAD with a WARNING naming ``hermes auth add``;
  any other exception is transient and only benches the row.
"""

from __future__ import annotations

import logging
import time
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple

from hermes_cli.auth import _OAUTH_GRANT_DEAD_CODES
from hermes_cli.auth_constants import AuthError

if TYPE_CHECKING:  # pragma: no cover
    from agent.credential_pool import CredentialPool, PooledCredential

logger = logging.getLogger(__name__)

_EXPIRY_SKEW_MS = 120_000


def plugin_row_is_expiring(entry: "PooledCredential") -> bool:
    """An expiry-stamped OAuth row (a plugin's, or Anthropic's) is due within the skew of ``expires_at_ms``.

    The token endpoint's ``expires_in`` is the only clock: many providers issue opaque bearers (Google's
    ``ya29.*``) with no JWT ``exp`` to decode, so without this a long session sends the dead bearer.
    """
    return entry.expires_at_ms is not None and int(entry.expires_at_ms) <= int(time.time() * 1000) + _EXPIRY_SKEW_MS


# Field names a rotation may legitimately replace: the token pair, its expiry, and the
# runtime-credential pair. Everything else on the row is pool-owned — identity/classification
# fields (``id`` rekeys the persist merge, ``source`` flips the row to borrowed and the next
# persist strips its tokens, ``priority``/``auth_type`` relabel it), endpoint fields
# (``base_url``/``inference_base_url`` would redirect the next request carrying the fresh token),
# and status bookkeeping. A colliding key is dropped rather than routed to ``extra``: ``to_dict``
# flattens ``extra`` back to top level, so a colliding value would die silently at persist anyway.
_REFRESHABLE_FIELDS = frozenset({
    "access_token", "refresh_token",
    "expires_at", "expires_at_ms", "last_refresh",
    "agent_key", "agent_key_expires_at",
})

# Rotation only counts if the response carried credential material; an expiry- or metadata-only
# result must bench like an empty one instead of reporting a stale bearer as refreshed.
_CREDENTIAL_FIELDS = frozenset({"access_token", "refresh_token", "agent_key"})


def _refreshable_value_ok(key: str, value: Any) -> bool:
    """A refused value keeps the stored one — never let a malformed rotation erase live material."""
    if key in _CREDENTIAL_FIELDS:
        return isinstance(value, str) and bool(value.strip())
    if key == "expires_at_ms":
        return value is None or isinstance(value, int)
    return value is None or isinstance(value, str)  # expires_at, last_refresh, agent_key_expires_at


def apply_plugin_refresh_result(entry: "PooledCredential", result: Any) -> "PooledCredential":
    """Merge a ``refresh_credential`` return value into *entry*.

    Keys in ``_REFRESHABLE_FIELDS`` go through ``dataclasses.replace``; other field names and
    malformed refreshable values are refused with a warning; everything else is merged into
    ``extra`` (mirroring ``PooledCredential.from_dict``). A result carrying no credential field is
    not a rotation: the entry is returned unchanged so the caller benches the row like an empty
    result instead of stamping a stale bearer refreshed.
    """
    if not result:
        return entry
    mapping: Mapping[str, Any] = dict(result)
    all_fields = {f.name for f in fields(type(entry))}
    field_updates: dict = {}
    extra_updates: dict = {}
    refused = []
    for key, value in mapping.items():
        if key in _REFRESHABLE_FIELDS:
            # Numeric strings/floats are common off a JSON token endpoint; only the
            # genuinely non-numeric are refused (an unparseable value crashes the
            # int() readers downstream rather than landing as a bad timestamp).
            if key == "expires_at_ms" and value is not None and not isinstance(value, int):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    refused.append(key)
                    continue
            if _refreshable_value_ok(key, value):
                field_updates[key] = value
            else:
                refused.append(key)
        elif key == "extra" and isinstance(value, Mapping):
            extra_updates.update(value)
        elif key in all_fields:
            refused.append(key)
        else:
            extra_updates[key] = value
    if refused:
        logger.warning("plugin refresh_credential for %s (%s) returned unusable field(s) %s; "
                       "keeping stored values (see model-provider-plugin.md)",
                       entry.provider, entry.label or entry.id, ", ".join(sorted(set(refused))))
    if not (set(field_updates) & _CREDENTIAL_FIELDS):
        return entry
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
    pool: "CredentialPool", entry: "PooledCredential", exc: Exception,
) -> Tuple[bool, Optional["PooledCredential"]]:
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

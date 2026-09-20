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
  natural token-endpoint shape) lands in ``entry.extra``; ``None`` = no rotation;
* raising ``AuthError(..., relogin_required=True)`` (or a grant-dead OAuth code)
  is terminal: the row goes DEAD with a WARNING naming ``hermes auth add``;
  any other exception is transient and only benches the row.
"""

from __future__ import annotations

import logging
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple

from hermes_cli.auth import _OAUTH_GRANT_DEAD_CODES
from hermes_cli.auth_constants import AuthError

if TYPE_CHECKING:  # pragma: no cover
    from agent.credential_pool import CredentialPool, PooledCredential

logger = logging.getLogger(__name__)


def apply_plugin_refresh_result(entry: "PooledCredential", result: Any) -> "PooledCredential":
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

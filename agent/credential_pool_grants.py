"""Refresh-grant generation reconciliation and terminal retirement.

Late-bind credential_pool's store adapters to preserve its lock and test seams.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential


class DeadRefreshGrant(Exception):
    """Retire a rejected grant after releasing the refresher's auth-store flock."""

    def __init__(self, entry: PooledCredential, exc: Exception):
        super().__init__()
        self.entry = entry
        self.exc = exc


class CredentialPoolGrantsMixin:
    def _pool_store_pair_is_peer_rotation(
        self: Any, entry: PooledCredential, pair: tuple[Any, Any],
    ) -> bool:
        if pair == (entry.access_token, entry.refresh_token):
            return False
        # An unchanged disk generation is not a peer rotation: our own refresh may
        # have succeeded while its save failed. Preserve that live pair for a retry.
        base = self._persisted_token_pairs.get(entry.id)
        return self.provider != "openai-codex" or base is None or pair != base

    def _mark_dead_refresh_grant(self: Any, entry: PooledCredential, exc: Exception) -> None:
        """Mark a row whose refresh token was terminally rejected DEAD, if the quarantine kept it.

        ``_quarantine_sources`` drops only singleton-seeded rows; an independent ``manual:*`` login
        (``hermes auth add``) survives, and an unmarked survivor re-enters rotation and re-fires the
        terminal WARNING on every later refresh attempt. DEAD leaves rotation until a write-side
        re-auth sync clears it (never via TTL).
        """
        from agent import credential_pool as cp

        with self._lock:
            current = next((item for item in self._entries if item.id == entry.id), None)
            if current is None or current.last_status == cp.STATUS_DEAD:
                return
            self._adopt(
                current,
                last_status=cp.STATUS_DEAD,
                last_status_at=cp.time.time(),
                last_error_code=None,
                last_error_reason=str(getattr(exc, "code", None) or "invalid_grant"),
                last_error_message=str(exc),
            )

    def _retire_dead_refresh_grant(self: Any, entry: PooledCredential, exc: Exception) -> None:
        """Retire exactly the Codex rows still holding *entry*'s rejected refresh token.

        The unit that died is one refresh-token chain, not a source: dropping every ``device_code``
        row deleted an independent account whenever another account's grant died, while a same-grant
        ``manual:*`` alias stayed selectable. Rows are first reconciled with their persisted copies,
        so a row a peer rotated or re-authed after this pool loaded keeps its new pair instead of being
        retired or overwritten by this process's stale snapshot, and a row a peer added still holding
        the dead token is retired too (``_persist`` would otherwise merge it back as-is). Matching
        ``device_code`` rows are dropped (their singleton was just cleared); other matches go DEAD
        until a re-auth. Pool lock first, then the auth-store flock — never the reverse.
        """
        from agent import credential_pool as cp

        dead_refresh = (entry.refresh_token or "").strip()

        def holds_dead_grant(item: PooledCredential) -> bool:
            return (item.refresh_token or "").strip() == dead_refresh if dead_refresh else item.id == entry.id

        with self._lock, cp._auth_store_lock():
            persisted = {row.get("id"): row for row in cp.read_credential_pool(self.provider) if isinstance(row, dict)}
            reconciled = []
            for item in self._entries:
                row = persisted.pop(item.id, None) or {}
                stored = (row.get("access_token"), row.get("refresh_token"))
                if any(stored) and self._pool_store_pair_is_peer_rotation(item, stored):
                    item = cp.PooledCredential.from_dict(self.provider, row)
                reconciled.append(item)
            peer_added = (cp.PooledCredential.from_dict(self.provider, row) for row in persisted.values() if row.get("id"))
            reconciled.extend(item for item in peer_added if holds_dead_grant(item))
            removed_ids = [item.id for item in reconciled if item.source == "device_code" and holds_dead_grant(item)]
            self._entries = sorted(
                (item for item in reconciled if item.id not in removed_ids), key=lambda item: item.priority)
            if self._current_id in removed_ids:
                self._current_id = None
            # Persist the removal before marking survivors: a plain persist merges disk-only rows back.
            self._persist(removed_ids=removed_ids)
            for item in [item for item in self._entries if holds_dead_grant(item)]:
                self._mark_dead_refresh_grant(item, exc)

    def _clear_terminal_tokens_state(self: Any, entry: PooledCredential, exc: Exception) -> None:
        """Drop the dead Codex/xAI token pair from auth.json unless a peer already rotated it."""
        from agent import credential_pool as cp

        logger = cp.logger
        display = cp._TOKENS_SINGLETON_PROVIDERS[self.provider][1]
        try:
            with cp._auth_store_lock():
                auth_store = cp._load_auth_store()
                state = cp._load_provider_state(auth_store, self.provider) or {}
                tokens = (state.get("tokens") or {}) if isinstance(state, dict) else None
                if isinstance(tokens, dict):
                    store_refresh = str(tokens.get("refresh_token") or "").strip()
                    store_access = str(tokens.get("access_token") or "").strip()
                    if store_refresh:
                        same_grant = store_refresh == str(entry.refresh_token or "").strip()
                    elif self.provider == "openai-codex":
                        # An access-only Codex singleton is this grant only if it carries this row's
                        # bearer; a different one is another login (A's, while independent B's died).
                        same_grant = not store_access or store_access == str(entry.access_token or "").strip()
                    else:
                        same_grant = True
                    if same_grant:
                        tokens.pop("access_token", None)
                        tokens.pop("refresh_token", None)
                        state["tokens"] = tokens
                        state["last_auth_error"] = {
                            "provider": self.provider,
                            "code": getattr(exc, "code", "unknown"),
                            "message": str(exc),
                            "reason": "credential_pool_refresh_failure",
                            "relogin_required": True,
                            "at": cp.datetime.now(cp.UTC).isoformat(),
                        }
                        cp._save_provider_state(auth_store, self.provider, state)
                        cp._save_auth_store(auth_store)
        except Exception as clear_exc:
            logger.error("Failed to clear terminal %s OAuth state: %s", display, clear_exc, exc_info=True)

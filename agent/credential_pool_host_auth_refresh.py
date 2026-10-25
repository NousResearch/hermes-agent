"""Exact-row refresh guard used only by selected Codex host authentication."""

from __future__ import annotations

from typing import Any, Optional

from hermes_cli.auth import _auth_store_lock, read_credential_pool


class CredentialPoolHostAuthRefreshMixin:
    """Keep the host-auth principal check inside the pool and auth-store locks."""

    def try_refresh_matching(
        self: Any,
        api_key_hint: Optional[str] = None,
        credential_id: Optional[str] = None,
        expected_principal: Optional[tuple[str, str]] = None,
    ) -> Optional[Any]:
        """Refresh the failed row exactly; unguarded callers retain legacy selection semantics."""
        from agent.credential_pool import (
            AUTH_TYPE_OAUTH,
            STATUS_DEAD,
            PooledCredential,
            _codex_principal_identity,
        )

        with self._lock:
            if expected_principal is not None:
                if self.provider != "openai-codex" or not credential_id:
                    return None
                entry = self._find(lambda candidate: candidate.id == credential_id)
                if (entry is None or entry.provider != "openai-codex" or entry.auth_type != AUTH_TYPE_OAUTH
                        or _codex_principal_identity(entry.access_token) != expected_principal):
                    return None
                with _auth_store_lock(timeout_seconds=self._single_use_refresh_lock_timeout()):
                    persisted = next(
                        (row for row in read_credential_pool(self.provider)
                         if isinstance(row, dict) and row.get("id") == credential_id),
                        None,
                    )
                    if not isinstance(persisted, dict):
                        return None
                    current = PooledCredential.from_dict(self.provider, persisted)
                    if (current.id != credential_id or current.provider != "openai-codex"
                            or current.auth_type != AUTH_TYPE_OAUTH or current.last_status == STATUS_DEAD
                            or _codex_principal_identity(current.access_token) != expected_principal):
                        return None
                    self._persisted_token_pairs[current.id] = (current.access_token, current.refresh_token)
                    self._replace_entry(entry, current)
                    self._current_id = current.id
                    return self._refresh_entry_impl(current, force=True, expected_principal=expected_principal)

            entry = self._find(lambda row: row.id == credential_id) if credential_id else None
            if entry is None:
                if api_key_hint:
                    entry = self._find(lambda row: row.runtime_api_key == api_key_hint)
                else:
                    entry = self._current_unlocked() or self._select_unlocked(refresh=False, count=False)[0]
            if entry is None:
                return None
            self._current_id = entry.id
            return self._try_refresh_current_unlocked()

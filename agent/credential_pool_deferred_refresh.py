"""Run deferred OAuth transactions without holding the live pool mutex."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, replace
from threading import Event
from typing import Any

from agent.credential_pool import CredentialPool, PooledCredential, SOURCE_MANUAL, STATUS_DEAD, STATUS_EXHAUSTED
from hermes_cli import auth


class _DeferredCredentialPool(CredentialPool):
    """Reuse the refresh transaction with a private mutex and one staged entry."""

    def __init__(self, provider: str, entry: PooledCredential, borrowed: bool, base: tuple[Any, Any] | None):
        super().__init__(provider, [entry])
        self.original = entry
        if borrowed:
            self._borrowed_root_ids.add(entry.id)
        if base is not None:
            self._persisted_token_pairs[entry.id] = base

    @contextmanager
    def _single_use_refresh_store(self, entry: PooledCredential, borrowed: bool, claude_code: bool):
        with super()._single_use_refresh_store(entry, borrowed, claude_code) as (eligible, root):
            # A local row removed after selection must not be written back by the staged copy.
            if eligible and not borrowed and entry.id in self._persisted_token_pairs:
                _, rows = auth._pool_rows(auth._load_auth_store(), self.provider)
                eligible = entry.id in auth._entry_ids(rows)
            yield eligible, root

    def _persistence_payloads(self) -> list[dict[str, Any]]:
        updated = super()._persistence_payloads()
        if self._borrowed_root_ids:
            return updated  # Root persistence already merges only matching IDs.

        # Read the local file under its transaction lock. Keep unrelated rows and their order.
        _, rows = auth._pool_rows(auth._load_auth_store(), self.provider)
        for index, row in enumerate(rows):
            if isinstance(row, dict) and self._is_original_row(row):
                return [*rows[:index], *updated, *rows[index + 1:]]
        return [*rows, *updated]

    def _is_original_row(self, row: dict[str, Any]) -> bool:
        if row.get("id"):
            return row["id"] == self.original.id
        return (row.get("source", SOURCE_MANUAL) == self.original.source
                and auth._credential_token_pair(row) == (self.original.access_token, self.original.refresh_token))


def refresh_deferred_entry(pool: CredentialPool, entry: PooledCredential, *, wait: bool) -> None:
    """Snapshot and publish under the live mutex, with all file I/O in between."""
    with pool._lock:
        pending = pool._deferred_refreshes.get(entry.id)
        if pending is None:
            current = pool._find(lambda row: _same_generation(row, entry))
            if current is None or (
                current.last_status != entry.last_status and current.last_status in (STATUS_DEAD, STATUS_EXHAUSTED)
            ):
                return
            entry = current
            finished = pool._deferred_refreshes[entry.id] = Event()
            borrowed = entry.id in pool._borrowed_root_ids
            base = pool._persisted_token_pairs.get(entry.id)

    if pending is not None:
        if wait:
            pending.wait()
        return

    try:
        staged = _DeferredCredentialPool(pool.provider, entry, borrowed, base)
        staged._refresh_entry(entry, force=False)
        _publish_deferred_entry(pool, entry, staged)
    finally:
        with pool._lock:
            del pool._deferred_refreshes[entry.id]
            finished.set()


def _publish_deferred_entry(pool: CredentialPool, entry: PooledCredential, staged: _DeferredCredentialPool) -> None:
    with pool._lock:
        for index, current in enumerate(pool._entries):
            if not _same_generation(current, entry):
                continue
            if staged._entries:
                pool._entries[index] = _merge_refreshed_entry(entry, current, staged._entries[0])
                if entry.id in staged._persisted_token_pairs:
                    pool._persisted_token_pairs[entry.id] = staged._persisted_token_pairs[entry.id]
            else:
                del pool._entries[index]
                pool._persisted_token_pairs.pop(entry.id, None)
                if pool._current_id == entry.id:
                    pool._current_id = None
            return


def _same_generation(current: PooledCredential, expected: PooledCredential) -> bool:
    """Round-robin can rebuild the row while keeping the same grant."""
    return (
        current.id == expected.id and current.source == expected.source and current.auth_type == expected.auth_type
        and current.access_token == expected.access_token and current.refresh_token == expected.refresh_token
        and current.last_refresh == expected.last_refresh
    )


def _merge_refreshed_entry(
    original: PooledCredential, current: PooledCredential, refreshed: PooledCredential,
) -> PooledCredential:
    """Publish refresh changes without reverting metadata updated on the live row."""
    changes = {}
    for field in fields(original):
        name = field.name
        if name in {"priority", "request_count"}:
            continue
        before, updated = getattr(original, name), getattr(refreshed, name)
        if before != updated and getattr(current, name) == before:
            changes[name] = updated
    return replace(current, **changes) if changes else current

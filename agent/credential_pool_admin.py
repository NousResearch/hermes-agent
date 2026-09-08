"""Locked credential-pool administration and target resolution."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential


def _cleared_status_copy(entry: PooledCredential) -> PooledCredential:
    from agent.credential_pool import _CLEAR_STATUS

    return replace(entry, **_CLEAR_STATUS,
                   extra={k: v for k, v in entry.extra.items() if k != "failure_reason"})


class CredentialPoolAdminMixin:
    def reset_status(self, credential_id: str) -> Optional[PooledCredential]:
        """Clear only the target's local error state, preserving sibling cooldowns."""
        with self._lock:
            entry = self._find(lambda e: e.id == credential_id)
            if entry is None:
                return None
            cleared = _cleared_status_copy(entry)
            self._replace_entry(entry, cleared)
            self._persist(status_cleared_ids=[cleared.id])
            return cleared
    def reset_statuses(self) -> int:
        """Clear exhaustion state on every entry. Returns how many were cleared.

        ``failure_reason`` lives in ``extra``, not a dataclass field, so it is
        stripped explicitly. The persist declares the cleared ids because the
        disk-recency merge reads a cleared ``last_status_at`` (None -> epoch 0)
        as a stale snapshot and would copy a still-binding cooldown back.
        """
        from agent.credential_pool import _CLEAR_STATUS

        with self._lock:
            stale = [
                e for e in self._entries
                if e.last_status or e.last_status_at or e.last_error_code or e.failure_reason
            ]
            if stale:
                stale_ids = {e.id for e in stale}
                self._entries = [
                    _cleared_status_copy(e) if e.id in stale_ids else e
                    for e in self._entries
                ]
                self._persist(status_cleared_ids=list(stale_ids))
            return len(stale)

    def remove_index(self, index: int) -> Optional[PooledCredential]:
        from agent.credential_pool import persist_pool_entries

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

    def move_entry(self, credential_id: str, priority: int) -> Optional[PooledCredential]:
        """Place an entry at a clamped zero-based position and persist contiguous priorities."""
        from agent.credential_pool import _normalize_pool_priorities

        with self._lock:
            entry = self._find(lambda e: e.id == credential_id)
            if entry is None:
                return None
            others = [e for e in self._entries if e.id != credential_id]
            others.insert(max(0, min(int(priority), len(others))), entry)
            entries = [replace(e, priority=p) for p, e in enumerate(others)]
            # Apply load-time ordering now so the reported position survives reload.
            _normalize_pool_priorities(self.provider, entries)
            self._entries = sorted(entries, key=lambda e: e.priority)
            self._persist()
            return self._find(lambda e: e.id == credential_id)

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
        from agent.credential_pool import _next_priority, write_credential_pool

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

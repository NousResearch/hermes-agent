"""Private local bootstrap grants; never put these credentials in discovery."""
from __future__ import annotations

import hashlib
import secrets
import threading
import time

_PURPOSE_CAPABILITIES = {
    'interactive': frozenset({'session:create', 'session:read', 'session:submit',
                              'session:control', 'session:approve', 'session:respond'}),
    'native-http': frozenset({'http:owner'}),
    'exposure': frozenset({'transport:delegate'}),
    'worker-adoption': frozenset({'worker:adopt'}),
}


class TicketStore:
    TTL_SECONDS = 30
    MAX_ENTRIES = 1024

    def __init__(self, instance_id: str, profile_ids: frozenset[str]):
        if not instance_id or not profile_ids:
            raise ValueError('bootstrap requires instance and served profiles')
        self.instance_id = instance_id
        self.profile_ids = frozenset(profile_ids)
        self._entries: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def mint(self, *, profile_id, subject, purpose) -> str:
        if profile_id not in self.profile_ids or purpose not in _PURPOSE_CAPABILITIES or not subject:
            raise PermissionError('bootstrap binding rejected')
        with self._lock:
            now = time.monotonic()
            self._entries = {k: v for k, v in self._entries.items() if v[0] > now}
            if len(self._entries) >= self.MAX_ENTRIES:
                raise PermissionError('bootstrap capacity exhausted')
            ticket = secrets.token_urlsafe(32)
            self._entries[hashlib.sha256(ticket.encode()).hexdigest()] = (
                now + self.TTL_SECONDS,
                {'instance_id': self.instance_id, 'profile_id': profile_id, 'subject': subject,
                 'purpose': purpose, 'capabilities': _PURPOSE_CAPABILITIES[purpose]})
            return ticket

    def redeem(self, ticket, *, profile_id, purpose) -> dict:
        """``profile_id=None`` accepts a ticket minted for ANY served profile; the returned grant's
        ``profile_id`` then selects the authority. A named profile must match exactly."""
        if not isinstance(ticket, str) or len(ticket) > 256:
            raise PermissionError('invalid bootstrap ticket')
        key = hashlib.sha256(ticket.encode()).hexdigest()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise PermissionError('invalid bootstrap ticket')
            expires, grant = entry
            if expires <= time.monotonic():
                del self._entries[key]
                raise PermissionError('expired bootstrap ticket')
            served = grant['profile_id'] in self.profile_ids if profile_id is None else grant['profile_id'] == profile_id
            if not served or grant['purpose'] != purpose:
                raise PermissionError('bootstrap binding rejected')
            del self._entries[key]
            return dict(grant)

    def revoke(self) -> None:
        with self._lock:
            self._entries.clear()

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
    # Same-user host operation, not permission to read or mutate another profile.
    'native-profile-create': frozenset({'profiles:create'}),
    # Rename/delete of the ONE profile it is minted for (#140345): a registry write on that home,
    # never a read of it or any other route.
    'native-profile-registry': frozenset({'profiles:registry'}),
    # A same-user exposure proxy (the Webapp) attaching for the remote login it authenticated: that
    # login's operator rights, never native ones (gateway-internals.md, "must not replace remote
    # callers' credentials with its own native grant"). See delegated_identity.
    'exposure': frozenset({'transport:delegate'}),
    'worker-adoption': frozenset({'worker:adopt'}),
}
# A grant's profile reach: its own home, or every served sibling for the purposes that attach a
# shared socket (Desktop's native one, the exposure proxy's delegated one).
_SCOPES = frozenset({'profile', 'host'})
_HOST_SCOPED_PURPOSES = frozenset({'interactive', 'exposure'})


def delegable_login(subject):
    """``(provider, issuer, user)`` of a login an exposure grant may carry, else None.

    Only the structured ``auth:v1`` form names a forwarded login. A ``local`` (or empty) provider
    would rebuild a native-looking principal at redemption (``authenticated_subject`` keeps a
    ``local`` provider's bare user id), so it never names one."""
    from gateway.session_identity import parse_authenticated_subject
    parts = parse_authenticated_subject(subject)
    if parts is None:
        return None
    provider, _issuer, user = parts
    if provider.strip() in {'', 'local'} or not user.strip():
        return None
    return parts


def grant_identity(grant) -> dict:
    """The native principal a redeemed interactive/worker ticket attaches as (the private bootstrap
    stamps profile, instance and capabilities: ``authenticated_subject`` keeps its subject bare)."""
    return {'user_id': grant['subject'], 'provider': 'local', 'profile_id': grant['profile_id'],
            'instance_id': grant['instance_id'], 'capabilities': grant['capabilities'],
            'native_bootstrap': True, 'profile_scope': grant.get('scope', 'profile')}


def delegated_identity(grant):
    """The principal a redeemed exposure ticket attaches as, else None.

    The forwarded login, shaped exactly like the dashboard WS gate's own identity for that login,
    with interactive session rights: its subject re-namespaces to the same ``auth:v1`` string the
    proxy's REST forwards carry. No ``native_bootstrap`` and never a ``local`` provider, so it binds
    no native transport (no first-claim adoption of unowned history) and owns no store-wide verb."""
    login = delegable_login(grant.get('subject'))
    if login is None:
        return None
    provider, issuer, user = login
    return {'user_id': user, 'provider': provider, 'issuer': issuer, 'profile_id': grant['profile_id'],
            'instance_id': grant['instance_id'], 'capabilities': _PURPOSE_CAPABILITIES['interactive'],
            'profile_scope': grant.get('scope', 'profile')}


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

    def mint(self, *, profile_id, subject, purpose, scope='profile') -> str:
        """``scope='host'`` lets an interactive or exposure connection select every profile this
        process serves (one shared socket); the default binds exactly ``profile_id``."""
        if (profile_id not in self.profile_ids or purpose not in _PURPOSE_CAPABILITIES or not subject
                or scope not in _SCOPES or (scope == 'host' and purpose not in _HOST_SCOPED_PURPOSES)):
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
                 'purpose': purpose, 'capabilities': _PURPOSE_CAPABILITIES[purpose], 'scope': scope})
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

    def outstanding(self) -> bool:
        """An unexpired ticket was minted and not yet redeemed: a client is attaching right now."""
        with self._lock:
            now = time.monotonic()
            return any(expires > now for expires, _grant in self._entries.values())

    def revoke(self) -> None:
        with self._lock:
            self._entries.clear()

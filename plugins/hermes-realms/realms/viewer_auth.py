"""Short-lived viewer capabilities, minted only behind Hermes authentication."""

from dataclasses import dataclass
import hashlib
import secrets
import threading
import time


@dataclass(frozen=True)
class Ticket:
    realm_id: str
    generation: str
    expires: float
    can_control: bool


class Tickets:
    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.RLock()
        self._tickets = {}

    def issue(self, realm_id, generation, *, ttl=900, can_control=False):
        if not 0 < ttl <= 3600:
            raise ValueError("Ticket lifetime must be <= one hour")
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = self._clock()
            self._tickets = {k: v for k, v in self._tickets.items() if v.expires > now}
            if len(self._tickets) >= 4096:
                raise ValueError("Too many viewer tickets")
            self._tickets[self._key(token)] = Ticket(
                realm_id, generation, now + ttl, bool(can_control)
            )
        return token

    @staticmethod
    def _key(token):
        return hashlib.sha256(token.encode()).digest()

    def check(self, token, realm_id, generation, *, control=False):
        if not isinstance(token, str) or len(token) > 128:
            return None
        with self._lock:
            ticket = self._tickets.get(self._key(token))
            if ticket is None or ticket.expires <= self._clock():
                return None
            if ticket.realm_id != realm_id or ticket.generation != generation:
                return None
            if control and not ticket.can_control:
                return None
            return ticket

    def revoke(self, realm_id):
        with self._lock:
            self._tickets = {
                k: v for k, v in self._tickets.items() if v.realm_id != realm_id
            }

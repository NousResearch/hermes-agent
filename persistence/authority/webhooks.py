"""Durable webhook notification obligations, not inference or recipient history.

Publication, pre-dial custody, source retirement and payload expiry share the store's
writer. No application callback receives a connection. Legacy v1 keys remain readable.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from .admissions import AdmissionGate


class WebhookLedger(str, Enum):
    FINAL = 'webhook.final.intent.v1.'
    ORIGIN = 'webhook.origin.v1.'


@dataclass(frozen=True, slots=True)
class StoreIncarnation:
    """Recipient directory and physical writer generation captured without creating it."""
    home: str
    directory: tuple[int, int]
    file: tuple[int, int] | None
    instance: str | None

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> StoreIncarnation:
        if set(value) != {'home', 'directory', 'store'} or not isinstance(value['home'], str):
            raise ValueError('invalid target incarnation')
        def identity(raw):
            if not isinstance(raw, (list, tuple)) or len(raw) != 2 or any(type(x) is not int for x in raw) or not raw[1]:
                raise ValueError('invalid file identity')
            return tuple(raw)
        store = value['store']
        if store is not None and (not isinstance(store, Mapping) or set(store) != {'file', 'instance'}
                                  or (store['instance'] is not None and not isinstance(store['instance'], str))):
            raise ValueError('invalid store generation')
        return cls(value['home'], identity(value['directory']), None if store is None else identity(store['file']),
                   None if store is None else store['instance'])

    def to_wire(self) -> dict[str, Any]:
        return dict(home=self.home, directory=list(self.directory), store=None if self.file is None else
                    dict(file=list(self.file), instance=self.instance))


@dataclass(frozen=True, slots=True)
class WebhookRecord:
    ledger: WebhookLedger
    origin_id: str
    revision: int
    value: Mapping[str, Any]


@runtime_checkable
class WebhookDeliveryRepository(Protocol):
    """Bounded reads and fenced v1-ledger transitions on one producer store."""

    def get(self, ledger: WebhookLedger, origin_id: str) -> WebhookRecord | None: ...

    def publish(self, *, epoch: int, record: WebhookRecord, native: Mapping[str, Any] | None = None,
                claim: bool = False, gate: AdmissionGate = None) -> WebhookRecord:
        """CAS on revision (zero for absent/legacy), epoch and owner gate in one write.

        First publication and every pre-dial claim check the live source, linked admission
        and webhook root/session in that snapshot. Retirement refuses unstarted obligations;
        an already-started handoff keeps observer custody, never permission for another dial.
        Compacted payload is immutable. Identity/destination cannot change on transition.
        """
        ...

    def candidates(self, ledger: WebhookLedger, *, after: str = '', limit: int = 8) -> tuple[str, ...]: ...

    def inspection(self, *, after: str = '', routes: tuple[str, ...] | None = None,
                   limit: int = 51) -> tuple[Mapping[str, Any], ...]:
        """Payload-free allowlisted fields, never native envelopes or recipient paths."""
        ...

    def check_owner(self, *, epoch: int, gate: AdmissionGate = None) -> None: ...

    def capture_incarnation(self) -> StoreIncarnation:
        """Read-only generation capture; never creates/migrates/repairs a recipient."""
        ...

    def require_incarnation(self, expected: StoreIncarnation) -> None:
        """Validate the OPEN writer and current path under its lock, not a reopened reader."""
        ...

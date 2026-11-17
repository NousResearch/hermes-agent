"""The suite against a local backend that enforces the owner lock on ``begin_epoch``.

``begin_epoch``'s contract says the caller holds the owner lock (``hermes_state_port``); the
suite must respect its own documented contract, so a backend that enforces the precondition
passes it whole — the positive operation matrix included, which is what holds the lock here
around its own ``begin_epoch`` call.
"""

import weakref

import pytest

from hermes_conformance.storage import AdmissionConformance, FenceConformance, PinningConformance
from hermes_state_port_local import LocalTurnAdmission, _OwnerLock
from tests.conformance.storage.test_local import local_backend


@pytest.fixture
def backend(tmp_path, monkeypatch):
    """The local backend with ``begin_epoch`` refusing any call made without a held owner lock."""
    held = weakref.WeakKeyDictionary()  # admission -> how many of its owner locks are held
    acquire_owner = LocalTurnAdmission.acquire_owner
    release = _OwnerLock.release
    begin_epoch = LocalTurnAdmission.begin_epoch

    def strict_acquire(self, **kwargs):
        lock = acquire_owner(self, **kwargs)
        lock._held_by = self
        held[self] = held.get(self, 0) + 1
        return lock

    def strict_release(self):
        owner = getattr(self, "_held_by", None)
        if owner is not None:
            held[owner] = held.get(owner, 1) - 1
        return release(self)

    def strict_begin_epoch(self, *, instance_id):
        self._pin()  # the pin still refuses first: a foreign pin is not answered by the lock
        assert held.get(self, 0) > 0, "begin_epoch requires the held owner lock"
        return begin_epoch(self, instance_id=instance_id)

    monkeypatch.setattr(LocalTurnAdmission, "acquire_owner", strict_acquire)
    monkeypatch.setattr(_OwnerLock, "release", strict_release)
    monkeypatch.setattr(LocalTurnAdmission, "begin_epoch", strict_begin_epoch)
    wrapped, handles = local_backend(tmp_path, monkeypatch)
    yield wrapped
    for db in handles:
        db.close()


class TestOwnerLockEnforcingLocalFences(FenceConformance):
    pass


class TestOwnerLockEnforcingLocalAdmission(AdmissionConformance):
    pass


class TestOwnerLockEnforcingLocalPinning(PinningConformance):
    pass

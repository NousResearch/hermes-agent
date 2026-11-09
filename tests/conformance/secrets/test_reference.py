"""The secret store conformance suite against a backend that holds every ownership level.

The reference backend (``reference.py``) is an in-memory store keyed on what the bound scope
names, serving one tenant: the fixture's agents ``a``/``b`` share one principal owner, ``c``/``d``
share a group owner. The local backend holds the agent level only, so these resolution invariants
cannot run against it (``test_local.py`` runs the pinning part there).
"""

import pytest

from hermes_conformance.secrets import LeveledSecretConformance, SecretPinningConformance
from tests.conformance.secrets.reference import reference_backend


@pytest.fixture
def backend(monkeypatch):
    from tests.conformance.secrets.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    return reference_backend()


class TestReferencePinning(SecretPinningConformance):
    pass


class TestReferenceLeveled(LeveledSecretConformance):
    pass

"""The view-hook conformance suite against the in-memory reference store (``reference_view.py``)."""

import pytest

from hermes_conformance.secrets import SecretViewConformance, ViewBackend
from tests.conformance.secrets.reference_view import MemoryViewProvider


@pytest.fixture
def view_backend(monkeypatch, tmp_path):
    from tests.conformance.secrets.mutants import apply_mutant
    apply_mutant(monkeypatch)  # only under test_mutants.py, which names a broken backend
    provider = MemoryViewProvider()
    home = tmp_path / "profile"
    home.mkdir()
    provider.boot(home)
    return ViewBackend(provider=provider, home=home,
                       seed_external=lambda env: provider.seed_external(home, env),
                       refresh=lambda: provider.refresh(home),
                       fail_writes=provider.failing_writes)


class TestReferenceView(SecretViewConformance):
    pass

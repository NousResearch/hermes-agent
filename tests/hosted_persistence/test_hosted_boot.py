"""The hosted boot check ``require_hosted_persistence``: it assembles the provider configured for the current home and
returns only for a backend that declares ``scope_pinning`` and whose ``assert_isolating`` passes, closing the probe
backend either way."""

from __future__ import annotations

import pytest

import persistence_assembly
from persistence import IsolationUnsupported
from tests.hosted_persistence.test_provider_discovery import (  # fixtures are used by name
    add_bundled,
    bundled_root,
    entry_points,
    plugin_module,
    plugin_source,
)


def configure_current_home(provider: str) -> None:
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(f"persistence:\n  provider: {provider}\n", encoding="utf-8")


def test_the_built_in_sqlite_is_refused(monkeypatch):
    from persistence.sqlite import SQLiteBackend

    built = []
    monkeypatch.setattr(SQLiteBackend, "__init__", lambda self, *a, **k: built.append(self))
    for configured in (None, "sqlite"):
        if configured:
            configure_current_home(configured)
        with pytest.raises(IsolationUnsupported, match="sqlite"):
            persistence_assembly.require_hosted_persistence()
    assert built == []


@pytest.mark.parametrize("pinning, isolating, refusal", [
    (False, "pass", "scope_pinning"),
    (True, "raise", "role bypasses row security"),
    (True, "absent", "no callable assert_isolating"),
    (True, "not-callable", "no callable assert_isolating"),
])
def test_a_backend_that_cannot_prove_isolation_is_refused_and_closed(bundled_root, entry_points, pinning, isolating,
                                                                    refusal):
    add_bundled(bundled_root, "probe", plugin_source(pinning=pinning, isolating=isolating))
    configure_current_home("probe")
    with pytest.raises(IsolationUnsupported, match=refusal):
        persistence_assembly.require_hosted_persistence()
    module = plugin_module("probe")
    assert len(module.BUILT) == 1 and module.CLOSED == module.BUILT


def test_an_isolating_backend_passes_and_is_closed(bundled_root, entry_points):
    add_bundled(bundled_root, "probe", plugin_source(pinning=True, isolating="pass"))
    configure_current_home("probe")
    assert persistence_assembly.require_hosted_persistence() is None
    module = plugin_module("probe")
    assert len(module.BUILT) == 1 and module.CLOSED == module.BUILT
    assert module.BUILT[0].read_only is False


def test_an_unknown_provider_is_refused_by_name(bundled_root, entry_points):
    configure_current_home("absent")
    with pytest.raises(ValueError, match="unknown persistence provider 'absent'"):
        persistence_assembly.require_hosted_persistence()

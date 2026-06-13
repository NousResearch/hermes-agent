"""Regression test for #45759.

An all-exhausted credential pool holds entries but no *usable* credential.
``list_authenticated_providers`` must not treat such a provider as
authenticated -- otherwise an aggregator whose quota is spent gets matched
during no-provider ``/model`` resolution, wins the model name, and sticks as
the session provider (the "sticky provider fallback pollution" bug).
"""

import pytest


class _FakePool:
    def __init__(self, available: bool):
        self._available = available

    def has_credentials(self) -> bool:
        # The pool still holds entries...
        return True

    def has_available(self) -> bool:
        # ...but none of them are usable when exhausted/dead.
        return self._available


def _patch_opencode_pool(monkeypatch, *, available: bool):
    """Make the opencode-go aggregator look configured but with a pool whose
    only credential is (un)available, depending on ``available``."""
    import hermes_cli.auth as auth
    import agent.credential_pool as cp

    monkeypatch.setattr(
        auth,
        "_load_auth_store",
        lambda: {
            "version": 1,
            "providers": {},
            "active_provider": None,
            "credential_pool": {"opencode-go": {"entries": [{"id": "x"}]}},
        },
    )
    monkeypatch.setattr(
        cp,
        "load_pool",
        lambda provider: _FakePool(available if provider == "opencode-go" else True),
    )


@pytest.fixture(autouse=True)
def _strip_provider_env(monkeypatch):
    """Don't let real provider keys in the environment authenticate providers
    through a different code path than the pool gate under test."""
    import os

    for key in list(os.environ):
        if "OPENCODE" in key or key.endswith("_API_KEY"):
            monkeypatch.delenv(key, raising=False)


def test_exhausted_pool_provider_is_not_authenticated(monkeypatch):
    """The fix: an exhausted pool is NOT authenticated. Fails on main, where
    the gate accepted any stored pool entry regardless of usability."""
    from hermes_cli.model_switch import get_authenticated_provider_slugs

    _patch_opencode_pool(monkeypatch, available=False)
    slugs = get_authenticated_provider_slugs(current_provider="alibaba")
    assert "opencode-go" not in slugs


def test_pool_provider_with_available_credential_is_authenticated(monkeypatch):
    """Control: with a usable credential the provider IS authenticated, proving
    the test drives the credential gate rather than excluding it for some other
    reason."""
    from hermes_cli.model_switch import get_authenticated_provider_slugs

    _patch_opencode_pool(monkeypatch, available=True)
    slugs = get_authenticated_provider_slugs(current_provider="alibaba")
    assert "opencode-go" in slugs

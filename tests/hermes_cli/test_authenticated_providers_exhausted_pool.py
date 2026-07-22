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


def test_opaque_legacy_pool_value_stays_visible(monkeypatch):
    """Legacy token-style auth-store values have no parsed pool entries."""
    from hermes_cli.model_switch import _credential_pool_is_usable

    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        lambda _provider: type(
            "EmptyPool",
            (),
            {
                "has_credentials": lambda self: False,
                "has_available": lambda self: False,
            },
        )(),
    )

    assert _credential_pool_is_usable("opencode-go", raw_pool_present=True)


def test_picker_shows_exhausted_pool_provider(monkeypatch):
    """The interactive picker must include providers whose credential pool
    entries are all exhausted, so the user can still switch to a different
    model under the same provider."""
    from hermes_cli.model_switch import list_picker_providers

    _patch_opencode_pool(monkeypatch, available=False)
    providers = list_picker_providers(
        current_provider="alibaba",
        user_providers={},
        custom_providers=[],
    )
    slugs = [p["slug"] for p in providers]
    assert "opencode-go" in slugs, (
        "Picker must show exhausted-pool providers so the user can select "
        "a different model under the same provider"
    )


class _StopPicker(BaseException):
    """Aborts a picker right after it requests its provider list, before any
    interactive prompt. Subclasses BaseException so the picker's own
    ``except Exception`` guards don't swallow it."""


def _spy_list_authenticated(recorded: dict):
    def _spy(*_args, **kwargs):
        recorded.update(kwargs)
        raise _StopPicker
    return _spy


def test_aux_task_picker_requests_exhausted_pool_visibility(monkeypatch):
    """The ``hermes model`` auxiliary-task picker (``_aux_select_for_task``)
    must request exhausted-pool visibility (``for_picker=True``) like the
    ``/model`` picker (#66584).

    The aux picker writes a *persistent* per-task provider/model config that
    the user runs later — long after a momentary rate-limit cooldown clears —
    so silently hiding a provider whose keys are all exhausted is exactly the
    bug the #66584 picker fix addressed, one interactive picker over.
    """
    import hermes_cli.main as main

    recorded: dict = {}
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        _spy_list_authenticated(recorded),
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})

    with pytest.raises(_StopPicker):
        main._aux_select_for_task("compression")

    assert recorded.get("for_picker") is True, (
        "aux-task picker must pass for_picker=True so exhausted-pool providers "
        "stay selectable (before the fix it omitted the flag → hidden)"
    )


def test_vision_provider_picker_requests_exhausted_pool_visibility(monkeypatch):
    """The vision provider/model picker (``_configure_vision_provider_model``)
    must also request exhausted-pool visibility — same rationale as #66584."""
    import hermes_cli.tools_config as tc

    recorded: dict = {}
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        _spy_list_authenticated(recorded),
    )

    with pytest.raises(_StopPicker):
        tc._configure_vision_provider_model({}, {})

    assert recorded.get("for_picker") is True

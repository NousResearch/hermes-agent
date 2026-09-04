"""Regression tests for switch_model() with ``providers:`` as a whitelist list.

``providers:`` in config.yaml may be either a list of provider slugs
(display whitelist, e.g. ``- deepseek``) or a dict of user-declared provider
blocks. ``switch_model()`` assumed the dict form and called
``user_providers.items()`` unconditionally, so the whitelist-list form crashed
with ``AttributeError: 'list' object has no attribute 'items'`` whenever a
model switch fell into the validation-override path (also ``.get()`` on the
headers lookup just above it).

These live under ``tests/cli`` because they exercise ``hermes_cli.model_switch``
directly as a unit; the gateway's ``/model`` command (and the desktop picker,
which funnels through ``_apply_model_switch``) all reach the same
``switch_model()``.
"""

import pytest

from hermes_cli.model_switch import ModelSwitchResult, switch_model


def _call_switch(user_providers):
    """Mirror the gateway's _apply_model_switch call shape — providers is
    passed straight from config.yaml as-is. api_key is passed explicitly so
    the test is hermetic: the repo's conftest strips real API keys from the
    environment, which would make switch_model() bail out at credential
    resolution and never reach the validation path under test."""
    return switch_model(
        raw_input="deepseek-v4-flash-vision-exp",
        current_provider="deepseek",
        current_model="deepseek-v4-flash",
        current_base_url="https://api.deepseek.com/v1",
        current_api_key="sk-test-key",
        is_global=False,
        explicit_provider="deepseek",
        user_providers=user_providers,
        custom_providers=[],
    )


def _force_validation_failure(monkeypatch, capture=None):
    """Force the ``if not validation.get("accepted")`` branch that used to
    call ``user_providers.items()`` on a list — deterministic, no network.
    switch_model() imports validate_requested_model into its body from
    hermes_cli.models on every call, so patching the source module works.
    When ``capture`` is a dict, the fake records the ``headers`` kwarg it was
    called with so tests can assert the fallback semantics."""
    # The repo conftest strips API keys from the environment; switch_model()
    # re-resolves credentials via env vars regardless of current_api_key, so
    # provide one here or it bails out before reaching the validation path.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")

    def _fake_validate(*args, **kwargs):
        if capture is not None:
            capture["headers"] = kwargs.get("headers")
        return {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": "fake validation failure",
        }

    monkeypatch.setattr("hermes_cli.models.validate_requested_model", _fake_validate)
    return capture


def test_providers_whitelist_list_does_not_crash(monkeypatch):
    """A ``providers: [deepseek, zai]`` whitelist must not raise on the
    validation-override path."""
    _force_validation_failure(monkeypatch)

    result = _call_switch(["deepseek", "zai"])

    assert isinstance(result, ModelSwitchResult)


def test_providers_dict_form_still_works(monkeypatch):
    """The dict form of ``providers:`` must keep working (no regression)."""
    _force_validation_failure(monkeypatch)

    result = _call_switch({})

    assert isinstance(result, ModelSwitchResult)


def test_providers_whitelist_list_passes_no_extra_headers(monkeypatch):
    """A whitelist list must produce no extra validation headers.

    ``user_providers.get()`` is only reached for the dict form, so a list
    falls through to ``None`` — the fallback semantics are "ignore the
    providers field for headers", not "empty dict". Pin that so the guard
    cannot silently degrade into a permissive empty-dict path."""
    capture = {}
    _force_validation_failure(monkeypatch, capture)

    _call_switch(["deepseek", "zai"])

    assert capture["headers"] is None

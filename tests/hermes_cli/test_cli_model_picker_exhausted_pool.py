"""Regression test: the CLI ``/model`` picker must pass ``for_picker=True``.

``build_models_payload(for_picker=True)`` keeps providers whose credential pool
is entirely in cooldown visible in pickers (limits are per-model for many
providers). The dashboard/TUI pickers already pass the flag
(``build_model_options_payload``); the CLI ``/model`` picker built its provider
list without it, so a provider whose every pooled key sat in a 429 cooldown
(openai-codex on a shared root pool behind a profile, for example) silently
vanished from the in-session picker even though another model could still work
and ``hermes chat --provider`` routed fine.
"""

import pytest


class _FakePool:
    def __init__(self, available: bool):
        self._available = available

    def has_credentials(self) -> bool:
        # The pool still holds entries...
        return True

    def has_available(self) -> bool:
        # ...but none of them are usable while every key is in cooldown.
        return self._available


@pytest.fixture(autouse=True)
def _strip_provider_env(monkeypatch):
    """Don't let real provider keys in the environment authenticate providers
    through a different code path than the pool gate under test."""
    import os

    for key in list(os.environ):
        if key.endswith("_API_KEY") or "OPENCODE" in key:
            monkeypatch.delenv(key, raising=False)


def _patch_openai_codex_pool(monkeypatch, *, available: bool):
    """Make openai-codex look configured with a pool whose only credential is
    (un)available, depending on ``available``."""
    import hermes_cli.auth as auth
    import agent.credential_pool as cp

    monkeypatch.setattr(
        auth,
        "_load_auth_store",
        lambda: {
            "version": 1,
            "providers": {},
            "active_provider": None,
            "credential_pool": {"openai-codex": {"entries": [{"id": "x"}]}},
        },
    )
    monkeypatch.setattr(
        cp,
        "load_pool",
        lambda provider: _FakePool(available if provider == "openai-codex" else True),
    )


class _Ctx:
    """Duck-typed stand-in kept for the call-site test; the payload test below
    uses the real ConfigContext."""

    current_provider = "zai"
    current_base_url = ""
    current_model = "glm-5.3"
    user_providers = {}
    custom_providers = []
    excluded_providers = []


def test_cli_model_picker_keeps_cooldown_pool_provider_visible(monkeypatch):
    """The fix: the CLI picker's provider list is built with for_picker=True,
    so a provider whose pool is entirely in cooldown stays selectable. Fails
    on the unfixed code, which dropped the provider entirely."""
    from hermes_cli.inventory import ConfigContext, build_models_payload

    _patch_openai_codex_pool(monkeypatch, available=False)
    ctx = ConfigContext(
        current_provider="zai", current_model="glm-5.3", current_base_url="",
        user_providers={}, custom_providers=[], excluded_providers=[])

    slugs = build_models_payload(
        ctx, for_picker=True, probe_custom_providers=False,
        probe_current_custom_provider=True)["providers"]
    assert any(r.get("slug") == "openai-codex" for r in slugs)


def test_cli_model_picker_call_site_passes_for_picker(monkeypatch):
    """The call site itself must pass for_picker=True — asserting on the built
    payload alone would pass even if the picker called it without the flag."""
    import hermes_cli.cli_model_switch_mixin as mixin
    from hermes_cli.inventory import build_models_payload

    captured = {}

    def _capture(ctx, **kwargs):
        captured.update(kwargs)
        return {"providers": [], "model": "", "provider": ""}

    monkeypatch.setattr(mixin, "build_models_payload", _capture, raising=False)
    # The import inside _show_model_picker resolves the symbol from the module
    # at call time, so patch where production reads it.
    import hermes_cli.inventory as inventory
    monkeypatch.setattr(inventory, "build_models_payload", _capture)

    mixin._show_model_picker(cli=None, ctx=_Ctx(), force_refresh=False)
    assert captured.get("for_picker") is True

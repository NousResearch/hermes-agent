"""An admitted plugin provider is selectable and status-accurate on every surface (#116408).

Residue after picker admission: `hermes model` had no flow for a plugin profile (selection was a silent
no-op), OAuth-shaped plugins had no ``_STATUS_BY_AUTH_TYPE`` builder (``authenticated`` stayed False with
a live pool entry) and external-process sign-in evidence existed for one bundled vendor only, so the
Desktop ``explicit_only`` picker hid out-of-tree ACP rows.
"""

import os
import stat

import pytest

from providers.base import ProviderProfile


@pytest.fixture
def plugin(monkeypatch):
    """Register *profile* the way plugin discovery does (providers registry + auth mirror), no leaks."""
    import providers
    from hermes_cli import auth
    from hermes_cli.auth_plugin_providers import PLUGIN_MIRRORED_PROVIDERS, register_plugin_provider

    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None, raising=False)
    monkeypatch.setattr(auth, "PROVIDER_REGISTRY", dict(auth.PROVIDER_REGISTRY))
    mirrored = set(PLUGIN_MIRRORED_PROVIDERS)
    monkeypatch.setattr("hermes_cli.auth_plugin_providers.PLUGIN_MIRRORED_PROVIDERS", mirrored)

    def _register(profile: ProviderProfile) -> ProviderProfile:
        providers.register_provider(profile)
        register_plugin_provider(profile)
        return profile

    return _register


def _fake_binary(tmp_path, monkeypatch, name: str) -> None:
    exe = tmp_path / name
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")


def _pool_entry(provider: str, **fields):
    from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool

    entry = PooledCredential(
        provider=provider, id="e1", label="acme", auth_type=AUTH_TYPE_OAUTH, priority=0,
        source="manual:example_device", base_url="https://example.invalid/v1", **fields)
    load_pool(provider).add_entry(entry)
    return entry


def test_hermes_model_routes_registered_plugin_profiles_to_the_generic_flow(plugin, monkeypatch, tmp_path):
    """Selecting an admitted external-process or OAuth plugin in `hermes model` persists config.model;
    an api_key profile still takes the api-key flow and an unknown slug stays a no-op."""
    import hermes_cli.main as main
    from hermes_cli import auth
    from hermes_cli.config import load_config

    plugin(ProviderProfile(name="example-acp", auth_type="external_process", base_url="acp://example",
                           process_command="example-acp-bin", fallback_models=("acp-a", "acp-b")))
    plugin(ProviderProfile(name="example-oauth", auth_type="oauth_external",
                           base_url="https://example.invalid/v1", fallback_models=("oauth-a",)))
    plugin(ProviderProfile(name="example-key", auth_type="api_key", env_vars=("EXAMPLE_API_KEY",),
                           base_url="https://example.invalid/v1", fallback_models=("key-a",)))
    _fake_binary(tmp_path, monkeypatch, "example-acp-bin")
    _pool_entry("example-oauth", access_token="tok-1", refresh_token="rt-1")
    monkeypatch.setattr(main, "_offer_reasoning_after_pick", lambda *a, **k: None)
    monkeypatch.setattr(auth, "_prompt_model_selection", lambda models, **k: models[-1])
    api_key_flow_calls = []
    monkeypatch.setattr(main, "_model_flow_api_key_provider",
                        lambda config, pid, current="": api_key_flow_calls.append(pid))

    for slug, expected in (("example-acp", ("acp-b", "acp://example")),
                           ("example-oauth", ("oauth-a", "https://example.invalid/v1"))):
        monkeypatch.setattr(main, "_pick_provider", lambda *a, _s=slug, **k: _s)
        main.select_provider_and_model(args=None)
        model = load_config()["model"]
        assert (model["provider"], model["default"], model["base_url"]) == (slug, *expected)

    monkeypatch.setattr(main, "_pick_provider", lambda *a, **k: "example-key")
    main.select_provider_and_model(args=None)
    assert api_key_flow_calls == ["example-key"]
    monkeypatch.setattr(main, "_pick_provider", lambda *a, **k: "no-such-provider")
    main.select_provider_and_model(args=None)
    assert load_config()["model"]["provider"] == "example-oauth"


def test_oauth_plugin_status_follows_the_credential_pool(plugin):
    """``get_auth_status``/``authenticated`` for an OAuth-shaped plugin read the pool: live token →
    logged in; expired token with refresh material → needs_refresh; nothing → signed out with the
    `hermes auth add` hint. Bundled OAuth providers never reach this builder."""
    from hermes_cli import auth, models

    plugin(ProviderProfile(name="example-oauth", auth_type="oauth_external",
                           base_url="https://example.invalid/v1", refresh_credential=lambda entry: {}))
    plugin(ProviderProfile(name="example-device", auth_type="oauth_device_code", base_url="https://d.invalid/v1"))

    signed_out = auth.get_auth_status("example-oauth")
    assert signed_out["logged_in"] is False and signed_out["configured"] is True
    assert "hermes auth add example-oauth" in signed_out["hint"]
    assert models._provider_has_credentials("example-oauth") is True  # registered = usable with provider:model

    _pool_entry("example-oauth", access_token="tok-1", refresh_token="rt-1")
    assert auth.get_auth_status("example-oauth")["logged_in"] is True

    _pool_entry("example-device", access_token="tok-old", refresh_token="rt-old", expires_at_ms=1_000)
    expired = auth.get_auth_status("example-device")
    assert (expired["logged_in"], expired["needs_refresh"]) == (False, False)  # no refresh hook → not refreshable
    _pool_entry("example-oauth", access_token="tok-old", refresh_token="rt-old", expires_at_ms=1_000)
    # the live entry still wins for example-oauth
    assert auth.get_auth_status("example-oauth")["logged_in"] is True

    assert auth.get_plugin_oauth_auth_status("openai-codex") == {"logged_in": False}


def test_any_external_process_plugin_counts_as_signed_in_when_its_binary_resolves(plugin, monkeypatch, tmp_path):
    """The Desktop ``explicit_only`` filter (``build_model_options_payload`` behind
    ``tui_gateway/methods_complete.py::model.options``) keeps an out-of-tree ACP row exactly like the
    bundled one: binary resolves → auth evidence; missing binary → hidden."""
    from hermes_cli import auth
    from hermes_cli.inventory import _external_process_signed_in, _filter_explicit_provider_rows

    plugin(ProviderProfile(name="example-acp", auth_type="external_process", base_url="acp://example",
                           process_command="example-acp-bin", fallback_models=("acp-a",)))

    class _Ctx:
        current_provider = "nous"

    rows = [{"slug": "example-acp", "models": ["acp-a"]}]
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    assert _external_process_signed_in("example-acp") is False
    assert _filter_explicit_provider_rows(rows, _Ctx()) == []

    _fake_binary(tmp_path, monkeypatch, "example-acp-bin")
    status = auth.get_external_process_provider_status("example-acp")
    assert status["auth_verified"] is True and status["auth_source"].startswith("command: ")
    assert [r["slug"] for r in _filter_explicit_provider_rows(rows, _Ctx())] == ["example-acp"]
    # Bundled Copilot keeps its token-store evidence: a resolving binary alone is not a sign-in there.
    assert auth._external_process_auth_evidence("copilot-acp", str(tmp_path / "copilot")) == \
        auth._copilot_acp_auth_evidence()

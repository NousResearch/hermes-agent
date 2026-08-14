"""Tests for providers.resolve_provider_profile - requested-provider-first.

Custom-provider entries canonicalize to provider="custom" and carry the
user's entry name in requested_provider. resolve_provider_profile() lets a
profile that sets ``activates_on_requested_provider = True`` (the llamacpp
user plugin) claim such entries by name or alias. Profiles without the
opt-in never shadow the provider lookup, so stock resolution is unchanged
with the plugin absent - and for every other provider even with it present.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tests.providers.test_plugin_discovery import _clear_provider_caches

OPTIN_STUB_INIT = (
    "from providers import register_provider\n"
    "from providers.base import ProviderProfile\n"
    "\n"
    "\n"
    "class LlamaCppProfile(ProviderProfile):\n"
    "    activates_on_requested_provider = True\n"
    "\n"
    "\n"
    "llamacpp = LlamaCppProfile(\n"
    '    name="llamacpp",\n'
    '    aliases=("llamacpp", "llama-swap"),\n'
    "    env_vars=(),\n"
    '    base_url="",\n'
    '    auth_type="api_key",\n'
    ")\n"
    "register_provider(llamacpp)\n"
)

STUB_MANIFEST = (
    "name: llamacpp-provider\n"
    "kind: model-provider\n"
    "version: 0.0.1\n"
    "description: Test stub\n"
)


def _hermes_home_with_stub(tmp_path, monkeypatch, *, install_stub: bool) -> Path:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    if install_stub:
        plugin_dir = hermes_home / "plugins" / "model-providers" / "llamacpp"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "__init__.py").write_text(OPTIN_STUB_INIT)
        (plugin_dir / "plugin.yaml").write_text(STUB_MANIFEST)
    return hermes_home


def test_requested_entry_name_activates_opted_in_profile(
    tmp_path, monkeypatch, caplog
):
    """provider=custom + requested=llamacpp -> llamacpp profile, debug-logged."""
    _hermes_home_with_stub(tmp_path, monkeypatch, install_stub=True)
    _clear_provider_caches()
    from providers import resolve_provider_profile

    with caplog.at_level(logging.DEBUG, logger="providers"):
        profile = resolve_provider_profile("custom", requested="llamacpp")
    assert profile is not None and profile.name == "llamacpp"
    assert any(
        "requested provider" in rec.getMessage() for rec in caplog.records
    ), "activation must be debug-log observable"
    _clear_provider_caches()


def test_requested_alias_activates_opted_in_profile(tmp_path, monkeypatch):
    """The llama-swap alias activates the profile the same way."""
    _hermes_home_with_stub(tmp_path, monkeypatch, install_stub=True)
    _clear_provider_caches()
    from providers import resolve_provider_profile

    profile = resolve_provider_profile("custom", requested="llama-swap")
    assert profile is not None and profile.name == "llamacpp"
    _clear_provider_caches()


def test_unknown_requested_name_falls_back_to_provider(tmp_path, monkeypatch):
    """A differently-named entry resolves the stock custom profile."""
    _hermes_home_with_stub(tmp_path, monkeypatch, install_stub=True)
    _clear_provider_caches()
    from providers import resolve_provider_profile

    profile = resolve_provider_profile("custom", requested="rigproxy")
    assert profile is not None and profile.name == "custom"
    _clear_provider_caches()


def test_non_opt_in_profile_never_shadows_provider(tmp_path, monkeypatch):
    """An entry named after an existing provider (deepseek) stays custom.

    deepseek has a registered profile, but it does not opt into
    requested-name activation - resolution must behave exactly as today.
    """
    _hermes_home_with_stub(tmp_path, monkeypatch, install_stub=True)
    _clear_provider_caches()
    from providers import resolve_provider_profile

    profile = resolve_provider_profile("custom", requested="deepseek")
    assert profile is not None and profile.name == "custom"
    _clear_provider_caches()


def test_plugin_absent_requested_llamacpp_resolves_custom(tmp_path, monkeypatch):
    """Without the plugin dir every entry resolves exactly as today."""
    _hermes_home_with_stub(tmp_path, monkeypatch, install_stub=False)
    _clear_provider_caches()
    from providers import resolve_provider_profile

    for requested in ("llamacpp", "llama-swap", "rigproxy", None):
        profile = resolve_provider_profile("custom", requested=requested)
        assert profile is not None and profile.name == "custom", (
            f"requested={requested!r} must resolve custom with no plugin"
        )
    _clear_provider_caches()


def test_no_requested_matches_plain_lookup(tmp_path, monkeypatch):
    """resolve_provider_profile(p) == get_provider_profile(p) for plain names."""
    _hermes_home_with_stub(tmp_path, monkeypatch, install_stub=True)
    _clear_provider_caches()
    from providers import get_provider_profile, resolve_provider_profile

    for name in ("custom", "openrouter", "deepseek", "does-not-exist", "", None):
        assert resolve_provider_profile(name) is get_provider_profile(
            str(name or "").strip().lower()
        )
    _clear_provider_caches()

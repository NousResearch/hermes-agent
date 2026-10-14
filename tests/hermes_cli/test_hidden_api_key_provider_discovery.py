"""A hidden api_key plugin profile with its key set stays off /model and env-key auto-detection.

``hidden`` profiles are meant to stay off discovery surfaces until the user opts in
(``ProviderProfile.listed``). ``listed_canonical_providers`` honoured that, but the /model picker's
canonical section and ``resolve_provider``'s env-key tier treated any set key as credentials, so a
hidden api_key profile surfaced and could be auto-selected. A set env key is not an opt-in: a plugin
may declare a key only so the subprocess env scrub covers it.
"""

import pytest

from providers.base import ProviderProfile

# Realistic values: has_usable_secret rejects short placeholders.
HIDDEN_KEY = "sk-example-hidden-0123456789abcdef"
SHOWN_KEY = "sk-example-shown-0123456789abcdef"


@pytest.fixture
def plugin(monkeypatch, tmp_path):
    """Register a profile the way plugin discovery does (registry + auth mirror + picker), no leaks."""
    import providers
    import hermes_cli.models  # bind first so monkeypatch restores its originals
    from hermes_cli import auth, models_catalog_static
    from hermes_cli.auth_plugin_providers import PLUGIN_MIRRORED_PROVIDERS, register_plugin_provider

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None, raising=False)
    monkeypatch.setattr(auth, "PROVIDER_REGISTRY", dict(auth.PROVIDER_REGISTRY))
    monkeypatch.setattr("hermes_cli.auth_plugin_providers.PLUGIN_MIRRORED_PROVIDERS", set(PLUGIN_MIRRORED_PROVIDERS))
    monkeypatch.setattr(models_catalog_static, "CANONICAL_PROVIDERS", list(models_catalog_static.CANONICAL_PROVIDERS))
    monkeypatch.setattr(models_catalog_static, "_canonical_slugs", set(models_catalog_static._canonical_slugs))
    monkeypatch.setattr(models_catalog_static, "_PROVIDER_LABELS", dict(models_catalog_static._PROVIDER_LABELS))
    # The picker reads the hermes_cli.models binding, which a suite has usually imported already:
    # point it at the same scoped list so registrations here reach it and never leak out.
    monkeypatch.setattr("hermes_cli.models.CANONICAL_PROVIDERS", models_catalog_static.CANONICAL_PROVIDERS)
    monkeypatch.setattr("hermes_cli.models._PROVIDER_LABELS", models_catalog_static._PROVIDER_LABELS)

    def _register(profile: ProviderProfile) -> ProviderProfile:
        providers.register_provider(profile)
        register_plugin_provider(profile)
        models_catalog_static.sync_plugin_provider_catalog()
        return profile

    return _register


def _profile(name: str, env: str, *, hidden: bool) -> ProviderProfile:
    return ProviderProfile(name=name, auth_type="api_key", env_vars=(env,), hidden=hidden,
                           base_url="https://example.invalid/v1", fallback_models=("m-1",))


def _picker_slugs(current: str = "") -> set[str]:
    from hermes_cli.model_switch_providers import list_authenticated_providers
    return {r["slug"] for r in list_authenticated_providers(current_provider=current)}


def test_hidden_api_key_profile_with_key_set_is_not_a_picker_row(plugin, monkeypatch):
    plugin(_profile("example-hidden", "EXAMPLE_HIDDEN_KEY", hidden=True))
    plugin(_profile("example-shown", "EXAMPLE_SHOWN_KEY", hidden=False))
    monkeypatch.setenv("EXAMPLE_HIDDEN_KEY", HIDDEN_KEY)
    monkeypatch.setenv("EXAMPLE_SHOWN_KEY", SHOWN_KEY)

    slugs = _picker_slugs()
    assert "example-shown" in slugs
    assert "example-hidden" not in slugs
    # The current provider always keeps its row, hidden or not: the user selected it by name.
    assert "example-hidden" in _picker_slugs(current="example-hidden")


def test_hidden_api_key_profile_is_never_env_auto_detected(plugin, monkeypatch):
    from hermes_cli import auth

    plugin(_profile("example-hidden", "EXAMPLE_HIDDEN_KEY", hidden=True))
    monkeypatch.setenv("EXAMPLE_HIDDEN_KEY", HIDDEN_KEY)
    def reader(name: str) -> str:
        return auth.os.environ.get(name, "")

    assert auth._env_key_auto_detected(reader, None) != "example-hidden"

    plugin(_profile("example-shown", "EXAMPLE_SHOWN_KEY", hidden=False))
    monkeypatch.setenv("EXAMPLE_SHOWN_KEY", SHOWN_KEY)
    assert auth._env_key_auto_detected(reader, None) is not None
    # Typed paths never consult the gate: naming it still resolves.
    assert auth.resolve_provider("example-hidden") == "example-hidden"


def test_hidden_profile_key_is_still_scrubbed_from_subprocess_env(plugin):
    """The reason a plugin declares a hidden key profile: the env scrub reads every registered key."""
    from tools.environments import local_env_policy

    plugin(_profile("example-hidden", "EXAMPLE_HIDDEN_KEY", hidden=True))
    assert "EXAMPLE_HIDDEN_KEY" in local_env_policy._build_provider_env_blocklist()


def test_opted_in_hidden_profile_is_listed_again(plugin, monkeypatch, tmp_path):
    """`hermes auth add` (a credential-pool row) is the opt-in: the profile is listed from then on."""
    import json

    plugin(_profile("example-hidden", "EXAMPLE_HIDDEN_KEY", hidden=True))
    monkeypatch.setenv("EXAMPLE_HIDDEN_KEY", HIDDEN_KEY)
    assert "example-hidden" not in _picker_slugs()
    row = {"id": "abc123", "label": "example-hidden", "auth_type": "api_key", "priority": 0,
           "source": "manual", "access_token": HIDDEN_KEY}
    (tmp_path / "hermes" / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"example-hidden": [row]}}))
    assert "example-hidden" in _picker_slugs()


def test_hidden_profile_key_alone_does_not_count_as_a_configured_provider(plugin, monkeypatch):
    """First-run setup must still run when the only key belongs to a hidden profile: resolve_provider
    would not pick it, so skipping setup would leave the user with no provider."""
    import os

    from hermes_cli import main as hmain
    from hermes_cli.setup_migration import _model_section_has_credentials

    for key in list(os.environ):
        if key.endswith(("_API_KEY", "_TOKEN", "_BASE_URL")):
            monkeypatch.delenv(key, raising=False)
    plugin(_profile("example-hidden", "EXAMPLE_HIDDEN_KEY", hidden=True))
    monkeypatch.setenv("EXAMPLE_HIDDEN_KEY", HIDDEN_KEY)
    assert hmain._has_any_provider_configured(strict_profile_scope=False) is False
    assert _model_section_has_credentials({}) is False

    plugin(_profile("example-shown", "EXAMPLE_SHOWN_KEY", hidden=False))
    monkeypatch.setenv("EXAMPLE_SHOWN_KEY", SHOWN_KEY)
    assert hmain._has_any_provider_configured(strict_profile_scope=False) is True
    assert _model_section_has_credentials({}) is True

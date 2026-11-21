"""A plugin profile's mirrored ``hermes auth`` registry row: shape, place, and env auto-detect.

A provider that leaves core keeps its registry row by declaring it on its profile
(``registry_auth_type``, ``base_url_env_var``, ``registry_after``), and the alias keys the mirror
adds for it never count as separate providers.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from providers.base import ProviderProfile


@pytest.fixture
def plugin(monkeypatch):
    """Register a profile the way discovery does (providers layer + auth mirror) on registry copies."""
    import providers
    from hermes_cli import auth

    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None, raising=False)
    monkeypatch.setattr("hermes_cli.auth_plugin_providers.PLUGIN_MIRRORED_PROVIDERS", set())
    # Restored in place: modules hold the dict itself (``from hermes_cli.auth import PROVIDER_REGISTRY``).
    saved = dict(auth.PROVIDER_REGISTRY)

    def _register(profile: ProviderProfile) -> ProviderProfile:
        from hermes_cli.auth_plugin_providers import register_plugin_provider

        providers.register_provider(profile)
        register_plugin_provider(profile)
        return profile

    yield _register
    auth.PROVIDER_REGISTRY.clear()
    auth.PROVIDER_REGISTRY.update(saved)


def _token_profile(name: str, *, ambient: bool) -> ProviderProfile:
    return ProviderProfile(
        name=name, aliases=(f"{name}-alias", f"{name}-other"), env_vars=(f"{name.upper()}_TOKEN",),
        base_url=f"https://{name}.example/v1", auth_type=f"{name}_flow", registry_auth_type="api_key",
        ambient_credentials=ambient)


def test_ambient_plugin_is_not_env_auto_detected_under_its_aliases(plugin, monkeypatch):
    """An ambient token never auto-selects the provider, not even through the alias keys the mirror
    points at its row; a non-ambient twin of the same shape is detected under its canonical id."""
    from hermes_cli import auth

    plugin(_token_profile("ambientco", ambient=True))
    plugin(_token_profile("keyedco", ambient=False))
    env = {"AMBIENTCO_TOKEN": "tok-ambient-1234567890"}

    assert auth.PROVIDER_REGISTRY["ambientco-alias"].id == "ambientco"
    assert auth._env_key_auto_detected(lambda n: env.get(n, ""), None) is None
    env["KEYEDCO_TOKEN"] = "tok-keyed-1234567890"
    assert auth._env_key_auto_detected(lambda n: env.get(n, ""), None) == "keyedco"


def test_only_a_key_held_for_tools_leaves_an_ambient_profile_unconfigured(plugin, monkeypatch):
    """An ambient profile's own token (``GH_TOKEN``-style, no ``tools`` entry) is explicit configuration
    on every surface. A key that also unlocks tools is not, until the user chooses that provider."""
    from unittest.mock import patch

    from agent.auxiliary_client import get_text_auxiliary_client
    from hermes_cli import auth
    from hermes_cli.config import OPTIONAL_ENV_VARS
    from hermes_cli.inventory import ConfigContext, build_models_payload
    from hermes_cli.main import _has_any_provider_configured

    for name, ambient in (("tokenco", True), ("toolco", True), ("keyedco", False)):
        plugin(replace(_token_profile(name, ambient=ambient), default_aux_model=f"{name}-small"))
    for name in ("TOOLCO_TOKEN", "KEYEDCO_TOKEN"):
        monkeypatch.setitem(OPTIONAL_ENV_VARS, name, {"tools": ["web_search"], "category": "provider"})
    status, borrowed = auth.get_auth_status, {}  # a gh login on the host must not decide the first-run check
    monkeypatch.setattr(auth, "get_auth_status",
                        lambda pid: borrowed.get(pid) or (status(pid) if pid.endswith("co") else {}))
    rows = [{"slug": name, "name": name, "models": [f"{name}-small"], "total_models": 1, "is_current": False,
             "is_user_defined": False, "source": "hermes"} for name in ("tokenco", "toolco", "keyedco")]
    ctx = ConfigContext(current_provider="openrouter", current_model="openai/gpt-5.5", current_base_url="",
                        user_providers={}, custom_providers=[])

    def surfaces(env_var: str) -> tuple:
        """(explicit ids, env pool row counts, first-run check, auxiliary fallback model, picker slugs)."""
        monkeypatch.setenv(env_var, "tok-1234567890abcdef")
        provider = env_var.removesuffix("_TOKEN").lower()
        with patch("hermes_cli.model_switch.list_authenticated_providers", return_value=rows):
            payload = build_models_payload(ctx, explicit_only=True)
        listed = [row["slug"] for row in payload["providers"] if row["slug"].endswith("co")]
        seen = ([name for name in ("tokenco", "toolco", "keyedco") if auth.is_provider_explicitly_configured(name)],
                auth._pool_entry_is_explicit({"source": f"env:{env_var}"}, provider),
                _has_any_provider_configured(), get_text_auxiliary_client()[1], listed)
        monkeypatch.delenv(env_var)
        return seen

    assert surfaces("TOKENCO_TOKEN") == (["tokenco"], True, True, "tokenco-small", ["tokenco"])
    assert surfaces("KEYEDCO_TOKEN") == (["keyedco"], True, True, "keyedco-small", ["keyedco"])
    assert surfaces("TOOLCO_TOKEN") == ([], False, False, None, [])
    borrowed["tokenco"] = {"logged_in": True, "key_source": "vendor_cli"}  # no env var: a vendor CLI's login
    assert (auth.is_provider_explicitly_configured("tokenco"), _has_any_provider_configured()) == (False, True)


def test_profile_declares_its_registry_row_shape_and_place(plugin, monkeypatch, capsys):
    """registry_auth_type / base_url_env_var / registry_after produce the row a built-in would have
    had, and `hermes auth add` takes the api-key path instead of demanding an auth_handler."""
    from hermes_cli import auth
    from hermes_cli.auth_commands import auth_add_command

    plugin(ProviderProfile(
        name="acme", display_name="Acme Cloud", aliases=("acme-alias",), env_vars=("ACME_TOKEN", "ACME_KEY"),
        base_url="https://acme.example/v1", auth_type="acme_flow", registry_auth_type="api_key",
        base_url_env_var="ACME_ENDPOINT", registry_after="lmstudio"))

    row = auth.PROVIDER_REGISTRY["acme"]
    assert (row.id, row.name, row.auth_type, row.inference_base_url) == (
        "acme", "Acme Cloud", "api_key", "https://acme.example/v1")
    assert (row.api_key_env_vars, row.base_url_env_var) == (("ACME_TOKEN", "ACME_KEY"), "ACME_ENDPOINT")
    ids = list(auth.PROVIDER_REGISTRY)
    assert ids[ids.index("lmstudio") + 1] == "acme"

    auth_add_command(SimpleNamespace(provider="acme-alias", api_key="sk-acme-1234567890", label="work",
                                     auth_type="", priority=None))
    assert 'Added acme credential #1: "work"' in capsys.readouterr().out


def test_auth_list_shows_a_plugin_provider_once_not_per_alias(plugin, monkeypatch, capsys):
    from hermes_cli.auth_commands import auth_list_command

    from agent import secret_scope

    plugin(_token_profile("listco", ambient=False))
    token = secret_scope.set_secret_scope({"LISTCO_TOKEN": "tok-list-1234567890"})
    try:
        auth_list_command(SimpleNamespace(provider=""))
    finally:
        secret_scope.reset_secret_scope(token)
    headers = [line for line in capsys.readouterr().out.splitlines() if line.startswith("listco")]
    assert headers == ["listco (1 credentials):"]

"""Plugin-mirrored ``PROVIDER_REGISTRY`` rows are per profile home.

One process serves many profiles (multiplex gateway, Desktop ``serve``). A ``$HERMES_HOME`` plugin
mirrors its profile into the process-wide auth registry, so after profile A served a turn, profile B
(no such plugin) auto-detected that provider from a stray env token and handed the raw token to the
plugin's endpoint through the generic env ladder. A home with no profile for a slug must see the slug
as absent everywhere: registry rows, resolve_provider, env auto-detect, credential lookups, runtime.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

SLUG = "ambient-fake"
TOKEN = "fake-ambient-token-0123456789abcdef"

_PLUGIN = textwrap.dedent(
    f"""
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(
        name="{SLUG}", aliases=("{SLUG}-alias",), display_name="Ambient Fake", auth_type="api_key",
        env_vars=("FAKE_AMBIENT_GH_TOKEN",), base_url="https://ambient-fake.invalid/v1",
        api_mode="chat_completions", registry_after="lmstudio", ambient_credentials=True))
    """
)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    import providers
    from hermes_cli import auth_plugin_providers as app
    from hermes_cli.auth import PROVIDER_REGISTRY

    plugin_home, bare_home = tmp_path / "with_plugin", tmp_path / "bare"
    plugin = plugin_home / "plugins" / "model-providers" / SLUG
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
    bare_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(bare_home))
    monkeypatch.setenv("FAKE_AMBIENT_GH_TOKEN", TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "x" * 35)
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {})
    monkeypatch.setattr(app, "PLUGIN_MIRRORED_PROVIDERS", set(app.PLUGIN_MIRRORED_PROVIDERS))
    rows = list(dict.items(PROVIDER_REGISTRY))
    yield plugin_home, bare_home
    PROVIDER_REGISTRY.clear()
    PROVIDER_REGISTRY.update(rows)
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


def _bound(home: Path, fn):
    token = set_hermes_home_override(home)
    try:
        return fn()
    except Exception as exc:  # the outcome under test may be a refusal
        return exc
    finally:
        reset_hermes_home_override(token)


def _view(home: Path) -> dict:
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials, resolve_provider
    from hermes_cli.auth_plugin_providers import canonical_registry_rows
    from hermes_cli.runtime_provider import resolve_runtime_provider

    def runtime(requested):
        out = resolve_runtime_provider(requested=requested)
        return out["provider"], out.get("base_url"), out.get("api_key") == TOKEN

    import providers

    return {  # a turn starts with a profile lookup; that is what loads the home's plugins
        "profile": _bound(home, lambda: getattr(providers.get_provider_profile(SLUG), "name", None)),
        "rows": _bound(home, lambda: SLUG in {pid for pid, _ in canonical_registry_rows()}),
        "lookup": _bound(home, lambda: (SLUG in PROVIDER_REGISTRY, PROVIDER_REGISTRY.get(f"{SLUG}-alias") is not None)),
        "explicit": _bound(home, lambda: resolve_provider(SLUG)),
        "auto": _bound(home, lambda: resolve_provider("auto")),
        "creds": _bound(home, lambda: resolve_api_key_provider_credentials(SLUG).get("api_key") == TOKEN),
        "runtime_explicit": _bound(home, lambda: runtime(SLUG)),
        "runtime_auto": _bound(home, lambda: runtime(None)),
    }


def _assert_owner(view: dict) -> None:
    assert view["profile"] == SLUG
    assert view["rows"] is True
    assert view["lookup"] == (True, True)
    assert view["explicit"] == SLUG
    assert view["creds"] is True
    assert view["runtime_explicit"] == (SLUG, "https://ambient-fake.invalid/v1", True)
    # ambient credentials never auto-select, even where the plugin lives
    assert view["auto"] == "gemini"
    assert view["runtime_auto"][0] == "gemini"


def _assert_absent(view: dict) -> None:
    from hermes_cli.auth import AuthError

    assert view["profile"] is None
    assert view["rows"] is False
    assert view["lookup"] == (False, False)
    assert isinstance(view["explicit"], AuthError)
    assert isinstance(view["creds"], AuthError)
    assert isinstance(view["runtime_explicit"], Exception)
    assert view["auto"] == "gemini"
    assert view["runtime_auto"][0] == "gemini" and view["runtime_auto"][2] is False


def test_mirrored_row_is_absent_in_a_home_without_the_plugin_a_b_a(homes):
    plugin_home, bare_home = homes

    _assert_owner(_view(plugin_home))
    _assert_absent(_view(bare_home))
    _assert_owner(_view(plugin_home))
    _assert_absent(_view(bare_home))


def test_setup_detection_ignores_another_homes_mirrored_row(homes, monkeypatch):
    from hermes_cli.setup_migration import _model_section_has_credentials

    import providers

    plugin_home, bare_home = homes
    monkeypatch.delenv("GEMINI_API_KEY")
    _bound(plugin_home, lambda: providers.get_provider_profile(SLUG))
    assert _bound(plugin_home, lambda: _model_section_has_credentials({"model": {"provider": SLUG}})) is True
    assert _bound(bare_home, lambda: _model_section_has_credentials({"model": {"provider": SLUG}})) is False


def test_hidden_rows_survive_registry_reordering(homes):
    """A later mirror inserted mid-registry while another home is bound must not drop hidden rows."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.auth_plugin_providers import _insert_row

    import providers

    plugin_home, bare_home = homes
    _bound(plugin_home, lambda: providers.get_provider_profile(SLUG))
    assert _bound(plugin_home, lambda: SLUG in PROVIDER_REGISTRY) is True
    row = dict.__getitem__(PROVIDER_REGISTRY, "gemini")
    _bound(bare_home, lambda: _insert_row(PROVIDER_REGISTRY, row, "lmstudio"))
    assert _bound(plugin_home, lambda: SLUG in PROVIDER_REGISTRY) is True

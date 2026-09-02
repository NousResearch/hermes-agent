"""A provider installed by ``hermes plugins install`` must actually be found.

``hermes plugins install owner/repo`` (and the plugin index behind
``hermes plugins search``) clones into ``$HERMES_HOME/plugins/<name>/`` — flat,
one directory per plugin. Provider discovery only ever scanned
``$HERMES_HOME/plugins/model-providers/<name>/``, and ``PluginManager`` skips
``kind: model-provider`` on purpose ("routed to their own discovery system").

Nothing joined those two halves, so the documented install path half-worked: the
CLI reported success and the provider silently did not exist. These tests pin
the join, and pin that provider discovery still keeps its hands off every other
plugin living in that same directory.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_PROFILE_SOURCE = textwrap.dedent(
    """
    from providers import register_provider
    from providers.base import ProviderProfile

    register_provider(ProviderProfile(
        name="{name}",
        aliases=("{name}-alias",),
        base_url="acp://{name}",
        auth_type="external_process",
    ))
    """
)


def _clear_provider_caches():
    """Force providers/__init__.py to re-discover on the next lookup."""
    import providers as _pkg

    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._PROVIDER_LIST_CACHE = None
    _pkg._discovered = False
    for mod in list(sys.modules):
        if mod.startswith("plugins.model_providers") or mod.startswith(
            "_hermes_user_provider"
        ):
            del sys.modules[mod]


def _write_plugin(directory: Path, *, name: str, kind: str, body: str | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plugin.yaml").write_text(
        f"name: {name}\nkind: {kind}\nversion: 1.0.0\ndescription: test fixture\n",
        encoding="utf-8",
    )
    (directory / "__init__.py").write_text(
        body if body is not None else _PROFILE_SOURCE.format(name=name),
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_caches()
    yield tmp_path
    _clear_provider_caches()


def test_a_plugin_installed_flat_is_discovered(hermes_home):
    """The layout `hermes plugins install` actually produces."""
    _write_plugin(
        hermes_home / "plugins" / "installed-acp", name="installed-acp", kind="model-provider"
    )
    from providers import get_provider_profile

    profile = get_provider_profile("installed-acp")
    assert profile is not None
    assert profile.base_url == "acp://installed-acp"
    assert profile.auth_type == "external_process"


def test_its_aliases_resolve_too(hermes_home):
    _write_plugin(
        hermes_home / "plugins" / "installed-acp", name="installed-acp", kind="model-provider"
    )
    from providers import get_provider_profile

    assert get_provider_profile("installed-acp-alias") is not None


def test_the_model_providers_layout_still_works(hermes_home):
    """The pre-existing nested location must not regress."""
    _write_plugin(
        hermes_home / "plugins" / "model-providers" / "nested-acp",
        name="nested-acp",
        kind="model-provider",
    )
    from providers import get_provider_profile

    assert get_provider_profile("nested-acp") is not None


def test_both_layouts_coexist(hermes_home):
    _write_plugin(
        hermes_home / "plugins" / "installed-acp", name="installed-acp", kind="model-provider"
    )
    _write_plugin(
        hermes_home / "plugins" / "model-providers" / "nested-acp",
        name="nested-acp",
        kind="model-provider",
    )
    from providers import get_provider_profile

    assert get_provider_profile("installed-acp") is not None
    assert get_provider_profile("nested-acp") is not None


# ── discovery must not overreach ─────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["standalone", "backend", "platform", "exclusive"])
def test_other_plugin_kinds_are_left_to_the_plugin_manager(hermes_home, kind):
    """Those belong to PluginManager, which owns their lifecycle and consent
    flow. Importing them here would run third-party code behind its back.

    The fixture registers a provider on import, so an unwanted import is
    *visible* — a plugin that merely raised would be swallowed by
    ``_import_plugin_dir``'s except and prove nothing.
    """
    _write_plugin(
        hermes_home / "plugins" / f"other-{kind}",
        name=f"other-{kind}",
        kind=kind,
        body=_PROFILE_SOURCE.format(name=f"other-{kind}"),
    )
    from providers import get_provider_profile, list_providers

    assert get_provider_profile(f"other-{kind}") is None
    assert not [p for p in list_providers() if p.name.startswith("other-")]


def test_a_plugin_with_no_manifest_is_ignored(hermes_home):
    directory = hermes_home / "plugins" / "manifestless"
    directory.mkdir(parents=True)
    # Registers on import, so an unwanted import shows up as a provider.
    (directory / "__init__.py").write_text(
        _PROFILE_SOURCE.format(name="manifestless"), encoding="utf-8"
    )
    from providers import get_provider_profile

    assert get_provider_profile("manifestless") is None


def test_an_unreadable_manifest_does_not_break_discovery(hermes_home):
    """A malformed third-party manifest must not blank the whole registry."""
    directory = hermes_home / "plugins" / "broken-manifest"
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text("kind: [this is: not valid\n", encoding="utf-8")
    (directory / "__init__.py").write_text("", encoding="utf-8")
    from providers import get_provider_profile, list_providers

    assert len(list_providers()) > 20  # the bundled set is still there
    assert get_provider_profile("copilot-acp") is not None


def test_quoted_kind_values_are_accepted(hermes_home):
    """The fallback line scan runs when PyYAML is unavailable; a quoted scalar
    is valid YAML and must parse the same either way."""
    directory = hermes_home / "plugins" / "quoted-acp"
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text(
        'name: quoted-acp\nkind: "model-provider"\nversion: 1.0.0\n', encoding="utf-8"
    )
    (directory / "__init__.py").write_text(
        _PROFILE_SOURCE.format(name="quoted-acp"), encoding="utf-8"
    )
    from providers import get_provider_profile

    assert get_provider_profile("quoted-acp") is not None


def test_discovery_survives_a_missing_hermes_home(tmp_path, monkeypatch):
    """No $HERMES_HOME/plugins/ at all — the bundled set must still load."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "does-not-exist"))
    _clear_provider_caches()
    try:
        from providers import get_provider_profile

        assert get_provider_profile("copilot-acp") is not None
    finally:
        _clear_provider_caches()

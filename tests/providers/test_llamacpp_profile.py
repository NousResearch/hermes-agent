"""Tests for the bundled llamacpp provider plugin.

The llamacpp profile ships bundled at
``plugins/model-providers/llamacpp/``. The bundled ``custom`` profile owns
the ``llamacpp`` alias by default; the llamacpp plugin imports after it
(sorted bundled discovery, last-writer-wins) and self-claims that alias.
These tests verify:

 1. A user plugin named llamacpp claiming the ``llamacpp`` and
    ``llama-swap`` aliases overrides the bundled llamacpp plugin, while
    custom keeps the rest of its alias family
 2. With the bundled plugin dir removed, ``llamacpp`` falls back to the
    custom profile and ``llama-swap`` resolves to nothing (stock behavior)
 3. The bundled plugin source registers the expected name and aliases
    from both placements (bundled tree and user plugin dir)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tests.providers.test_plugin_discovery import _clear_provider_caches

LLAMACPP_STUB_INIT = (
    "from providers import register_provider\n"
    "from providers.base import ProviderProfile\n"
    "\n"
    "llamacpp = ProviderProfile(\n"
    '    name="llamacpp",\n'
    '    aliases=("llamacpp", "llama-swap"),\n'
    "    env_vars=(),\n"
    '    base_url="",\n'
    '    auth_type="api_key",\n'
    ")\n"
    "register_provider(llamacpp)\n"
)

LLAMACPP_STUB_MANIFEST = (
    "name: llamacpp-provider\n"
    "kind: model-provider\n"
    "version: 0.0.1\n"
    "description: Test stub\n"
)


def _fresh_hermes_home(tmp_path, monkeypatch) -> Path:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _bundled_plugin_dir() -> Path:
    """The bundled plugin source dir in this checkout.

    Read-only source lookup; tests that need registration copy the source
    into a sandboxed placement rather than importing from here.
    """
    return (
        Path(__file__).resolve().parents[2]
        / "plugins"
        / "model-providers"
        / "llamacpp"
    )


def test_llamacpp_user_plugin_overrides_bundled_plugin(tmp_path, monkeypatch):
    """A user plugin named llamacpp replaces the bundled llamacpp plugin."""
    hermes_home = _fresh_hermes_home(tmp_path, monkeypatch)
    plugin_dir = hermes_home / "plugins" / "model-providers" / "llamacpp"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(LLAMACPP_STUB_INIT)
    (plugin_dir / "plugin.yaml").write_text(LLAMACPP_STUB_MANIFEST)

    _clear_provider_caches()
    from providers import get_provider_profile

    profile = get_provider_profile("llamacpp")
    assert profile is not None
    assert profile.name == "llamacpp", (
        f"Expected the llamacpp user plugin to own 'llamacpp'; got {profile.name!r}"
    )
    assert get_provider_profile("llama-swap") is profile
    # The stub does not set activates_on_requested_provider; the bundled
    # plugin does. Its absence proves the user copy won the collision.
    assert getattr(profile, "activates_on_requested_provider", False) is False

    # Only the claimed aliases move; custom keeps the rest of its family.
    for untouched in ("llama.cpp", "llama-cpp", "ollama", "vllm"):
        still_custom = get_provider_profile(untouched)
        assert still_custom is not None and still_custom.name == "custom", (
            f"Alias {untouched!r} should still resolve to custom"
        )

    _clear_provider_caches()


def test_llamacpp_absent_falls_back_to_custom(tmp_path, monkeypatch):
    """With the bundled plugin dir removed, stock resolution is restored."""
    _fresh_hermes_home(tmp_path, monkeypatch)

    import providers as _pkg

    stripped_root = tmp_path / "bundled-minus-llamacpp"
    shutil.copytree(
        _pkg._BUNDLED_PLUGINS_DIR,
        stripped_root,
        ignore=shutil.ignore_patterns("llamacpp", "__pycache__"),
    )
    monkeypatch.setattr(_pkg, "_BUNDLED_PLUGINS_DIR", stripped_root)

    _clear_provider_caches()
    from providers import get_provider_profile

    profile = get_provider_profile("llamacpp")
    assert profile is not None and profile.name == "custom", (
        f"Stock 'llamacpp' should resolve to custom; got {profile and profile.name!r}"
    )
    assert get_provider_profile("llama-swap") is None

    _clear_provider_caches()


def test_bundled_plugin_source_registers(tmp_path, monkeypatch):
    """The bundled plugin source registers the expected identity."""
    real_dir = _bundled_plugin_dir()
    hermes_home = _fresh_hermes_home(tmp_path, monkeypatch)
    plugin_dir = hermes_home / "plugins" / "model-providers" / "llamacpp"
    plugin_dir.parent.mkdir(parents=True)
    shutil.copytree(
        real_dir, plugin_dir, ignore=shutil.ignore_patterns(".git", "__pycache__")
    )

    _clear_provider_caches()
    from providers import get_provider_profile

    profile = get_provider_profile("llamacpp")
    assert profile is not None
    assert profile.name == "llamacpp"
    assert "llamacpp" in profile.aliases
    assert "llama-swap" in profile.aliases
    assert get_provider_profile("llama-swap") is profile
    assert getattr(profile, "activates_on_requested_provider", False) is True

    _clear_provider_caches()


def test_bundled_placement_registers_identical_profile(tmp_path, monkeypatch):
    """Placement portability: the plugin source registers the identical
    profile from a user plugin dir and from a bundled model-providers
    tree. Only the import module name may differ between placements."""
    import dataclasses

    real_dir = _bundled_plugin_dir()

    def _snapshot():
        from providers import get_provider_profile

        p = get_provider_profile("llamacpp")
        assert p is not None
        return (
            type(p).__name__,
            dataclasses.asdict(p),
            getattr(p, "activates_on_requested_provider", False),
        )

    # Placement 1: user plugin dir under a sandboxed HERMES_HOME.
    hermes_home = _fresh_hermes_home(tmp_path, monkeypatch)
    user_dir = hermes_home / "plugins" / "model-providers" / "llamacpp"
    user_dir.parent.mkdir(parents=True)
    shutil.copytree(
        real_dir, user_dir, ignore=shutil.ignore_patterns(".git", "__pycache__")
    )
    _clear_provider_caches()
    user_snapshot = _snapshot()

    # Placement 2: same source as a bundled plugin, user copy removed.
    shutil.rmtree(user_dir)
    bundled_root = tmp_path / "bundled-tree" / "model-providers"
    bundled_root.mkdir(parents=True)
    shutil.copytree(
        real_dir,
        bundled_root / "llamacpp",
        ignore=shutil.ignore_patterns(".git", "__pycache__"),
    )
    import providers as _pkg

    monkeypatch.setattr(_pkg, "_BUNDLED_PLUGINS_DIR", bundled_root)
    _clear_provider_caches()
    bundled_snapshot = _snapshot()

    assert bundled_snapshot == user_snapshot

    _clear_provider_caches()

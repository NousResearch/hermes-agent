"""Tests for the llamacpp user provider plugin.

The llamacpp profile ships as a user plugin at
``$HERMES_HOME/plugins/model-providers/llamacpp/``. The bundled ``custom``
profile owns the ``llamacpp`` alias by default; the plugin self-claims that
alias so user-plugin discovery (which runs after bundled discovery,
last-writer-wins) repoints it. These tests verify:

 1. A user plugin named llamacpp claiming the ``llamacpp`` and ``llama-swap``
    aliases overrides the bundled custom profile for those lookups only
 2. Without the plugin dir, ``llamacpp`` falls back to the custom profile
    and ``llama-swap`` resolves to nothing (stock behavior)
 3. The real installed plugin source (when present on this machine)
    registers the expected name and aliases
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

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


def _installed_plugin_dir() -> Path | None:
    """The real user plugin dir on this machine, if installed.

    tests/conftest.py sandboxes HERMES_HOME to a tempdir before any test
    module is imported, so ``get_hermes_home()`` cannot see the operator's
    home from here. Look at the platform-default location directly - this
    is a read-only source lookup; registration still happens from a copy
    inside the sandboxed tmp HERMES_HOME, never from the real one.
    """
    d = Path.home() / ".hermes" / "plugins" / "model-providers" / "llamacpp"
    return d if (d / "__init__.py").exists() else None


def test_llamacpp_user_plugin_overrides_custom_alias(tmp_path, monkeypatch):
    """A user plugin named llamacpp takes over the alias custom owns."""
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

    # Only the claimed aliases move; custom keeps the rest of its family.
    for untouched in ("llama.cpp", "llama-cpp", "ollama", "vllm"):
        still_custom = get_provider_profile(untouched)
        assert still_custom is not None and still_custom.name == "custom", (
            f"Alias {untouched!r} should still resolve to custom"
        )

    _clear_provider_caches()


def test_llamacpp_absent_falls_back_to_custom(tmp_path, monkeypatch):
    """Without the plugin dir, stock resolution is unchanged."""
    _fresh_hermes_home(tmp_path, monkeypatch)

    _clear_provider_caches()
    from providers import get_provider_profile

    profile = get_provider_profile("llamacpp")
    assert profile is not None and profile.name == "custom", (
        f"Stock 'llamacpp' should resolve to custom; got {profile and profile.name!r}"
    )
    assert get_provider_profile("llama-swap") is None

    _clear_provider_caches()


@pytest.mark.skipif(
    _installed_plugin_dir() is None,
    reason="real llamacpp user plugin not installed on this machine",
)
def test_installed_llamacpp_plugin_source_registers(tmp_path, monkeypatch):
    """The actual installed plugin source registers the expected identity."""
    real_dir = _installed_plugin_dir()
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

    _clear_provider_caches()

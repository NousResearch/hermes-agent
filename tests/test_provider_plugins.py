"""Test provider plugin discovery and loading infrastructure."""

import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest


class TestProviderPluginDiscovery:
    """Test plugins/providers/__init__.py discovery and loading."""

    def test_discover_returns_list(self):
        from plugins.providers import discover_provider_plugins
        result = discover_provider_plugins()
        assert isinstance(result, list)

    def test_get_unknown_provider_returns_none(self):
        from plugins.providers import get_provider_plugin
        assert get_provider_plugin("nonexistent") is None

    def test_discovery_cache_prevents_rescan(self):
        """After first discovery, _discovered flag prevents re-scanning."""
        import plugins.providers as pp

        # Force a fresh discovery
        pp._discovered = False
        pp._provider_cache.clear()

        pp.get_provider_plugin("anything")
        assert pp._discovered is True

        # Mark cache with a sentinel
        pp._provider_cache["_sentinel"] = lambda: None

        # Second call should NOT reset the cache (sentinel survives)
        pp.get_provider_plugin("anything_else")
        assert "_sentinel" in pp._provider_cache

    def test_discover_skips_hidden_and_underscore_dirs(self, tmp_path):
        """Directories starting with _ or . are skipped."""
        import plugins.providers as pp

        # Create dirs that should be skipped
        (tmp_path / ".hidden" / "__init__.py").parent.mkdir(parents=True)
        (tmp_path / ".hidden" / "__init__.py").write_text("resolve = lambda: {}")
        (tmp_path / "_private" / "__init__.py").parent.mkdir(parents=True)
        (tmp_path / "_private" / "__init__.py").write_text("resolve = lambda: {}")

        with patch.object(pp, "_PROVIDERS_DIR", tmp_path):
            results = pp.discover_provider_plugins()
            names = [name for name, _, _ in results]
            assert ".hidden" not in names
            assert "_private" not in names

    def test_discover_skips_dirs_without_init(self, tmp_path):
        """Directories without __init__.py are skipped."""
        import plugins.providers as pp

        (tmp_path / "no_init_dir").mkdir()

        with patch.object(pp, "_PROVIDERS_DIR", tmp_path):
            results = pp.discover_provider_plugins()
            names = [name for name, _, _ in results]
            assert "no_init_dir" not in names


class TestAliasResolution:
    """Test that ALIASES in a provider plugin are resolved correctly."""

    def test_alias_resolves_to_same_function(self, tmp_path):
        """A plugin with ALIASES should be reachable by any alias."""
        import plugins.providers as pp

        # Create a fake provider plugin with aliases
        plugin_dir = tmp_path / "fakeprov"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text(textwrap.dedent("""\
            ALIASES = ["fakealias", "anothername"]
            def resolve(**kwargs):
                return {"provider": "fake"}
        """))

        # Reset cache and discover from our tmp dir
        pp._discovered = False
        pp._provider_cache.clear()
        with patch.object(pp, "_PROVIDERS_DIR", tmp_path):
            pp._discover_all()

            assert "fakeprov" in pp._provider_cache
            assert "fakealias" in pp._provider_cache
            assert "anothername" in pp._provider_cache

            # All should point to the same resolve function
            assert pp._provider_cache["fakeprov"] is pp._provider_cache["fakealias"]
            assert pp._provider_cache["fakeprov"] is pp._provider_cache["anothername"]

        # Cleanup: restore state
        pp._provider_cache.clear()
        pp._discovered = False

    def test_alias_lookup_case_insensitive(self, tmp_path):
        """Alias lookup is case-insensitive."""
        import plugins.providers as pp

        plugin_dir = tmp_path / "caseprov"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text(textwrap.dedent("""\
            ALIASES = ["CaseAlias"]
            def resolve(**kwargs):
                return {"provider": "case"}
        """))

        pp._discovered = False
        pp._provider_cache.clear()

        with patch.object(pp, "_PROVIDERS_DIR", tmp_path):
            pp._discover_all()
            pp._discovered = True

            # Should be stored lowercase
            assert "casealias" in pp._provider_cache

            # get_provider_plugin lowercases the input
            result = pp.get_provider_plugin("CaseAlias")
            assert result is not None

        pp._provider_cache.clear()
        pp._discovered = False

    def test_plugin_without_aliases(self, tmp_path):
        """A plugin without ALIASES attribute works fine (just primary name)."""
        import plugins.providers as pp

        plugin_dir = tmp_path / "noalias"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text(textwrap.dedent("""\
            def resolve(**kwargs):
                return {"provider": "noalias"}
        """))

        pp._discovered = False
        pp._provider_cache.clear()

        with patch.object(pp, "_PROVIDERS_DIR", tmp_path):
            pp._discover_all()
            assert "noalias" in pp._provider_cache

        pp._provider_cache.clear()
        pp._discovered = False


class TestLoadModuleEdgeCases:
    """Test _load_module handles edge cases gracefully."""

    def test_load_module_missing_init(self, tmp_path):
        """_load_module returns None for dir without __init__.py."""
        import plugins.providers as pp

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = pp._load_module(empty_dir)
        assert result is None

    def test_load_module_with_syntax_error(self, tmp_path):
        """_load_module returns None if __init__.py has a syntax error."""
        import plugins.providers as pp

        bad_dir = tmp_path / "badplugin"
        bad_dir.mkdir()
        (bad_dir / "__init__.py").write_text("def resolve(:\n  pass  # syntax error")

        result = pp._load_module(bad_dir)
        assert result is None

    def test_load_module_with_import_error(self, tmp_path):
        """_load_module returns None if __init__.py raises ImportError."""
        import plugins.providers as pp

        bad_dir = tmp_path / "importfail"
        bad_dir.mkdir()
        (bad_dir / "__init__.py").write_text(
            "import nonexistent_module_xyz_12345\ndef resolve(): pass"
        )

        result = pp._load_module(bad_dir)
        assert result is None

    def test_load_module_cached(self, tmp_path):
        """_load_module returns cached module on second call."""
        import plugins.providers as pp

        plugin_dir = tmp_path / "cached"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("def resolve(**kw): return {}")

        mod1 = pp._load_module(plugin_dir)
        assert mod1 is not None

        mod2 = pp._load_module(plugin_dir)
        assert mod2 is mod1

        # Cleanup
        module_name = f"plugins.providers.cached"
        sys.modules.pop(module_name, None)

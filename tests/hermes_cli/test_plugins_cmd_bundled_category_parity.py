"""``hermes plugins list`` must skip the same bundled categories the loader skips.

The loader (``plugins_discovery``) excludes a set of bundled top-level categories
because they have their own dedicated discovery and load paths. The CLI listing
(``plugins_cmd._discover_all_plugins``) kept its own copy of that set, and the two
copies drifted: ``cron_providers/`` was in the loader's set but not the CLI's.

A bundled cron provider therefore never reached the loader, but the CLI still
listed it as a toggleable opt-in plugin. The listing offered an enable/disable
surface for code that the loader does not load that way.
"""

from pathlib import Path
from unittest.mock import patch

import pytest


def _make_category_plugin(parent: Path, category: str, name: str, manifest: dict) -> Path:
    """Create a category-namespaced plugin: <parent>/<category>/<name>/plugin.yaml."""
    d = parent / category / name
    d.mkdir(parents=True, exist_ok=True)
    import hermes_yaml as yaml
    (d / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (d / "__init__.py").write_text("def register(ctx): pass\n", encoding="utf-8")
    return d


class TestBundledCategorySkipParity:
    """The CLI listing and the loader must agree on which bundled categories are excluded."""

    @pytest.mark.parametrize(
        "category",
        ["memory", "context_engine", "computer_use", "model-providers", "config_backends", "cron_providers"],
    )
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_bundled_category_is_not_listed_as_a_toggleable_plugin(
        self, mock_user_dir, mock_bundled_dir, tmp_path, category
    ):
        """A bundled plugin under an exclusively-loaded category must not be listed."""
        from hermes_cli.plugins_cmd import _discover_all_plugins

        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        user.mkdir()
        _make_category_plugin(bundled, category, "someprovider", {
            "name": f"{category}-provider", "version": "1.0.0",
        })
        mock_user_dir.return_value = user
        mock_bundled_dir.return_value = bundled

        entries = _discover_all_plugins()
        keys = [e[5] for e in entries]

        assert not any(key.startswith(f"{category}/") for key in keys), (
            f"the bundled {category}/ category leaked into the plugin listing (keys: {keys})"
        )

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_the_cli_skip_set_matches_the_loader_skip_set(
        self, mock_user_dir, mock_bundled_dir, tmp_path
    ):
        """The two skip sets are the same fact written down twice, so they must agree.

        Duplicated constants drift; this asserts the two copies stay in step so the
        next category added to the loader cannot silently leak into the listing.
        """
        from hermes_cli.plugins_cmd import _discover_all_plugins
        from hermes_cli.plugins_discovery import collect_directory_manifests
        from hermes_cli.plugins_manifest import manifest_key

        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        user.mkdir()
        # One plugin inside each exclusively-loaded category, plus one ordinary plugin
        # that both paths must agree to show.
        for category in ("memory", "context_engine", "computer_use", "model-providers",
                         "config_backends", "cron_providers"):
            _make_category_plugin(bundled, category, "someprovider", {
                "name": f"{category}-provider", "version": "1.0.0",
            })
        _make_category_plugin(bundled, "web", "brave", {"name": "web-brave", "version": "1.0.0"})
        mock_user_dir.return_value = user
        mock_bundled_dir.return_value = bundled

        loader_manifests = collect_directory_manifests()
        loader_keys = {manifest_key(m) for m in loader_manifests if m.source == "bundled"}
        cli_keys = {e[5] for e in _discover_all_plugins()}

        assert cli_keys == loader_keys, (
            "the CLI listing and the loader disagree about the bundled plugin set "
            f"(cli-only: {sorted(cli_keys - loader_keys)}, loader-only: {sorted(loader_keys - cli_keys)})"
        )

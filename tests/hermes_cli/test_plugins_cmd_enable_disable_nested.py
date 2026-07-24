"""Tests for nested/alias-normalized enable & disable flows.

Companion to test_plugins_cmd_category_discovery.py. That file covers the
*listing* side of nested category plugins (issue #41066). These tests cover
the *mutation* side: `hermes plugins enable/disable` must resolve a bare name
OR a full path-derived key (e.g. `observability/nemo_relay`) to the canonical
registry key and write THAT — the same string PluginManager gates on — so a
nested bundled plugin can actually be toggled.
"""

import sys  # noqa: F401
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_plugin_dir(parent: Path, name: str, manifest: dict) -> Path:
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    import yaml
    (d / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
    (d / "__init__.py").write_text("def register(ctx): pass\n", encoding="utf-8")
    return d


def _make_category_plugin(parent: Path, category: str, name: str, manifest: dict) -> Path:
    return _make_plugin_dir(parent / category, name, manifest)


@pytest.fixture
def nested_plugin_env(tmp_path):
    """A user-plugins dir containing one nested and one flat plugin, with the
    bundled dir pointed at an empty path. Returns the tmp_path."""
    _make_category_plugin(tmp_path, "observability", "nemo_relay", {
        "name": "nemo_relay", "version": "1.0.0", "description": "relay obs"
    })
    _make_plugin_dir(tmp_path, "disk-cleanup", {
        "name": "disk-cleanup", "version": "1.0.0"
    })
    return tmp_path


# ---------------------------------------------------------------------------
# _resolve_plugin_key
# ---------------------------------------------------------------------------


class TestResolvePluginKey:
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_full_key_resolves_to_itself(self, mock_user, mock_bundled, nested_plugin_env):
        from hermes_cli.plugins_cmd import _resolve_plugin_key
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        assert _resolve_plugin_key("observability/nemo_relay") == "observability/nemo_relay"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_bare_leaf_name_resolves_to_key(self, mock_user, mock_bundled, nested_plugin_env):
        from hermes_cli.plugins_cmd import _resolve_plugin_key
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        # "nemo_relay" (bare) must normalize to the path-derived key.
        assert _resolve_plugin_key("nemo_relay") == "observability/nemo_relay"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_flat_plugin_resolves_to_name(self, mock_user, mock_bundled, nested_plugin_env):
        from hermes_cli.plugins_cmd import _resolve_plugin_key
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        assert _resolve_plugin_key("disk-cleanup") == "disk-cleanup"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_unknown_returns_none(self, mock_user, mock_bundled, nested_plugin_env):
        from hermes_cli.plugins_cmd import _resolve_plugin_key
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        assert _resolve_plugin_key("does-not-exist") is None

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_ambiguous_leaf_name_returns_none(self, mock_user, mock_bundled, tmp_path):
        """Same leaf name under two categories must NOT silently pick one."""
        from hermes_cli.plugins_cmd import _resolve_plugin_key
        _make_category_plugin(tmp_path, "image_gen", "openai", {"name": "image-gen-openai"})
        _make_category_plugin(tmp_path, "model-providers", "openai", {"name": "mp-openai"})
        mock_user.return_value = tmp_path
        mock_bundled.return_value = tmp_path / "nonexistent"
        # Bare "openai" is ambiguous -> None; the full key still resolves.
        assert _resolve_plugin_key("openai") is None
        assert _resolve_plugin_key("image_gen/openai") == "image_gen/openai"

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_directory_basename_resolves_only_when_unique(self, mock_user, mock_bundled, tmp_path):
        from hermes_cli.plugins_cmd import _resolve_plugin_key, _resolve_plugin_key_and_source

        _make_category_plugin(tmp_path, "tools", "clone-name", {"name": "manifest-name"})
        mock_user.return_value = tmp_path
        mock_bundled.return_value = tmp_path / "nonexistent"

        assert _resolve_plugin_key("clone-name") == "tools/clone-name"
        assert _resolve_plugin_key_and_source("clone-name") == ("tools/clone-name", "user")

        _make_category_plugin(tmp_path, "other", "clone-name", {"name": "other-name"})
        assert _resolve_plugin_key("clone-name") is None
        assert _resolve_plugin_key_and_source("clone-name") is None


# ---------------------------------------------------------------------------
# cmd_enable / cmd_disable — write the canonical key
# ---------------------------------------------------------------------------


class TestEnableDisableNested:
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_enable_bare_name_writes_key(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        cmd_enable("nemo_relay", allow_tool_override=False)  # bare name

        saved = mock_save_en.call_args[0][0]
        # The canonical key — NOT the bare name — must be persisted, because
        # that is what PluginManager matches when deciding to load.
        assert "observability/nemo_relay" in saved
        assert "nemo_relay" not in saved or "observability/nemo_relay" in saved

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_enable_full_key_writes_key(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        cmd_enable("observability/nemo_relay", allow_tool_override=False)
        saved = mock_save_en.call_args[0][0]
        assert "observability/nemo_relay" in saved

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_enable_clears_manifest_name_alias_from_disabled(
        self, mock_en, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, tmp_path,
    ):
        """#40190 follow-up: enabling by canonical key must clear a stale
        disable entry recorded under the *manifest name*.

        The web providers ship with a manifest name that differs from the
        key (``web-firecrawl`` vs ``web/firecrawl``). A user who ran
        ``hermes plugins disable web-firecrawl`` gets ``web-firecrawl`` in
        ``plugins.disabled``. Since the loader's disable check matches on
        the manifest name too, ``enable web/firecrawl`` must remove that
        entry or "explicit disable wins" keeps the plugin off.
        """
        from hermes_cli.plugins_cmd import cmd_enable
        _make_category_plugin(tmp_path, "web", "firecrawl", {
            "name": "web-firecrawl", "version": "1.0.0",
            "description": "firecrawl", "kind": "backend",
        })
        mock_user.return_value = tmp_path
        mock_bundled.return_value = tmp_path / "nonexistent"
        # Disabled under the manifest name (neither key nor bare leaf).
        with patch(
            "hermes_cli.plugins_cmd._get_disabled_set",
            return_value={"web-firecrawl"},
        ):
            cmd_enable("web/firecrawl", allow_tool_override=False)

        saved_en = mock_save_en.call_args[0][0]
        saved_dis = mock_save_dis.call_args[0][0]
        assert "web/firecrawl" in saved_en
        assert "web-firecrawl" not in saved_dis  # manifest-name alias cleared

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_disable_bare_name_writes_key_and_clears_alias(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        from hermes_cli.plugins_cmd import cmd_disable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        # Simulate an existing config where the plugin was enabled under the
        # legacy bare name — disabling must clear that too, or the plugin would
        # keep loading (PluginManager accepts the bare name as well).
        mock_en.return_value = {"nemo_relay"}

        cmd_disable("nemo_relay")
        saved_dis = mock_save_dis.call_args[0][0]
        saved_en = mock_save_en.call_args[0][0]
        assert "observability/nemo_relay" in saved_dis
        assert "nemo_relay" not in saved_en  # stale bare alias dropped

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    def test_enable_unknown_plugin_exits(self, mock_user, mock_bundled, nested_plugin_env):
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        with pytest.raises(SystemExit):
            cmd_enable("does-not-exist")

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._resolve_tool_override_grant")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_enable_rejects_ambiguous_directory_basename(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis, mock_grant,
        mock_user, mock_bundled, tmp_path,
    ):
        from hermes_cli.plugins_cmd import cmd_enable

        _make_category_plugin(tmp_path, "tools", "same-dir", {"name": "first"})
        _make_category_plugin(tmp_path, "other", "same-dir", {"name": "second"})
        mock_user.return_value = tmp_path
        mock_bundled.return_value = tmp_path / "nonexistent"

        with pytest.raises(SystemExit):
            cmd_enable("same-dir", allow_tool_override=True)

        mock_save_en.assert_not_called()
        mock_save_dis.assert_not_called()
        mock_grant.assert_not_called()

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_enable_flat_plugin_unchanged(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        """Flat plugins keep writing their bare name (key == name) — no regression."""
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        cmd_enable("disk-cleanup", allow_tool_override=False)
        saved = mock_save_en.call_args[0][0]
        assert "disk-cleanup" in saved


# ---------------------------------------------------------------------------
# cmd_enable — built-in tool override consent (issue #29249)
# ---------------------------------------------------------------------------


class TestEnableToolOverrideConsent:
    """Enabling a non-bundled plugin must surface a consent decision about the
    privileged ``allow_tool_override`` capability, and persist the operator's
    choice under ``plugins.entries.<key>.allow_tool_override``."""

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._set_plugin_entry_flag")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_flag_true_grants_override_without_prompt(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis, mock_set_flag,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        cmd_enable("disk-cleanup", allow_tool_override=True)

        mock_set_flag.assert_called_once_with(
            "disk-cleanup", "allow_tool_override", True
        )

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._set_plugin_entry_flag")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_flag_false_declines_override_without_prompt(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis, mock_set_flag,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        cmd_enable("disk-cleanup", allow_tool_override=False)

        mock_set_flag.assert_called_once_with(
            "disk-cleanup", "allow_tool_override", False
        )

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._set_plugin_entry_flag")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_interactive_yes_grants_override(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis, mock_set_flag,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        with patch("rich.console.Console.input", return_value="y"):
            cmd_enable("disk-cleanup")  # no flag -> prompt

        mock_set_flag.assert_called_once_with(
            "disk-cleanup", "allow_tool_override", True
        )

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._set_plugin_entry_flag")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_interactive_blank_enter_defaults_to_deny(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis, mock_set_flag,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        """A blind Enter must NOT grant a privileged capability."""
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        with patch("rich.console.Console.input", return_value=""):
            cmd_enable("disk-cleanup")

        mock_set_flag.assert_called_once_with(
            "disk-cleanup", "allow_tool_override", False
        )

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._set_plugin_entry_flag")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_interactive_eof_defaults_to_deny(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis, mock_set_flag,
        mock_user, mock_bundled, nested_plugin_env,
    ):
        """Non-interactive stdin (EOFError) must fail closed to deny."""
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        with patch("rich.console.Console.input", side_effect=EOFError):
            cmd_enable("disk-cleanup")

        mock_set_flag.assert_called_once_with(
            "disk-cleanup", "allow_tool_override", False
        )

    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._set_plugin_entry_flag")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_bundled_plugin_never_prompts_or_writes_entry(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis, mock_set_flag,
        mock_user, mock_bundled, tmp_path,
    ):
        """Bundled plugins are trusted — no consent prompt, no entry write."""
        from hermes_cli.plugins_cmd import cmd_enable
        # Bundled dir holds the plugin; user dir is empty.
        _make_plugin_dir(tmp_path / "bundled", "trusted_bundled", {
            "name": "trusted_bundled", "version": "1.0.0",
        })
        mock_user.return_value = tmp_path / "empty"
        mock_bundled.return_value = tmp_path / "bundled"

        # Console.input would raise if called — proving no prompt fired.
        with patch("rich.console.Console.input", side_effect=AssertionError("prompted")):
            cmd_enable("trusted_bundled")

        mock_set_flag.assert_not_called()


class TestCompositeMenuWritesCanonicalKey:
    """#40190 follow-up: the interactive `hermes plugins` menu must persist
    the CANONICAL KEY (``web/firecrawl``), never the bare manifest name
    (``web-firecrawl``), so its disabled-list entries stay aligned with what
    ``cmd_enable`` clears and what PluginManager gates on. Writing the bare
    name is what silently vetoed a bundled backend forever (pi314).
    """

    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_fallback_unchecked_plugin_disables_by_key_not_name(
        self, mock_en, mock_save_en, mock_save_dis,
    ):
        from hermes_cli.plugins_cmd import _run_composite_fallback
        from rich.console import Console

        # key differs from the manifest name, mirroring web/firecrawl.
        plugin_keys = ["web/firecrawl"]
        plugin_labels = ["web-firecrawl — firecrawl [bundled]"]
        plugin_selected = set()  # unchecked → should be disabled

        # First input() toggles nothing (blank Enter confirms immediately),
        # second (category prompt) is skipped with blank Enter.
        with patch("builtins.input", return_value=""):
            _run_composite_fallback(
                plugin_keys, plugin_labels, plugin_selected,
                set(), [], Console(),
            )

        saved_dis = mock_save_dis.call_args[0][0]
        assert "web/firecrawl" in saved_dis      # canonical key persisted
        assert "web-firecrawl" not in saved_dis   # never the bare name

    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_fallback_checked_plugin_enables_by_key_and_clears_aliases(
        self, mock_en, mock_save_en, mock_save_dis,
    ):
        from hermes_cli.plugins_cmd import _run_composite_fallback
        from rich.console import Console

        plugin_keys = ["web/firecrawl"]
        plugin_labels = ["web-firecrawl — firecrawl [bundled]"]
        plugin_selected = {0}  # checked → enabled

        # Pre-existing stale bare-leaf disable should be cleared on enable.
        with patch("builtins.input", return_value=""):
            _run_composite_fallback(
                plugin_keys, plugin_labels, plugin_selected,
                {"firecrawl"}, [], Console(),
            )

        saved_en = mock_save_en.call_args[0][0]
        saved_dis = mock_save_dis.call_args[0][0]
        assert "web/firecrawl" in saved_en
        assert "firecrawl" not in saved_dis  # stale bare-leaf alias cleared

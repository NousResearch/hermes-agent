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
    def test_dashboard_file_actions_resolve_key_to_real_user_directory(
        self,
        mock_user,
        mock_bundled,
        nested_plugin_env,
    ):
        from hermes_cli.plugins_cmd import _user_installed_plugin_dir

        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"

        assert _user_installed_plugin_dir("observability/nemo_relay") == (
            nested_plugin_env / "observability" / "nemo_relay"
        ).resolve()


# ---------------------------------------------------------------------------
# cmd_enable / cmd_disable — write the canonical key
# ---------------------------------------------------------------------------


class TestEnableDisableNested:
    def test_enable_canonical_key_activates_inactive_external_override(
        self,
        tmp_path,
        monkeypatch,
    ):
        from hermes_cli import plugins_cmd
        from hermes_cli.plugin_activation import PluginActivationState

        bundled = tmp_path / "bundled"
        user = tmp_path / "user"
        _make_plugin_dir(
            bundled,
            "shared",
            {"name": "shared", "version": "1.0.0", "kind": "backend"},
        )
        _make_plugin_dir(
            user,
            "shared",
            {"name": "shared", "version": "9.0.0", "kind": "backend"},
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_bundled_plugins_dir",
            lambda: bundled,
        )
        monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user)
        monkeypatch.setattr(plugins_cmd, "_discover_entrypoint_plugins", lambda: [])
        enabled_keys = set()
        disabled_keys = set()
        monkeypatch.setattr(
            "hermes_cli.config.load_plugin_activation_state",
            lambda: PluginActivationState(
                enabled=frozenset(enabled_keys),
                disabled=frozenset(disabled_keys),
            ),
        )
        monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(enabled_keys))
        monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set(disabled_keys))

        saved_enabled = []
        saved_disabled = []
        override_grants = []

        def _save_enabled(value):
            enabled_keys.clear()
            enabled_keys.update(value)
            saved_enabled.append(set(value))

        def _save_disabled(value):
            disabled_keys.clear()
            disabled_keys.update(value)
            saved_disabled.append(set(value))

        monkeypatch.setattr(plugins_cmd, "_save_enabled_set", _save_enabled)
        monkeypatch.setattr(plugins_cmd, "_save_disabled_set", _save_disabled)
        monkeypatch.setattr(
            plugins_cmd,
            "_set_plugin_entry_flag",
            lambda *args: override_grants.append(args),
        )

        assert plugins_cmd._resolve_plugin_key("shared") == "shared"
        plugins_cmd.cmd_enable("shared", allow_tool_override=False)

        assert saved_enabled == [{"shared"}]
        assert saved_disabled == [set()]
        assert override_grants == [("shared", "allow_tool_override", False)]
        winner = next(
            entry
            for entry in plugins_cmd._discover_all_plugins()
            if entry[5] == "shared"
        )
        assert winner[3] == "user"
        assert winner[1] == "9.0.0"

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
    def test_enable_unknown_plugin_exits(self, mock_user, mock_bundled, nested_plugin_env):
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        with pytest.raises(SystemExit):
            cmd_enable("does-not-exist")

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
                plugin_keys, plugin_labels, ["bundled"], ["backend"],
                plugin_selected,
                set(), [], Console(),
            )

        saved_dis = mock_save_dis.call_args[0][0]
        assert "web/firecrawl" in saved_dis      # canonical key persisted
        assert "web-firecrawl" not in saved_dis   # never the bare name

    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_fallback_bundled_default_does_not_persist_external_consent(
        self, mock_en, mock_save_en, mock_save_dis,
    ):
        from hermes_cli.plugins_cmd import _run_composite_fallback
        from rich.console import Console

        # Bundled plugins are selected by default without operator consent.
        # Confirming that default must not create an allow-list entry that a
        # future user/project plugin with the same key could inherit.
        with patch("builtins.input", return_value=""):
            _run_composite_fallback(
                ["web/firecrawl"],
                ["web-firecrawl — firecrawl [bundled]"],
                ["bundled"],
                ["backend"],
                {0},
                set(),
                [],
                Console(),
            )

        mock_save_en.assert_not_called()
        mock_save_dis.assert_not_called()

    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_fallback_persists_only_selected_external_plugins(
        self, mock_en, mock_save_en, mock_save_dis,
    ):
        from hermes_cli.plugins_cmd import _run_composite_fallback
        from rich.console import Console

        with patch("builtins.input", return_value=""):
            _run_composite_fallback(
                ["web/firecrawl", "custom-tools"],
                ["web-firecrawl [bundled]", "custom-tools"],
                ["bundled", "user"],
                ["backend", "standalone"],
                {0, 1},
                set(),
                [],
                Console(),
            )

        mock_save_en.assert_called_once_with({"custom-tools"})
        mock_save_dis.assert_called_once_with(set())

    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_fallback_bundled_standalone_remains_explicit_opt_in(
        self, mock_en, mock_save_en, mock_save_dis,
    ):
        from hermes_cli.plugins_cmd import _run_composite_fallback
        from rich.console import Console

        with patch("builtins.input", return_value=""):
            _run_composite_fallback(
                ["observability/langfuse"],
                ["langfuse [bundled]"],
                ["bundled"],
                ["standalone"],
                {0},
                set(),
                [],
                Console(),
            )

        mock_save_en.assert_called_once_with({"observability/langfuse"})
        mock_save_dis.assert_called_once_with(set())


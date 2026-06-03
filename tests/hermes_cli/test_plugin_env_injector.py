"""Tests for hermes_cli.config plugin env-var injection.

Covers both the bundled-platform path (back-compat with the old
``_inject_platform_plugin_env_vars``) and the new user-installed plugin
path that surfaces ``requires_env`` from ``~/.hermes/plugins/*/plugin.yaml``
in the dashboard Keys/Config tabs and the setup wizard.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hermes_cli import config as cfg
from hermes_cli.config import (
    OPTIONAL_ENV_VARS,
    _inject_env_vars_from_manifest,
    _inject_env_vars_from_plugins_dir,
    _inject_plugin_env_vars,
    _inject_platform_plugin_env_vars,
    _merge_plugin_env_entry,
)


@pytest.fixture
def snapshot_optional_env_vars():
    """Snapshot OPTIONAL_ENV_VARS and restore it after the test.

    The dict is a module-level global; tests must not leak entries into
    sibling tests.
    """
    saved = dict(OPTIONAL_ENV_VARS)
    saved_flag = cfg._plugin_env_vars_injected
    try:
        yield
    finally:
        OPTIONAL_ENV_VARS.clear()
        OPTIONAL_ENV_VARS.update(saved)
        cfg._plugin_env_vars_injected = saved_flag


# ── _merge_plugin_env_entry ──────────────────────────────────────────────────


class TestMergePluginEnvEntry:
    """The per-entry merger applies secret heuristics, hardcoded-wins, etc."""

    def test_basic_entry(self, snapshot_optional_env_vars):
        _merge_plugin_env_entry(
            "PLAIN_VAR",
            {"description": "A plain var", "prompt": "Plain"},
            label="Demo",
            default_category="plugins",
        )
        assert OPTIONAL_ENV_VARS["PLAIN_VAR"]["description"] == "A plain var"
        assert OPTIONAL_ENV_VARS["PLAIN_VAR"]["prompt"] == "Plain"
        assert OPTIONAL_ENV_VARS["PLAIN_VAR"]["password"] is False
        assert OPTIONAL_ENV_VARS["PLAIN_VAR"]["category"] == "plugins"

    def test_token_suffix_heuristic_marks_as_secret(self, snapshot_optional_env_vars):
        _merge_plugin_env_entry(
            "MY_BOT_TOKEN", {}, label="Demo", default_category="plugins",
        )
        assert OPTIONAL_ENV_VARS["MY_BOT_TOKEN"]["password"] is True

    def test_password_false_overrides_suffix_heuristic(self, snapshot_optional_env_vars):
        _merge_plugin_env_entry(
            "MY_TOKEN",
            {"password": False},
            label="Demo",
            default_category="plugins",
        )
        assert OPTIONAL_ENV_VARS["MY_TOKEN"]["password"] is False

    def test_secret_true_marks_as_secret_even_without_suffix(self, snapshot_optional_env_vars):
        _merge_plugin_env_entry(
            "OBSCURE_NAME",
            {"secret": True},
            label="Demo",
            default_category="plugins",
        )
        assert OPTIONAL_ENV_VARS["OBSCURE_NAME"]["password"] is True

    def test_hardcoded_entry_wins(self, snapshot_optional_env_vars):
        OPTIONAL_ENV_VARS["EXISTING"] = {"description": "original"}
        _merge_plugin_env_entry(
            "EXISTING",
            {"description": "new"},
            label="Demo",
            default_category="plugins",
        )
        assert OPTIONAL_ENV_VARS["EXISTING"] == {"description": "original"}

    def test_label_used_for_default_description(self, snapshot_optional_env_vars):
        _merge_plugin_env_entry(
            "NEW_VAR", {}, label="Awesome Plugin", default_category="plugins",
        )
        assert OPTIONAL_ENV_VARS["NEW_VAR"]["description"] == "Awesome Plugin configuration"

    def test_url_passed_through(self, snapshot_optional_env_vars):
        _merge_plugin_env_entry(
            "WITH_URL",
            {"url": "https://example.com/keys"},
            label="Demo",
            default_category="plugins",
        )
        assert OPTIONAL_ENV_VARS["WITH_URL"]["url"] == "https://example.com/keys"

    def test_category_override(self, snapshot_optional_env_vars):
        _merge_plugin_env_entry(
            "CATEGORIZED",
            {"category": "observability"},
            label="Demo",
            default_category="plugins",
        )
        assert OPTIONAL_ENV_VARS["CATEGORIZED"]["category"] == "observability"


# ── _inject_env_vars_from_manifest ───────────────────────────────────────────


class TestInjectEnvVarsFromManifest:
    """Reads one plugin.yaml and merges its requires_env + optional_env."""

    def test_reads_required_and_optional_env(self, tmp_path, snapshot_optional_env_vars):
        plugin_dir = tmp_path / "demo"
        plugin_dir.mkdir()
        manifest_path = plugin_dir / "plugin.yaml"
        manifest_path.write_text(yaml.dump({
            "name": "demo",
            "label": "Demo Plugin",
            "requires_env": [
                {"name": "DEMO_KEY", "description": "required key", "secret": True},
            ],
            "optional_env": ["DEMO_OPTIONAL"],
        }), encoding="utf-8")

        _inject_env_vars_from_manifest(manifest_path, plugin_dir, "plugins")

        assert OPTIONAL_ENV_VARS["DEMO_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["DEMO_KEY"]["description"] == "required key"
        assert "DEMO_OPTIONAL" in OPTIONAL_ENV_VARS
        # bare string falls back to label-based description
        assert OPTIONAL_ENV_VARS["DEMO_OPTIONAL"]["description"] == "Demo Plugin configuration"

    def test_malformed_yaml_is_swallowed(self, tmp_path, snapshot_optional_env_vars):
        plugin_dir = tmp_path / "bad"
        plugin_dir.mkdir()
        manifest_path = plugin_dir / "plugin.yaml"
        manifest_path.write_text(":::not valid yaml [[[", encoding="utf-8")

        # Must not raise — malformed YAML can't break CLI import.
        _inject_env_vars_from_manifest(manifest_path, plugin_dir, "plugins")

    def test_empty_manifest_no_op(self, tmp_path, snapshot_optional_env_vars):
        plugin_dir = tmp_path / "empty"
        plugin_dir.mkdir()
        manifest_path = plugin_dir / "plugin.yaml"
        manifest_path.write_text("", encoding="utf-8")

        before = dict(OPTIONAL_ENV_VARS)
        _inject_env_vars_from_manifest(manifest_path, plugin_dir, "plugins")
        assert OPTIONAL_ENV_VARS == before


# ── kind → UI category mapping ───────────────────────────────────────────────


def _write_manifest(plugin_dir: Path, kind: str | None, env_name: str = "DEMO_VAR") -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    body: dict = {
        "name": plugin_dir.name,
        "label": plugin_dir.name.title(),
        "requires_env": [{"name": env_name, "description": "x"}],
    }
    if kind is not None:
        body["kind"] = kind
    manifest_path = plugin_dir / "plugin.yaml"
    manifest_path.write_text(yaml.dump(body), encoding="utf-8")
    return manifest_path


class TestKindToCategoryMapping:
    """`kind:` field overrides the caller's default_category so dashboard UI renders them."""

    def test_kind_platform_maps_to_messaging(self, tmp_path, snapshot_optional_env_vars):
        manifest_path = _write_manifest(tmp_path / "max-messenger", kind="platform", env_name="MAX_TOKEN_TEST")
        _inject_env_vars_from_manifest(manifest_path, tmp_path / "max-messenger", "plugins")
        assert OPTIONAL_ENV_VARS["MAX_TOKEN_TEST"]["category"] == "messaging"

    def test_kind_tool_maps_to_tool(self, tmp_path, snapshot_optional_env_vars):
        manifest_path = _write_manifest(tmp_path / "demo-tool", kind="tool", env_name="DEMO_TOOL_VAR")
        _inject_env_vars_from_manifest(manifest_path, tmp_path / "demo-tool", "plugins")
        assert OPTIONAL_ENV_VARS["DEMO_TOOL_VAR"]["category"] == "tool"

    def test_kind_mcp_maps_to_tool(self, tmp_path, snapshot_optional_env_vars):
        manifest_path = _write_manifest(tmp_path / "demo-mcp", kind="mcp", env_name="DEMO_MCP_VAR")
        _inject_env_vars_from_manifest(manifest_path, tmp_path / "demo-mcp", "plugins")
        assert OPTIONAL_ENV_VARS["DEMO_MCP_VAR"]["category"] == "tool"

    def test_kind_provider_maps_to_provider(self, tmp_path, snapshot_optional_env_vars):
        manifest_path = _write_manifest(tmp_path / "demo-provider", kind="provider", env_name="DEMO_PROV_VAR")
        _inject_env_vars_from_manifest(manifest_path, tmp_path / "demo-provider", "plugins")
        assert OPTIONAL_ENV_VARS["DEMO_PROV_VAR"]["category"] == "provider"

    def test_kind_model_provider_maps_to_provider(self, tmp_path, snapshot_optional_env_vars):
        manifest_path = _write_manifest(tmp_path / "demo-mprov", kind="model-provider", env_name="DEMO_MPROV_VAR")
        _inject_env_vars_from_manifest(manifest_path, tmp_path / "demo-mprov", "plugins")
        assert OPTIONAL_ENV_VARS["DEMO_MPROV_VAR"]["category"] == "provider"

    def test_unknown_kind_falls_back_to_default_category(self, tmp_path, snapshot_optional_env_vars):
        manifest_path = _write_manifest(tmp_path / "demo-weird", kind="weird-thing", env_name="DEMO_WEIRD_VAR")
        _inject_env_vars_from_manifest(manifest_path, tmp_path / "demo-weird", "plugins")
        assert OPTIONAL_ENV_VARS["DEMO_WEIRD_VAR"]["category"] == "plugins"

    def test_missing_kind_falls_back_to_default_category(self, tmp_path, snapshot_optional_env_vars):
        manifest_path = _write_manifest(tmp_path / "demo-nokind", kind=None, env_name="DEMO_NOKIND_VAR")
        _inject_env_vars_from_manifest(manifest_path, tmp_path / "demo-nokind", "setting")
        assert OPTIONAL_ENV_VARS["DEMO_NOKIND_VAR"]["category"] == "setting"

    def test_explicit_entry_category_wins_over_kind_mapping(self, tmp_path, snapshot_optional_env_vars):
        # Entry-level `category:` overrides everything (kind-derived and default).
        plugin_dir = tmp_path / "demo-explicit"
        plugin_dir.mkdir()
        manifest_path = plugin_dir / "plugin.yaml"
        manifest_path.write_text(yaml.dump({
            "name": "demo-explicit",
            "label": "Demo",
            "kind": "platform",
            "requires_env": [
                {"name": "DEMO_EXPLICIT_VAR", "description": "x", "category": "setting"},
            ],
        }), encoding="utf-8")
        _inject_env_vars_from_manifest(manifest_path, plugin_dir, "plugins")
        assert OPTIONAL_ENV_VARS["DEMO_EXPLICIT_VAR"]["category"] == "setting"


# ── _inject_env_vars_from_plugins_dir ────────────────────────────────────────


class TestInjectEnvVarsFromPluginsDir:
    """Directory walker handles flat + (optional) category layouts."""

    @staticmethod
    def _make_plugin(plugins_root: Path, rel: str, manifest: dict) -> None:
        plugin_dir = plugins_root / rel
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")

    def test_flat_layout(self, tmp_path, snapshot_optional_env_vars):
        plugins_root = tmp_path / "plugins"
        self._make_plugin(plugins_root, "alpha", {
            "name": "alpha",
            "requires_env": ["ALPHA_TOKEN"],
        })
        self._make_plugin(plugins_root, "beta", {
            "name": "beta",
            "requires_env": ["BETA_API_KEY"],
        })

        _inject_env_vars_from_plugins_dir(plugins_root, default_category="user-plugin")

        assert "ALPHA_TOKEN" in OPTIONAL_ENV_VARS
        assert "BETA_API_KEY" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["ALPHA_TOKEN"]["category"] == "user-plugin"
        # Both are password-suffixed → marked secret.
        assert OPTIONAL_ENV_VARS["ALPHA_TOKEN"]["password"] is True

    def test_category_layout_recursed(self, tmp_path, snapshot_optional_env_vars):
        plugins_root = tmp_path / "plugins"
        # category/<plugin>/plugin.yaml — the category dir itself has no manifest.
        self._make_plugin(plugins_root, "observability/langfuse", {
            "name": "langfuse",
            "requires_env": ["LANGFUSE_PUBLIC_KEY"],
        })

        _inject_env_vars_from_plugins_dir(
            plugins_root, default_category="plugins", recurse_categories=True,
        )
        assert "LANGFUSE_PUBLIC_KEY" in OPTIONAL_ENV_VARS

    def test_category_layout_not_recursed_by_default(self, tmp_path, snapshot_optional_env_vars):
        plugins_root = tmp_path / "plugins"
        self._make_plugin(plugins_root, "observability/langfuse", {
            "name": "langfuse",
            "requires_env": ["LANGFUSE_PUBLIC_KEY"],
        })

        _inject_env_vars_from_plugins_dir(plugins_root, default_category="plugins")
        assert "LANGFUSE_PUBLIC_KEY" not in OPTIONAL_ENV_VARS

    def test_missing_root_dir_is_noop(self, tmp_path, snapshot_optional_env_vars):
        before = dict(OPTIONAL_ENV_VARS)
        _inject_env_vars_from_plugins_dir(
            tmp_path / "does-not-exist", default_category="plugins",
        )
        assert OPTIONAL_ENV_VARS == before

    def test_mixed_layout_both_picked_up(self, tmp_path, snapshot_optional_env_vars):
        plugins_root = tmp_path / "plugins"
        # Flat plugin
        self._make_plugin(plugins_root, "flat-one", {
            "name": "flat-one",
            "requires_env": ["FLAT_TOKEN"],
        })
        # Category plugin
        self._make_plugin(plugins_root, "category/nested", {
            "name": "nested",
            "requires_env": ["NESTED_TOKEN"],
        })

        _inject_env_vars_from_plugins_dir(
            plugins_root, default_category="plugins", recurse_categories=True,
        )
        assert "FLAT_TOKEN" in OPTIONAL_ENV_VARS
        assert "NESTED_TOKEN" in OPTIONAL_ENV_VARS


# ── _inject_plugin_env_vars (the top-level entry point) ──────────────────────


class TestInjectPluginEnvVars:
    """End-to-end: bundled platforms + user plugins both feed OPTIONAL_ENV_VARS."""

    def test_user_plugins_picked_up_via_hermes_home(self, tmp_path, snapshot_optional_env_vars):
        # Build a fake ~/.hermes/plugins/ tree
        hermes_home = tmp_path / "hermes"
        plugins_root = hermes_home / "plugins"
        plugin_dir = plugins_root / "max-messenger"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({
            "name": "max-messenger",
            "description": "Max messenger bot adapter",
            "requires_env": [
                {"name": "MAX_BOT_TOKEN", "description": "Bot token", "secret": True},
            ],
        }), encoding="utf-8")

        # Reset injector state so it re-runs against our fake HERMES_HOME.
        cfg._plugin_env_vars_injected = False
        OPTIONAL_ENV_VARS.pop("MAX_BOT_TOKEN", None)

        with patch.dict(os.environ, {"HERMES_HOME": str(hermes_home)}):
            _inject_plugin_env_vars()

        assert "MAX_BOT_TOKEN" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["MAX_BOT_TOKEN"]["password"] is True
        assert OPTIONAL_ENV_VARS["MAX_BOT_TOKEN"]["description"] == "Bot token"

    def test_idempotent(self, snapshot_optional_env_vars):
        cfg._plugin_env_vars_injected = False
        _inject_plugin_env_vars()
        first_snapshot = dict(OPTIONAL_ENV_VARS)
        # Calling again must be a no-op (the flag prevents re-injection).
        _inject_plugin_env_vars()
        assert OPTIONAL_ENV_VARS == first_snapshot

    def test_bundled_platforms_still_surfaced(self, snapshot_optional_env_vars):
        cfg._plugin_env_vars_injected = False
        OPTIONAL_ENV_VARS.pop("TEAMS_CLIENT_ID", None)
        _inject_plugin_env_vars()
        # Regression: the original bundled-platform behavior must keep working.
        assert "TEAMS_CLIENT_ID" in OPTIONAL_ENV_VARS

    def test_legacy_alias_still_callable(self, snapshot_optional_env_vars):
        # External code may have imported `_inject_platform_plugin_env_vars`
        # by name before this refactor.  The alias must still resolve to a
        # callable that does the right thing (idempotent injection).
        cfg._plugin_env_vars_injected = False
        _inject_platform_plugin_env_vars()
        assert "TEAMS_CLIENT_ID" in OPTIONAL_ENV_VARS

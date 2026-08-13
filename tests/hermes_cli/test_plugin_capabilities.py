"""Tests for the plugin capability model + consent flow (#64228).

Covers: declaration parsing, consent grant/persist, update re-consent on
added capabilities, fail-closed behavior on missing/corrupt consent state,
and backward compatibility with the legacy ``allow_*`` gates.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_cli.plugin_capabilities import (
    CAPABILITY_REGISTRY,
    VALID_CAPABILITY_IDS,
    capability_set_hash,
    consent_hash,
    declared_set_changed,
    granted_capabilities,
    parse_declared_capabilities,
    pending_capabilities,
    plugin_capability_granted,
    record_consent,
)


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a tmp dir with an empty config.yaml."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    return tmp_path


def _read_cfg(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}


# ── Registry sanity ──────────────────────────────────────────────────────────


class TestRegistry:
    def test_every_capability_has_legacy_gate(self):
        for spec in CAPABILITY_REGISTRY.values():
            assert spec.legacy_path, spec.id
            assert spec.description

    def test_known_ids(self):
        assert "tools.override" in VALID_CAPABILITY_IDS
        assert "llm.model_override" in VALID_CAPABILITY_IDS


# ── Declaration parsing ──────────────────────────────────────────────────────


class TestDeclarationParsing:
    def test_parses_known_ids(self):
        got = parse_declared_capabilities(["tools.override", "llm.model_override"])
        assert got == ["tools.override", "llm.model_override"]

    def test_drops_unknown_ids(self):
        got = parse_declared_capabilities(["tools.override", "root.everything"])
        assert got == ["tools.override"]

    def test_non_list_ignored(self):
        assert parse_declared_capabilities("tools.override") == []
        assert parse_declared_capabilities({"a": 1}) == []
        assert parse_declared_capabilities(None) == []

    def test_non_string_entries_ignored(self):
        assert parse_declared_capabilities([1, None, "tools.override"]) == [
            "tools.override"
        ]

    def test_dedup_preserves_order(self):
        got = parse_declared_capabilities(
            ["llm.model_override", "tools.override", "llm.model_override"]
        )
        assert got == ["llm.model_override", "tools.override"]

    def test_manifest_field_lands_on_parsed_manifest(self, tmp_path):
        """PluginManifest picks up ``capabilities:`` from plugin.yaml."""
        from hermes_cli.plugins import PluginManager

        plugin_dir = tmp_path / "capplug"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: capplug\nversion: '1.0'\n"
            "capabilities:\n  - tools.override\n  - bogus.capability\n",
            encoding="utf-8",
        )
        mgr = PluginManager()
        manifest = mgr._parse_manifest(
            plugin_dir / "plugin.yaml", plugin_dir, "user", ""
        )
        assert manifest is not None
        assert manifest.capabilities == ["tools.override"]

    def test_manifest_without_capabilities_field(self, tmp_path):
        from hermes_cli.plugins import PluginManager

        plugin_dir = tmp_path / "plainplug"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: plainplug\n", encoding="utf-8"
        )
        mgr = PluginManager()
        manifest = mgr._parse_manifest(
            plugin_dir / "plugin.yaml", plugin_dir, "user", ""
        )
        assert manifest is not None
        assert manifest.capabilities == []


# ── Consent grant + persistence ──────────────────────────────────────────────


class TestConsentPersistence:
    def test_record_consent_persists_grant(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        cfg = _read_cfg(hermes_home)
        entry = cfg["plugins"]["entries"]["capplug"]
        assert entry["granted_capabilities"] == ["tools.override"]
        assert entry["capabilities_consent"]["hash"] == capability_set_hash(
            ["tools.override"]
        )
        assert entry["capabilities_consent"]["granted_at"]
        # Bridge: legacy key mirrored so existing enforcement sites work.
        assert entry["allow_tool_override"] is True

    def test_record_consent_mirrors_nested_legacy_key(self, hermes_home):
        record_consent(
            "capplug", ["llm.model_override"], ["llm.model_override"]
        )
        entry = _read_cfg(hermes_home)["plugins"]["entries"]["capplug"]
        assert entry["llm"]["allow_model_override"] is True

    def test_granted_capabilities_roundtrip(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        assert granted_capabilities("capplug") == frozenset({"tools.override"})

    def test_grant_is_union_with_previous(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        record_consent(
            "capplug",
            ["llm.model_override"],
            ["tools.override", "llm.model_override"],
        )
        assert granted_capabilities("capplug") == frozenset(
            {"tools.override", "llm.model_override"}
        )

    def test_capability_granted_after_consent(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        assert plugin_capability_granted("capplug", "tools.override") is True

    def test_declined_stays_off(self, hermes_home):
        # No record_consent call — nothing granted.
        assert plugin_capability_granted("capplug", "tools.override") is False
        assert pending_capabilities("capplug", ["tools.override"]) == [
            "tools.override"
        ]


# ── Update re-consent ────────────────────────────────────────────────────────


class TestUpdateReconsent:
    def test_added_capability_is_pending(self, hermes_home):
        # v1 declared + granted tools.override.
        record_consent("capplug", ["tools.override"], ["tools.override"])
        # v2 adds llm.model_override.
        declared_v2 = ["tools.override", "llm.model_override"]
        assert pending_capabilities("capplug", declared_v2) == [
            "llm.model_override"
        ]
        assert declared_set_changed("capplug", declared_v2) is True
        # The added capability stays ungranted until re-consent.
        assert plugin_capability_granted("capplug", "llm.model_override") is False
        # The previously granted one keeps working.
        assert plugin_capability_granted("capplug", "tools.override") is True

    def test_unchanged_set_needs_no_reconsent(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        assert pending_capabilities("capplug", ["tools.override"]) == []
        assert declared_set_changed("capplug", ["tools.override"]) is False

    def test_hash_order_insensitive(self):
        a = capability_set_hash(["tools.override", "llm.model_override"])
        b = capability_set_hash(["llm.model_override", "tools.override"])
        assert a == b

    def test_no_consent_record_counts_as_changed(self, hermes_home):
        assert declared_set_changed("capplug", ["tools.override"]) is True

    def test_reconsent_grants_addition(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        declared_v2 = ["tools.override", "llm.model_override"]
        record_consent(
            "capplug", pending_capabilities("capplug", declared_v2), declared_v2
        )
        assert plugin_capability_granted("capplug", "llm.model_override") is True
        assert declared_set_changed("capplug", declared_v2) is False


# ── Fail closed ──────────────────────────────────────────────────────────────


class TestFailClosed:
    def test_missing_config_not_granted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "nonexistent"))
        assert plugin_capability_granted("capplug", "tools.override") is False
        assert granted_capabilities("capplug") == frozenset()

    def test_corrupt_granted_list_not_granted(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    capplug:\n"
            "      granted_capabilities: not-a-list\n",
            encoding="utf-8",
        )
        assert plugin_capability_granted("capplug", "tools.override") is False

    def test_corrupt_entry_types_not_granted(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    capplug: 42\n", encoding="utf-8"
        )
        assert plugin_capability_granted("capplug", "tools.override") is False

    def test_unknown_capability_denied(self, hermes_home):
        record_consent("capplug", ["tools.override"], ["tools.override"])
        assert plugin_capability_granted("capplug", "root.everything") is False

    def test_unknown_ids_in_granted_list_ignored(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    capplug:\n"
            "      granted_capabilities: [root.everything, 42]\n",
            encoding="utf-8",
        )
        assert granted_capabilities("capplug") == frozenset()
        assert plugin_capability_granted("capplug", "tools.override") is False

    def test_corrupt_consent_hash_counts_as_changed(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    capplug:\n"
            "      capabilities_consent: broken\n",
            encoding="utf-8",
        )
        assert consent_hash("capplug") is None
        assert declared_set_changed("capplug", ["tools.override"]) is True


# ── Legacy gate backward compat ──────────────────────────────────────────────


class TestLegacyGateCompat:
    def test_legacy_allow_tool_override_still_works(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    oldplug:\n"
            "      allow_tool_override: true\n",
            encoding="utf-8",
        )
        assert plugin_capability_granted("oldplug", "tools.override") is True

    def test_legacy_nested_llm_key_still_works(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    oldplug:\n"
            "      llm:\n        allow_model_override: true\n",
            encoding="utf-8",
        )
        assert plugin_capability_granted("oldplug", "llm.model_override") is True

    def test_legacy_false_stays_denied(self, hermes_home):
        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    oldplug:\n"
            "      allow_tool_override: false\n",
            encoding="utf-8",
        )
        assert plugin_capability_granted("oldplug", "tools.override") is False

    def test_tool_override_gate_uses_canonical_path(self, hermes_home):
        """PluginContext._tool_override_allowed honors capability grant."""
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        record_consent("capplug", ["tools.override"], ["tools.override"])
        manifest = PluginManifest(name="capplug", source="user", key="capplug")
        ctx = PluginContext(manifest, PluginManager())
        assert ctx._tool_override_allowed("write_file") is True

    def test_tool_override_gate_denies_without_grant(self, hermes_home):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        manifest = PluginManifest(name="capplug", source="user", key="capplug")
        ctx = PluginContext(manifest, PluginManager())
        assert ctx._tool_override_allowed("write_file") is False

    def test_tool_override_gate_legacy_key(self, hermes_home):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        (hermes_home / "config.yaml").write_text(
            "plugins:\n  entries:\n    oldplug:\n"
            "      allow_tool_override: true\n",
            encoding="utf-8",
        )
        manifest = PluginManifest(name="oldplug", source="user", key="oldplug")
        ctx = PluginContext(manifest, PluginManager())
        assert ctx._tool_override_allowed("write_file") is True

    def test_bundled_plugin_trusted(self, hermes_home):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        manifest = PluginManifest(name="bplug", source="bundled", key="bplug")
        ctx = PluginContext(manifest, PluginManager())
        assert ctx._tool_override_allowed("write_file") is True
        assert ctx.has_capability("tools.override") is True


# ── ctx.has_capability probing ───────────────────────────────────────────────


class TestHasCapability:
    def test_probe_granted(self, hermes_home):
        from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

        record_consent("capplug", ["llm.model_override"], ["llm.model_override"])
        manifest = PluginManifest(name="capplug", source="user", key="capplug")
        ctx = PluginContext(manifest, PluginManager())
        assert ctx.has_capability("llm.model_override") is True
        assert ctx.has_capability("tools.override") is False
        assert ctx.has_capability("nonsense.capability") is False


# ── Consent CLI flow ─────────────────────────────────────────────────────────


class TestConsentFlow:
    def _console(self, answers=None, interactive=True):
        console = MagicMock()
        it = iter(answers or [])
        console.input.side_effect = lambda *a, **k: next(it, "")
        return console

    def test_consent_yes_records_grant(self, hermes_home, monkeypatch):
        from hermes_cli.plugins_cmd import _run_capability_consent

        console = self._console(["y"])
        with patch("sys.stdin") as stdin, patch("sys.stdout") as stdout:
            stdin.isatty.return_value = True
            stdout.isatty.return_value = True
            granted = _run_capability_consent(
                console, "capplug", ["tools.override"], context="install"
            )
        assert granted is True
        assert plugin_capability_granted("capplug", "tools.override") is True

    def test_consent_decline_leaves_ungranted(self, hermes_home):
        from hermes_cli.plugins_cmd import _run_capability_consent

        console = self._console(["n"])
        with patch("sys.stdin") as stdin, patch("sys.stdout") as stdout:
            stdin.isatty.return_value = True
            stdout.isatty.return_value = True
            granted = _run_capability_consent(
                console, "capplug", ["tools.override"], context="install"
            )
        assert granted is False
        assert plugin_capability_granted("capplug", "tools.override") is False

    def test_non_interactive_fails_closed(self, hermes_home):
        from hermes_cli.plugins_cmd import _run_capability_consent

        console = self._console()
        with patch("sys.stdin") as stdin, patch("sys.stdout") as stdout:
            stdin.isatty.return_value = False
            stdout.isatty.return_value = False
            granted = _run_capability_consent(
                console, "capplug", ["tools.override"], context="install"
            )
        assert granted is False
        assert plugin_capability_granted("capplug", "tools.override") is False
        # No prompt was shown.
        console.input.assert_not_called()

    def test_already_granted_skips_prompt(self, hermes_home):
        from hermes_cli.plugins_cmd import _run_capability_consent

        record_consent("capplug", ["tools.override"], ["tools.override"])
        console = self._console()
        granted = _run_capability_consent(
            console, "capplug", ["tools.override"], context="enable"
        )
        assert granted is True
        console.input.assert_not_called()

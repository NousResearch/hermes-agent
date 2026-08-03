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
from types import SimpleNamespace
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


def _candidate(
    name: str,
    key: str,
    *,
    source: str = "bundled",
    kind: str = "backend",
    version: str = "1",
):
    return (name, version, "", source, None, key, kind)


def _patch_candidate_inventory(monkeypatch, plugins_cmd, candidates) -> None:
    monkeypatch.setattr(
        plugins_cmd,
        "_discover_plugin_candidates",
        lambda **_kwargs: list(candidates),
    )
    monkeypatch.setattr(
        plugins_cmd,
        "_discover_plugin_activation_candidates",
        lambda **_kwargs: list(candidates),
    )


def _patch_activation_io(monkeypatch, plugins_cmd, *, enabled=(), disabled=()):
    writes = []
    monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set(enabled))
    monkeypatch.setattr(plugins_cmd, "_get_disabled_set", lambda: set(disabled))
    monkeypatch.setattr(
        plugins_cmd,
        "_save_enabled_set",
        lambda value: writes.append(("enabled", set(value))),
    )
    monkeypatch.setattr(
        plugins_cmd,
        "_save_disabled_set",
        lambda value: writes.append(("disabled", set(value))),
    )
    monkeypatch.setattr(plugins_cmd, "_set_plugin_entry_flag", lambda *args, **kwargs: None)
    monkeypatch.setattr(plugins_cmd, "_toggle_plugin_toolset", lambda *args, **kwargs: None)
    return writes


class TestBasicAuthActivationMutation:
    def test_unblock_preserves_shared_legacy_provider_deny(self, monkeypatch):
        from hermes_cli import plugins_cmd

        candidates = [
            _candidate("basic", "dashboard_auth/basic"),
            _candidate("basic", "legacy/basic", source="legacy"),
        ]
        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_activation_candidates",
            lambda **_kwargs: candidates,
        )
        cfg = {"plugins": {"disabled": ["basic"]}}

        assert plugins_cmd.ensure_basic_auth_plugin_enabled_in_config(cfg) is True
        assert cfg["plugins"]["disabled"] == ["legacy/basic"]

    def test_same_key_project_override_conflicts_atomically(self, monkeypatch):
        from hermes_cli import plugins_cmd

        candidates = [
            _candidate("basic", "dashboard_auth/basic"),
            _candidate(
                "project-basic",
                "dashboard_auth/basic",
                source="project",
            ),
        ]
        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_activation_candidates",
            lambda **_kwargs: candidates,
        )
        cfg = {"plugins": {"disabled": ["dashboard_auth/basic"]}}

        with pytest.raises(plugins_cmd.PluginActivationConflictError):
            plugins_cmd.ensure_basic_auth_plugin_enabled_in_config(cfg)
        assert cfg == {
            "plugins": {"disabled": ["dashboard_auth/basic"]}
        }

    def test_same_key_override_alias_deny_is_detected(self, monkeypatch):
        from hermes_cli import plugins_cmd

        candidates = [
            _candidate("basic", "dashboard_auth/basic"),
            _candidate(
                "project-basic",
                "dashboard_auth/basic",
                source="project",
            ),
        ]
        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_activation_candidates",
            lambda **_kwargs: candidates,
        )
        cfg = {"plugins": {"disabled": ["project-basic", "other"]}}

        with pytest.raises(plugins_cmd.PluginActivationConflictError):
            plugins_cmd.ensure_basic_auth_plugin_enabled_in_config(cfg)
        assert cfg == {
            "plugins": {"disabled": ["project-basic", "other"]}
        }

    def test_indistinguishable_same_key_override_conflicts_atomically(
        self,
        monkeypatch,
    ):
        from hermes_cli import plugins_cmd

        candidates = [
            _candidate("basic", "dashboard_auth/basic"),
            _candidate("basic", "dashboard_auth/basic", source="user"),
        ]
        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_activation_candidates",
            lambda **_kwargs: candidates,
        )
        cfg = {"plugins": {"disabled": ["dashboard_auth/basic", "other"]}}

        with pytest.raises(plugins_cmd.PluginActivationConflictError):
            plugins_cmd.ensure_basic_auth_plugin_enabled_in_config(cfg)
        assert cfg == {
            "plugins": {"disabled": ["dashboard_auth/basic", "other"]}
        }

    def test_inventory_failure_leaves_config_unchanged(self, monkeypatch):
        from hermes_cli import plugins_cmd

        def fail_inventory(**_kwargs):
            raise OSError("unreadable plugin directory")

        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_activation_candidates",
            fail_inventory,
        )
        cfg = {"plugins": {"disabled": ["basic", "other"]}}

        with pytest.raises(plugins_cmd.PluginActivationConflictError):
            plugins_cmd.ensure_basic_auth_plugin_enabled_in_config(cfg)
        assert cfg == {"plugins": {"disabled": ["basic", "other"]}}


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
    @pytest.mark.parametrize("surface", ("cli", "dashboard"))
    @pytest.mark.parametrize("kind", ("backend", "platform", "model-provider"))
    @pytest.mark.parametrize("grant_identity", ("key", "manifest"))
    @pytest.mark.parametrize("explicitly_disabled", (False, True))
    def test_enable_bundled_default_repairs_legacy_generic_grant(
        self,
        surface,
        kind,
        grant_identity,
        explicitly_disabled,
        monkeypatch,
    ):
        from hermes_cli import plugins_cmd
        from hermes_cli.plugin_activation import PluginActivationState

        key = f"category/default-{kind}"
        manifest_name = f"default-{kind}-manifest"
        stale_grant = key if grant_identity == "key" else manifest_name
        candidates = [_candidate(manifest_name, key, kind=kind)]

        monkeypatch.setattr(
            plugins_cmd,
            "_resolve_plugin_key_and_source",
            lambda *_args, **_kwargs: (key, "bundled", manifest_name, kind),
        )
        _patch_candidate_inventory(monkeypatch, plugins_cmd, candidates)
        writes = _patch_activation_io(
            monkeypatch,
            plugins_cmd,
            enabled={stale_grant},
            disabled={key} if explicitly_disabled else set(),
        )

        if surface == "cli":
            plugins_cmd.cmd_enable(manifest_name)
        else:
            result = plugins_cmd.dashboard_set_agent_plugin_enabled(
                manifest_name,
                enabled=True,
            )
            assert result["ok"] is True
            assert result["unchanged"] is False

        expected = [("enabled", set())]
        if explicitly_disabled:
            expected.append(("disabled", set()))
        assert writes == expected

        repaired = PluginActivationState(enabled=frozenset())
        assert repaired.is_active(
            name=manifest_name,
            key=key,
            source="bundled",
            kind=kind,
        )
        assert not repaired.is_active(
            name=manifest_name,
            key=key,
            source="user",
            kind=kind,
        )

    @pytest.mark.parametrize("surface", ("cli", "dashboard"))
    @pytest.mark.parametrize(
        ("grant_identity", "expected_unchanged"),
        (("key", True), ("external_manifest", True), ("bundled_manifest", False)),
    )
    def test_bundled_alias_enable_preserves_same_key_external_winner(
        self,
        surface,
        grant_identity,
        expected_unchanged,
        monkeypatch,
    ):
        from hermes_cli import plugins_cmd
        from hermes_cli.plugin_activation import PluginActivationState

        key = "shared/key"
        bundled_name = "bundled-copy"
        external_name = "external-copy"
        candidates = [
            _candidate(bundled_name, key),
            _candidate(external_name, key, source="user", version="2"),
        ]
        grant = {
            "key": key,
            "external_manifest": external_name,
            "bundled_manifest": bundled_name,
        }[grant_identity]
        _patch_candidate_inventory(monkeypatch, plugins_cmd, candidates)
        writes = _patch_activation_io(monkeypatch, plugins_cmd, enabled={grant})

        assert plugins_cmd._resolve_plugin_key_and_source(
            bundled_name,
            for_enable=True,
        ) == (key, "user", external_name, "backend")

        if surface == "cli":
            plugins_cmd.cmd_enable(bundled_name, allow_tool_override=False)
        else:
            result = plugins_cmd.dashboard_set_agent_plugin_enabled(
                bundled_name,
                enabled=True,
            )
            assert result["ok"] is True
            assert result["unchanged"] is expected_unchanged

        if expected_unchanged:
            assert writes == []
            effective_enabled = {grant}
        else:
            assert writes == [
                ("enabled", {grant, key}),
                ("disabled", set()),
            ]
            effective_enabled = {grant, key}
        winner, status = plugins_cmd._resolve_plugin_entry_winners(
            candidates,
            PluginActivationState(enabled=frozenset(effective_enabled)),
        )[0]
        assert status == "enabled"
        assert winner[0] == external_name
        assert winner[3] == "user"

    @pytest.mark.parametrize("surface", ("cli", "dashboard"))
    @pytest.mark.parametrize("disabled", (set(), {"target/key"}))
    @pytest.mark.parametrize(
        ("candidates", "expected_disabled"),
        (
            (
                (
                    _candidate("shared-name", "target/key"),
                    _candidate(
                        "shared-name",
                        "other/key",
                        source="user",
                        kind="standalone",
                    ),
                ),
                set(),
            ),
            (
                (
                    _candidate("target-name", "target/key"),
                    _candidate(
                        "target/key",
                        "other/key",
                        source="user",
                        kind="standalone",
                    ),
                ),
                {"other/key"},
            ),
        ),
    )
    def test_default_grant_repair_preserves_cross_key_consent(
        self,
        surface,
        disabled,
        candidates,
        expected_disabled,
        monkeypatch,
    ):
        from hermes_cli import plugins_cmd

        manifest_name, _version, _description, source, _path, key, kind = (
            candidates[0]
        )
        stale_grant = next(
            identity
            for identity in (manifest_name, key)
            if any(
                identity in {other[0], other[5]}
                for other in candidates[1:]
            )
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_resolve_plugin_key_and_source",
            lambda *_args, **_kwargs: (key, source, manifest_name, kind),
        )
        _patch_candidate_inventory(monkeypatch, plugins_cmd, candidates)
        writes = _patch_activation_io(
            monkeypatch,
            plugins_cmd,
            enabled={stale_grant},
            disabled=disabled,
        )

        if surface == "cli":
            plugins_cmd.cmd_enable(manifest_name)
        else:
            result = plugins_cmd.dashboard_set_agent_plugin_enabled(
                manifest_name,
                enabled=True,
            )
            assert result["ok"] is True

        if not disabled:
            assert writes == []
        else:
            assert writes[:2] == [
                ("enabled", {stale_grant}),
                ("disabled", expected_disabled),
            ]

    def test_default_grant_repair_preserves_same_key_opt_in_consent(self):
        from hermes_cli import plugins_cmd

        key = "shared/key"
        candidates = [
            _candidate("opt-in", key, kind="standalone"),
            _candidate("default", key),
        ]
        enabled = {key, "default"}

        assert plugins_cmd._clear_default_plugin_activation_grants(
            enabled,
            key=key,
            candidates=candidates,
        )
        assert enabled == {key}

    def test_disable_preserves_shared_grant_for_other_group(self):
        from hermes_cli import plugins_cmd
        from hermes_cli.plugin_activation import PluginActivationState

        enabled = {"target/key", "shared-name"}
        disabled = {"unrelated"}
        groups = {
            "target/key": {"target/key", "shared-name"},
            "other/key": {"other/key", "shared-name"},
        }

        plugins_cmd._disable_plugin_activation(
            enabled,
            disabled,
            key="target/key",
            identity_groups=groups,
        )

        assert enabled == {"shared-name"}
        assert disabled == {"unrelated", "target/key"}
        state = PluginActivationState(
            enabled=frozenset(enabled),
            disabled=frozenset(disabled),
        )
        assert not state.is_active(
            name="shared-name",
            key="target/key",
            source="user",
            kind="standalone",
        )
        assert state.is_active(
            name="shared-name",
            key="other/key",
            source="user",
            kind="standalone",
        )

    def test_disable_rejects_fully_overlapping_groups_without_mutation(self):
        from hermes_cli import plugins_cmd

        enabled = {"a"}
        disabled = {"unrelated"}
        with pytest.raises(plugins_cmd.PluginActivationConflictError):
            plugins_cmd._disable_plugin_activation(
                enabled,
                disabled,
                key="a",
                identity_groups={"a": {"a", "b"}, "b": {"a", "b"}},
            )
        assert enabled == {"a"}
        assert disabled == {"unrelated"}

    def test_manifest_mutations_do_not_toggle_independent_legacy_provider(self):
        from hermes_cli import plugins_cmd
        from hermes_cli.plugin_activation import PluginActivationState

        key = "model-providers/foo"
        manifest = _candidate("bundled-name", key, kind="model-provider")
        legacy = _candidate("foo", key, source="legacy", kind="model-provider")
        groups = plugins_cmd._plugin_mutation_identity_groups([manifest, legacy])

        disabled = {"bundled-name", "foo"}
        plugins_cmd._clear_plugin_activation_denies(
            disabled,
            key=key,
            identity_groups=groups,
        )
        assert disabled == {"foo"}
        assert PluginActivationState(disabled=frozenset(disabled)).is_active(
            name="bundled-name",
            key=key,
            source="bundled",
            kind="model-provider",
        )

        enabled = {"bundled-name", "foo"}
        disabled = set()
        plugins_cmd._disable_plugin_activation(
            enabled,
            disabled,
            key=key,
            identity_groups=groups,
        )
        assert enabled == {"foo"}
        assert disabled == {"bundled-name"}
        assert PluginActivationState(
            enabled=frozenset(enabled),
            disabled=frozenset(disabled),
        ).is_active(
            name="foo",
            key=key,
            source="legacy",
            kind="model-provider",
        )

    def test_legacy_group_id_cannot_collide_with_manifest_key(self):
        from hermes_cli import plugins_cmd

        manifest_key = "@legacy:foo:0"
        groups = plugins_cmd._plugin_mutation_identity_groups(
            [
                _candidate(manifest_key, manifest_key, source="user"),
                _candidate(
                    "foo",
                    "model-providers/foo",
                    source="legacy",
                    kind="model-provider",
                ),
            ]
        )

        assert len(groups) == 2
        assert groups[manifest_key] == {manifest_key}
        assert {"model-providers/foo", "foo"} in groups.values()

    @pytest.mark.parametrize(
        ("enabled", "disabled"),
        (({"project-name"}, set()), (set(), {"project-name"})),
    )
    def test_composite_inventory_preserves_inactive_project_policy(
        self,
        enabled,
        disabled,
    ):
        from hermes_cli import plugins_cmd

        key = "shared/key"
        bundled = _candidate("bundled-name", key)
        project = _candidate("project-name", key, source="project")
        groups = plugins_cmd._plugin_composite_identity_groups(
            [bundled],
            [bundled, project],
        )

        new_enabled, new_disabled = plugins_cmd._composite_activation_sets(
            [key],
            ["bundled"],
            ["backend"],
            {0},
            disabled,
            enabled=enabled,
            plugin_identities=[{key, bundled[0]}],
            plugin_grant_identities=[{key, bundled[0]}],
            identity_groups=groups,
            activation_candidates=[bundled, project],
        )

        assert new_enabled == enabled
        assert new_disabled == disabled

    def test_enable_rejects_replacement_deny_that_would_block_third_group(self):
        from hermes_cli import plugins_cmd

        disabled = {"shared-name"}
        groups = {
            "target": {"target", "shared-name"},
            "other": {"other", "shared-name"},
            "@legacy:active:0": {"other", "legacy-active"},
        }

        with pytest.raises(plugins_cmd.PluginActivationConflictError):
            plugins_cmd._clear_plugin_activation_denies(
                disabled,
                key="target",
                identity_groups=groups,
            )

        assert disabled == {"shared-name"}

    def test_default_enable_does_not_require_denying_active_legacy_group(self):
        from hermes_cli import plugins_cmd

        key = "model-providers/foo"
        disabled = {"bundled-name"}
        groups = {
            key: {key, "bundled-name"},
            "@legacy:foo:0": {key, "foo"},
        }

        plugins_cmd._clear_plugin_activation_denies(
            disabled,
            key=key,
            identity_groups=groups,
        )

        assert disabled == set()

    @pytest.mark.parametrize("surface", ("cli", "dashboard"))
    def test_disable_ignores_inactive_project_alias_for_runtime_status(
        self,
        surface,
        monkeypatch,
    ):
        from hermes_cli import plugins_cmd

        key = "shared/key"
        bundled = _candidate("bundled-name", key)
        project = _candidate("project-name", key, source="project")

        monkeypatch.setattr(
            plugins_cmd,
            "_resolve_plugin_key_and_source",
            lambda *_args, **_kwargs: (key, "bundled", bundled[0], "backend"),
        )
        def _inventory(*, include_inactive_project=False, **_kwargs):
            return [bundled, project] if include_inactive_project else [bundled]

        monkeypatch.setattr(plugins_cmd, "_discover_plugin_candidates", _inventory)
        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_activation_candidates",
            _inventory,
        )
        writes = _patch_activation_io(
            monkeypatch,
            plugins_cmd,
            disabled={project[0]},
        )

        if surface == "cli":
            plugins_cmd.cmd_disable(key)
        else:
            result = plugins_cmd.dashboard_set_agent_plugin_enabled(
                key,
                enabled=False,
            )
            assert result == {"ok": True, "name": key, "key": key, "unchanged": False}

        assert writes[:2] == [
            ("enabled", set()),
            ("disabled", {project[0], key}),
        ]

    def test_manifestless_provider_can_be_resolved_and_disabled(
        self,
        tmp_path,
        monkeypatch,
    ):
        from hermes_cli import plugins_cmd
        from hermes_cli.plugin_activation import PluginActivationState

        provider_name = "manifestless-73132"
        key = f"model-providers/{provider_name}"
        user_plugins = tmp_path / "plugins"
        provider_dir = user_plugins / "model-providers" / provider_name
        provider_dir.mkdir(parents=True)
        (provider_dir / "__init__.py").write_text("", encoding="utf-8")

        monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: user_plugins)
        monkeypatch.setattr(
            "hermes_cli.plugins.get_bundled_plugins_dir",
            lambda: tmp_path / "bundled",
        )
        monkeypatch.setattr(plugins_cmd, "_discover_entrypoint_plugins", lambda: [])
        monkeypatch.setattr(
            "hermes_cli.config.load_plugin_activation_state",
            lambda: PluginActivationState(enabled=frozenset({key})),
        )
        writes = _patch_activation_io(monkeypatch, plugins_cmd, enabled={key})
        monkeypatch.setattr(
            plugins_cmd,
            "_strict_plugin_activation_candidates",
            lambda: [_candidate(key, key, source="user", kind="model-provider")],
        )

        result = plugins_cmd.dashboard_set_agent_plugin_enabled(key, enabled=False)

        assert result["ok"] is True
        assert result["unchanged"] is False
        assert writes == [("enabled", set()), ("disabled", {key})]

    @pytest.mark.parametrize("surface", ("cli", "dashboard"))
    def test_default_grant_repair_inventory_failure_preserves_grant(
        self,
        surface,
        monkeypatch,
        capsys,
    ):
        from hermes_cli import plugins_cmd

        key = "default/key"
        candidate = _candidate("default-name", key)

        monkeypatch.setattr(
            plugins_cmd,
            "_resolve_plugin_key_and_source",
            lambda *_args, **_kwargs: (key, "bundled", candidate[0], "backend"),
        )

        def _candidates(*, strict=False, **_kwargs):
            if strict:
                raise RuntimeError("broken entry point")
            return [candidate]

        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_activation_candidates",
            _candidates,
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_candidates",
            lambda **_kwargs: [candidate],
        )
        writes = _patch_activation_io(monkeypatch, plugins_cmd, enabled={key})

        if surface == "cli":
            plugins_cmd.cmd_enable(key)
            assert "Skipped legacy activation-grant cleanup" in capsys.readouterr().out
        else:
            result = plugins_cmd.dashboard_set_agent_plugin_enabled(key, enabled=True)
            assert result["ok"] is True
            assert result["unchanged"] is True
            assert "cleanup" in result["warning"]

        assert writes == []

    @pytest.mark.parametrize("surface", ("cli", "dashboard"))
    def test_mutation_rejects_incomplete_inventory_without_writes(
        self,
        surface,
        monkeypatch,
        capsys,
    ):
        from hermes_cli import plugins_cmd

        key = "external/key"
        candidate = _candidate("external-name", key, source="user")
        monkeypatch.setattr(
            plugins_cmd,
            "_resolve_plugin_key_and_source",
            lambda *_args, **_kwargs: (key, "user", candidate[0], "backend"),
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_candidates",
            lambda **_kwargs: [candidate],
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_activation_candidates",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("incomplete")),
        )
        writes = _patch_activation_io(monkeypatch, plugins_cmd)

        if surface == "cli":
            with pytest.raises(SystemExit):
                plugins_cmd.cmd_enable(key, allow_tool_override=False)
            assert "complete plugin inventory" in capsys.readouterr().out
        else:
            result = plugins_cmd.dashboard_set_agent_plugin_enabled(key, enabled=True)
            assert result["ok"] is False
            assert "complete plugin inventory" in result["error"]

        assert writes == []

    @pytest.mark.parametrize("surface", ("cli", "dashboard"))
    def test_install_enable_uses_safe_activation_transaction(
        self,
        surface,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        from hermes_cli import plugins_cmd

        target = tmp_path / "installed"
        target.mkdir()
        monkeypatch.setattr(
            plugins_cmd,
            "_resolve_git_url",
            lambda _identifier: ("https://example.test/plugin.git", None),
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_install_plugin_core",
            lambda *_args, **_kwargs: (target, {}, "installed-name"),
        )
        monkeypatch.setattr(plugins_cmd, "_prompt_plugin_env_vars", lambda *_args: None)
        monkeypatch.setattr(plugins_cmd, "_display_after_install", lambda *_args: None)
        monkeypatch.setattr(plugins_cmd, "_missing_requires_env_names", lambda _m: [])
        monkeypatch.setattr(plugins_cmd, "_invalidate_provider_discovery", lambda: None)
        monkeypatch.setattr(
            plugins_cmd,
            "_enable_plugin_in_config",
            lambda _name: (_ for _ in ()).throw(
                plugins_cmd.PluginActivationConflictError("inventory incomplete")
            ),
        )
        writes = _patch_activation_io(monkeypatch, plugins_cmd)

        if surface == "cli":
            with pytest.raises(SystemExit):
                plugins_cmd.cmd_install("owner/repo", enable=True)
            assert "installed but could not be enabled safely" in capsys.readouterr().out
        else:
            result = plugins_cmd.dashboard_install_plugin(
                "owner/repo",
                force=False,
                enable=True,
            )
            assert result["ok"] is True
            assert result["enabled"] is False
            assert "could not be enabled safely" in result["warnings"][0]

        assert writes == []

    def test_dashboard_reinstall_refreshes_already_enabled_plugin(
        self,
        tmp_path,
        monkeypatch,
    ):
        from hermes_cli import plugins_cmd

        target = tmp_path / "installed"
        target.mkdir()
        monkeypatch.setattr(
            plugins_cmd,
            "_install_plugin_core",
            lambda *_args, **_kwargs: (target, {}, "installed-name"),
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_enable_plugin_in_config",
            lambda _name: ("installed-name", "user", True, False, None),
        )
        monkeypatch.setattr(plugins_cmd, "_missing_requires_env_names", lambda _m: [])
        refreshes = []
        monkeypatch.setattr(
            plugins_cmd,
            "_invalidate_provider_discovery",
            lambda: refreshes.append(True),
        )

        result = plugins_cmd.dashboard_install_plugin(
            "owner/repo",
            force=True,
            enable=True,
        )

        assert result["enabled"] is True
        assert refreshes == [True]

    def test_enabled_filter_honors_canonical_group_alias_deny(self):
        from hermes_cli import plugins_cmd

        winner = ("bundled-name", "1", "", "bundled", None, "shared/key", "backend")
        result = plugins_cmd._filter_plugin_entries(
            [winner],
            SimpleNamespace(enabled=True, no_bundled=False, user=False),
            set(),
            {"lower-alias"},
            identity_groups={"shared/key": {"shared/key", "bundled-name", "lower-alias"}},
        )
        assert result == []

    def test_bad_entrypoint_metadata_still_preserves_runtime_identity(
        self,
        monkeypatch,
    ):
        from hermes_cli import plugins_cmd

        class BrokenDistribution:
            @property
            def metadata(self):
                raise RuntimeError("bad metadata")

        entrypoint = SimpleNamespace(
            name="broken",
            value="broken:register",
            dist=BrokenDistribution(),
        )
        monkeypatch.setattr(
            plugins_cmd.importlib.metadata,
            "entry_points",
            lambda: SimpleNamespace(select=lambda **_kwargs: [entrypoint]),
        )

        expected = [("broken", "", "", "broken:register")]
        assert plugins_cmd._discover_entrypoint_plugins() == expected
        assert plugins_cmd._discover_entrypoint_plugins(strict=True) == expected

    @pytest.mark.parametrize("surface", ("cli", "dashboard"))
    @pytest.mark.parametrize(
        ("candidates", "disabled", "error"),
        (
            (
                [
                    (
                        "shared-b",
                        "1.0.0",
                        "",
                        "bundled",
                        None,
                        "shared-a",
                        "backend",
                    ),
                    (
                        "shared-a",
                        "1.0.0",
                        "",
                        "bundled",
                        None,
                        "shared-b",
                        "backend",
                    ),
                ],
                {"shared-a"},
                "all runtime activation identities overlap",
            ),
        ),
    )
    def test_enable_surfaces_reject_fully_overlapping_identities_without_writes(
        self,
        candidates,
        disabled,
        error,
        surface,
        monkeypatch,
        capsys,
    ):
        from hermes_cli import plugins_cmd

        manifest_name, _version, _description, source, _path, key, kind = (
            candidates[0]
        )
        writes = []

        monkeypatch.setattr(
            plugins_cmd,
            "_resolve_plugin_key_and_source",
            lambda *_args, **_kwargs: (
                key,
                source,
                manifest_name,
                kind,
            ),
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_activation_candidates",
            lambda **_kwargs: candidates,
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_candidates",
            lambda **_kwargs: candidates,
        )
        monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: set())
        monkeypatch.setattr(
            plugins_cmd,
            "_get_disabled_set",
            lambda: set(disabled),
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_save_enabled_set",
            lambda value: writes.append(("enabled", set(value))),
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_save_disabled_set",
            lambda value: writes.append(("disabled", set(value))),
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_toggle_plugin_toolset",
            lambda *args, **kwargs: writes.append(("toolset", args)),
        )

        if surface == "cli":
            with pytest.raises(SystemExit) as exc_info:
                plugins_cmd.cmd_enable(key)
            assert exc_info.value.code == 1
            actual_error = " ".join(capsys.readouterr().out.split())
        else:
            result = plugins_cmd.dashboard_set_agent_plugin_enabled(
                key,
                enabled=True,
            )
            assert result["ok"] is False
            assert result["key"] == key
            actual_error = result["error"]

        assert writes == []
        assert error in actual_error

    @pytest.mark.parametrize("surface", ("cli", "dashboard"))
    def test_enable_surfaces_use_candidate_alias_when_key_belongs_to_legacy(
        self,
        surface,
        monkeypatch,
    ):
        from hermes_cli import plugins_cmd

        key = "model-providers/foo"
        target = _candidate("foo-plugin", key, source="user", kind="model-provider")
        legacy = _candidate("foo", key, source="legacy", kind="model-provider")
        _patch_candidate_inventory(monkeypatch, plugins_cmd, [target, legacy])
        monkeypatch.setattr(
            plugins_cmd,
            "_resolve_plugin_key_and_source",
            lambda *_args, **_kwargs: (key, "user", "foo-plugin", "model-provider"),
        )
        writes = _patch_activation_io(monkeypatch, plugins_cmd)

        if surface == "cli":
            plugins_cmd.cmd_enable(key, allow_tool_override=False)
        else:
            result = plugins_cmd.dashboard_set_agent_plugin_enabled(key, enabled=True)
            assert result["ok"] is True

        assert writes[:2] == [
            ("enabled", {"foo-plugin"}),
            ("disabled", set()),
        ]

    def test_enable_clears_lower_candidate_manifest_deny_when_key_is_allowed(
        self,
        monkeypatch,
    ):
        from hermes_cli import plugins_cmd

        key = "web/firecrawl"
        candidates = [
            ("legacy-firecrawl", "1.0.0", "", "bundled", None, key, "backend"),
            ("web-firecrawl", "2.0.0", "", "user", None, key, "backend"),
            (
                "other-firecrawl",
                "1.0.0",
                "",
                "bundled",
                None,
                "firecrawl",
                "backend",
            ),
        ]
        saved = {}

        monkeypatch.setattr(
            plugins_cmd,
            "_resolve_plugin_key_and_source",
            lambda _name, *, for_enable=False: (
                key,
                "user",
                "web-firecrawl",
                "backend",
            ),
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_activation_candidates",
            lambda **_kwargs: candidates,
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_discover_plugin_candidates",
            lambda **_kwargs: candidates,
        )
        monkeypatch.setattr(plugins_cmd, "_get_enabled_set", lambda: {key})
        monkeypatch.setattr(
            plugins_cmd,
            "_get_disabled_set",
            lambda: {"legacy-firecrawl", "firecrawl"},
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_save_enabled_set",
            lambda value: saved.__setitem__("enabled", set(value)),
        )
        monkeypatch.setattr(
            plugins_cmd,
            "_save_disabled_set",
            lambda value: saved.__setitem__("disabled", set(value)),
        )
        monkeypatch.setattr(plugins_cmd, "_set_plugin_entry_flag", lambda *args: None)

        plugins_cmd.cmd_enable(key, allow_tool_override=False)

        assert saved == {"enabled": {key}, "disabled": {"firecrawl"}}

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
        monkeypatch.setattr(
            plugins_cmd,
            "_discover_entrypoint_plugins",
            lambda **_kwargs: [],
        )
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

    @patch("hermes_cli.plugins_cmd._strict_plugin_activation_candidates")
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_enable_bare_name_writes_key(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, mock_strict, nested_plugin_env,
    ):
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        mock_strict.return_value = [
            _candidate(
                "nemo_relay",
                "observability/nemo_relay",
                source="user",
                kind="standalone",
            )
        ]

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

    @patch("hermes_cli.plugins_cmd._strict_plugin_activation_candidates")
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_enable_flat_plugin_unchanged(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis,
        mock_user, mock_bundled, mock_strict, nested_plugin_env,
    ):
        """Flat plugins keep writing their bare name (key == name) — no regression."""
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        mock_strict.return_value = [
            _candidate("disk-cleanup", "disk-cleanup", source="user", kind="standalone")
        ]

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


    @patch("hermes_cli.plugins_cmd._strict_plugin_activation_candidates")
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._set_plugin_entry_flag")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_interactive_eof_defaults_to_deny(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis, mock_set_flag,
        mock_user, mock_bundled, mock_strict, nested_plugin_env,
    ):
        """Non-interactive stdin (EOFError) must fail closed to deny."""
        from hermes_cli.plugins_cmd import cmd_enable
        mock_user.return_value = nested_plugin_env
        mock_bundled.return_value = nested_plugin_env / "nonexistent"
        mock_strict.return_value = [
            _candidate("disk-cleanup", "disk-cleanup", source="user", kind="standalone")
        ]

        with patch("rich.console.Console.input", side_effect=EOFError):
            cmd_enable("disk-cleanup")

        mock_set_flag.assert_called_once_with(
            "disk-cleanup", "allow_tool_override", False
        )

    @patch("hermes_cli.plugins_cmd._strict_plugin_activation_candidates")
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._set_plugin_entry_flag")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_bundled_plugin_never_prompts_or_writes_entry(
        self, mock_en, mock_dis, mock_save_en, mock_save_dis, mock_set_flag,
        mock_user, mock_bundled, mock_strict, tmp_path,
    ):
        """Bundled plugins are trusted — no consent prompt, no entry write."""
        from hermes_cli.plugins_cmd import cmd_enable
        # Bundled dir holds the plugin; user dir is empty.
        _make_plugin_dir(tmp_path / "bundled", "trusted_bundled", {
            "name": "trusted_bundled", "version": "1.0.0",
        })
        mock_user.return_value = tmp_path / "empty"
        mock_bundled.return_value = tmp_path / "bundled"
        mock_strict.return_value = [
            _candidate(
                "trusted_bundled",
                "trusted_bundled",
                source="bundled",
                kind="standalone",
            )
        ]

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
    def test_fallback_reenable_clears_candidate_manifest_name_denies(
        self, mock_en, mock_save_en, mock_save_dis,
    ):
        from hermes_cli.plugins_cmd import _run_composite_fallback
        from rich.console import Console

        with patch("builtins.input", return_value=""):
            _run_composite_fallback(
                ["web/firecrawl"],
                ["web-firecrawl [bundled]"],
                ["bundled"],
                ["backend"],
                {0},
                {
                    "web/firecrawl",
                    "firecrawl",
                    "legacy-firecrawl",
                    "unrelated-plugin",
                },
                [],
                Console(),
                plugin_identities=[{"web-firecrawl", "legacy-firecrawl"}],
            )

        mock_save_en.assert_called_once_with(set())
        mock_save_dis.assert_called_once_with({"firecrawl", "unrelated-plugin"})

    @pytest.mark.parametrize(
        (
            "unselected_key",
            "selected_identities",
            "unselected_identities",
            "initial_disabled",
        ),
        (
            (
                "firecrawl",
                frozenset({"web/firecrawl", "web-firecrawl"}),
                frozenset({"firecrawl", "other-firecrawl"}),
                frozenset({"firecrawl"}),
            ),
            (
                "video/firecrawl",
                frozenset({"web/firecrawl", "shared-firecrawl"}),
                frozenset({"video/firecrawl", "shared-firecrawl"}),
                frozenset({"shared-firecrawl"}),
            ),
        ),
        ids=("key-leaf", "shared-manifest"),
    )
    @pytest.mark.parametrize("reverse", (False, True))
    def test_activation_sets_are_order_independent_for_collisions(
        self,
        unselected_key,
        selected_identities,
        unselected_identities,
        initial_disabled,
        reverse,
    ):
        from hermes_cli.plugins_cmd import _composite_activation_sets

        rows = [
            ("web/firecrawl", selected_identities, True),
            (unselected_key, unselected_identities, False),
        ]
        if reverse:
            rows.reverse()

        chosen = {i for i, row in enumerate(rows) if row[2]}
        enabled, disabled = _composite_activation_sets(
            [row[0] for row in rows],
            ["bundled"] * len(rows),
            ["backend"] * len(rows),
            chosen,
            set(initial_disabled),
            plugin_identities=[row[1] for row in rows],
        )

        assert enabled == set()
        assert disabled == {unselected_key}

    def test_activation_sets_fail_closed_when_identities_fully_overlap(
        self,
        caplog,
    ):
        from hermes_cli.plugins_cmd import _composite_activation_sets

        enabled, disabled = _composite_activation_sets(
            ["shared-a", "shared-b"],
            ["bundled", "bundled"],
            ["backend", "backend"],
            {0},
            {"shared-a"},
            plugin_identities=[
                {"shared-a", "shared-b"},
                {"shared-a", "shared-b"},
            ],
        )

        assert enabled == set()
        assert disabled == {"shared-a"}
        assert "keeping the existing activation policy fail-closed" in caplog.text

    def test_activation_sets_preserve_hidden_legacy_grant(self):
        from hermes_cli.plugins_cmd import _composite_activation_sets

        key = "model-providers/foo"
        enabled, disabled = _composite_activation_sets(
            [key],
            ["user"],
            ["model-provider"],
            {0},
            {"foo-plugin"},
            enabled={"foo"},
            plugin_identities=[{key, "foo-plugin"}],
            plugin_grant_identities=[{key, "foo-plugin"}],
            identity_groups={
                key: {key, "foo-plugin"},
                "@legacy:foo:0": {key, "foo"},
            },
        )

        assert enabled == {"foo", "foo-plugin"}
        assert disabled == set()

    def test_activation_sets_revoke_grant_shared_only_by_unchecked_rows(self):
        from hermes_cli.plugins_cmd import _composite_activation_sets

        enabled, disabled = _composite_activation_sets(
            ["a", "b"],
            ["user", "user"],
            ["standalone", "standalone"],
            set(),
            set(),
            enabled={"shared"},
            plugin_identities=[{"a", "b", "shared"}] * 2,
            identity_groups={
                "a": {"a", "b", "shared"},
                "b": {"a", "b", "shared"},
            },
        )

        assert enabled == set()
        assert disabled & {"a", "b", "shared"}

    def test_activation_sets_allow_shared_grant_for_all_selected_rows(self):
        from hermes_cli.plugins_cmd import _composite_activation_sets

        enabled, disabled = _composite_activation_sets(
            ["a", "b"],
            ["user", "user"],
            ["standalone", "standalone"],
            {0, 1},
            {"shared"},
            plugin_identities=[{"a", "b", "shared"}] * 2,
            plugin_grant_identities=[{"shared"}] * 2,
            identity_groups={
                "a": {"a", "b", "shared"},
                "b": {"a", "b", "shared"},
            },
        )

        assert enabled == {"a", "b"}
        assert disabled == set()

    def test_activation_sets_prefer_unique_grant_and_revoke_masked_consent(self):
        from hermes_cli.plugins_cmd import _composite_activation_sets

        groups = {"default": {"default", "shared"}, "shared": {"shared", "opt-in"}}
        common = dict(
            plugin_keys=["default", "shared"],
            plugin_sources=["bundled", "user"],
            plugin_kinds=["backend", "standalone"],
            plugin_identities=list(groups.values()),
            plugin_grant_identities=[{"default"}, {"shared", "opt-in"}],
            identity_groups=groups,
        )

        enabled, disabled = _composite_activation_sets(
            chosen={0, 1},
            disabled=set(),
            **common,
        )
        assert enabled == {"opt-in"}
        assert disabled == set()

        enabled, disabled = _composite_activation_sets(
            chosen={0},
            disabled=set(),
            enabled={"shared"},
            **common,
        )
        assert enabled == set()
        assert disabled == {"opt-in"}

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


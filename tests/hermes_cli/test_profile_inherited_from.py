"""Tests for profile inherited_from — config inheritance chain.

Covers:
- _deep_merge_with_conflicts
- _resolve_inherited_config (single parent, multi-parent, chain, circular)
- _collect_inherited_from_union
- _merge_and_flatten_configs
- inherit_profile (CLI creation)
- create_profile multi-source clone
- load_cli_config with inherited_from
"""

import textwrap
from pathlib import Path

import pytest
from hermes_cli.profiles import (
    _deep_merge_with_conflicts,
    _resolve_inherited_config,
    _collect_inherited_from_union,
    _merge_and_flatten_configs,
    inherit_profile,
    create_profile,
    normalize_profile_name,
)


class TestDeepMergeWithConflicts:
    """Unit tests for _deep_merge_with_conflicts."""

    def test_no_overlap(self):
        base = {"a": 1}
        overlay = {"b": 2}
        merged, conflicts = _deep_merge_with_conflicts(base, overlay)
        assert merged == {"a": 1, "b": 2}
        assert conflicts == []

    def test_scalar_override_same_value(self):
        merged, conflicts = _deep_merge_with_conflicts({"a": 1}, {"a": 1})
        assert merged == {"a": 1}
        assert conflicts == []

    def test_scalar_conflict(self):
        merged, conflicts = _deep_merge_with_conflicts({"a": 1}, {"a": 2})
        assert merged == {"a": 1}  # base wins
        assert len(conflicts) == 1
        assert conflicts[0] == ("a", 1, 2, "base", "overlay")

    def test_nested_dict_merge(self):
        base = {"display": {"compact": True, "bell": True}}
        overlay = {"display": {"compact": False, "editor": False}}
        merged, conflicts = _deep_merge_with_conflicts(base, overlay)
        assert merged == {
            "display": {"compact": True, "bell": True, "editor": False}
        }
        assert len(conflicts) == 1
        assert conflicts[0][0] == "display.compact"

    def test_nested_dict_no_conflict(self):
        base = {"display": {"compact": True}}
        overlay = {"display": {"bell": False}}
        merged, conflicts = _deep_merge_with_conflicts(base, overlay)
        assert merged == {"display": {"compact": True, "bell": False}}
        assert conflicts == []

    def test_overlay_adds_new_top_level_key(self):
        base = {"display": {"compact": True}}
        overlay = {"plugins": {"foo": True}}
        merged, conflicts = _deep_merge_with_conflicts(base, overlay)
        assert merged == {"display": {"compact": True}, "plugins": {"foo": True}}
        assert conflicts == []

    def test_list_replaced_not_merged(self):
        base = {"items": [1, 2]}
        overlay = {"items": [3, 4]}
        merged, conflicts = _deep_merge_with_conflicts(base, overlay)
        assert merged == {"items": [1, 2]}  # base wins
        assert len(conflicts) == 1

    def test_prefix_in_conflict_messages(self):
        _, conflicts = _deep_merge_with_conflicts(
            {"a": {"b": {"c": 1}}}, {"a": {"b": {"c": 2}}}
        )
        assert conflicts[0][0] == "a.b.c"


class TestResolveInheritedConfig:
    """Integration tests for _resolve_inherited_config."""

    def _make_profile(self, tmp_path, name, config: dict):
        """Create a minimal profile directory for testing."""
        profile_dir = tmp_path / "profiles" / name
        profile_dir.mkdir(parents=True)
        if config:
            import yaml
            (profile_dir / "config.yaml").write_text(
                yaml.dump(config, default_flow_style=False)
            )
        # Trick get_profile_dir to find our test profiles
        return profile_dir

    def test_single_parent_no_conflict(self, tmp_path, monkeypatch):
        """Profile inherits from one parent with no conflicts."""
        self._make_profile(tmp_path, "base", {
            "display": {"compact": True, "bell": False},
            "agent": {"max_turns": 50},
        })
        self._make_profile(tmp_path, "child", {
            "inherited_from": "base",
            "display": {"skin": "dark"},
        })

        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )

        cfg, warnings = _resolve_inherited_config("child")
        assert cfg["display"]["compact"] is True  # from base
        assert cfg["display"]["bell"] is False    # from base
        assert cfg["display"]["skin"] == "dark"   # child's own
        assert cfg["agent"]["max_turns"] == 50    # from base
        assert warnings == []

    def test_child_overrides_parent(self, tmp_path, monkeypatch):
        """Child's own key overrides parent."""
        self._make_profile(tmp_path, "base", {"display": {"compact": True}})
        self._make_profile(tmp_path, "child", {
            "inherited_from": "base",
            "display": {"compact": False},
        })

        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )

        cfg, warnings = _resolve_inherited_config("child")
        assert cfg["display"]["compact"] is False  # child overrides
        assert warnings == []  # child override is intentional, not a warning

    def test_multi_parent_conflict(self, tmp_path, monkeypatch):
        """Two parents conflict → first parent wins + warning."""
        self._make_profile(tmp_path, "alpha", {"display": {"compact": True}})
        self._make_profile(tmp_path, "beta", {"display": {"compact": False}})
        self._make_profile(tmp_path, "child", {
            "inherited_from": ["alpha", "beta"],
        })

        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )

        cfg, warnings = _resolve_inherited_config("child")
        assert cfg["display"]["compact"] is True  # first parent wins
        assert len(warnings) >= 1
        assert any("display.compact" in w for w in warnings)

    def test_multi_parent_no_conflict(self, tmp_path, monkeypatch):
        """Two parents with disjoint keys merge cleanly."""
        self._make_profile(tmp_path, "alpha", {"display": {"compact": True}})
        self._make_profile(tmp_path, "beta", {"agent": {"max_turns": 50}})
        self._make_profile(tmp_path, "child", {
            "inherited_from": ["alpha", "beta"],
        })

        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )

        cfg, warnings = _resolve_inherited_config("child")
        assert cfg["display"]["compact"] is True
        assert cfg["agent"]["max_turns"] == 50
        assert warnings == []

    def test_chain_inheritance(self, tmp_path, monkeypatch):
        """Grandchild inherits through parent → grandparent."""
        self._make_profile(tmp_path, "grandpa", {"agent": {"max_turns": 30}})
        self._make_profile(tmp_path, "parent", {
            "inherited_from": "grandpa",
            "display": {"compact": True},
        })
        self._make_profile(tmp_path, "child", {
            "inherited_from": "parent",
            "display": {"skin": "dark"},
        })

        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )

        cfg, warnings = _resolve_inherited_config("child")
        assert cfg["agent"]["max_turns"] == 30   # from grandpa
        assert cfg["display"]["compact"] is True # from parent
        assert cfg["display"]["skin"] == "dark"  # child's own
        assert warnings == []

    def test_circular_detection(self, tmp_path, monkeypatch):
        """Circular inherited_from raises ValueError."""
        self._make_profile(tmp_path, "a", {"inherited_from": "b"})
        self._make_profile(tmp_path, "b", {"inherited_from": "a"})

        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )

        with pytest.raises(ValueError, match="Circular inherited_from"):
            _resolve_inherited_config("a")

    def test_no_inherited_from(self, tmp_path, monkeypatch):
        """Profile without inherited_from just returns its own config."""
        self._make_profile(tmp_path, "solo", {
            "display": {"compact": False},
        })

        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )

        cfg, warnings = _resolve_inherited_config("solo")
        assert cfg["display"]["compact"] is False
        assert warnings == []


class TestCollectInheritedFromUnion:
    """Unit tests for _collect_inherited_from_union."""

    def _make_profile(self, tmp_path, name, config: dict):
        profile_dir = tmp_path / "profiles" / name
        profile_dir.mkdir(parents=True)
        if config:
            import yaml
            (profile_dir / "config.yaml").write_text(
                yaml.dump(config, default_flow_style=False)
            )
        return profile_dir

    def test_union_from_multiple_sources(self, tmp_path, monkeypatch):
        self._make_profile(tmp_path, "a", {"inherited_from": "z"})
        self._make_profile(tmp_path, "b", {"inherited_from": ["z", "y"]})
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )
        result = _collect_inherited_from_union(["a", "b"])
        assert result == ["z", "y"]  # deduplicated, order preserved

    def test_no_inherited_from_anywhere(self, tmp_path, monkeypatch):
        self._make_profile(tmp_path, "a", {"display": {"compact": True}})
        self._make_profile(tmp_path, "b", {"agent": {"max_turns": 50}})
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )
        result = _collect_inherited_from_union(["a", "b"])
        assert result == []


class TestMergeAndFlattenConfigs:
    """Unit tests for _merge_and_flatten_configs."""

    def _make_profile(self, tmp_path, name, config: dict):
        profile_dir = tmp_path / "profiles" / name
        profile_dir.mkdir(parents=True)
        if config:
            import yaml
            (profile_dir / "config.yaml").write_text(
                yaml.dump(config, default_flow_style=False)
            )
        return profile_dir

    def test_disjoint_sources_merge(self, tmp_path, monkeypatch):
        self._make_profile(tmp_path, "a", {"display": {"compact": True}})
        self._make_profile(tmp_path, "b", {"agent": {"max_turns": 50}})
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )
        cfg, warnings = _merge_and_flatten_configs(["a", "b"])
        assert cfg["display"]["compact"] is True
        assert cfg["agent"]["max_turns"] == 50
        assert warnings == []

    def test_conflicting_sources_first_wins(self, tmp_path, monkeypatch):
        self._make_profile(tmp_path, "a", {"display": {"compact": True}})
        self._make_profile(tmp_path, "b", {"display": {"compact": False}})
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )
        cfg, warnings = _merge_and_flatten_configs(["a", "b"])
        assert cfg["display"]["compact"] is True  # first wins
        assert len(warnings) >= 1
        assert any("compact" in w for w in warnings)


class TestInheritProfile:
    """Tests for inherit_profile() — CLI-facing function."""

    def _make_source(self, tmp_path, name, config: dict):
        profile_dir = tmp_path / "profiles" / name
        profile_dir.mkdir(parents=True)
        import yaml
        (profile_dir / "config.yaml").write_text(
            yaml.dump(config, default_flow_style=False)
        )
        return profile_dir

    def test_creates_child_with_inherited_from(self, tmp_path, monkeypatch):
        """inherit_profile creates a profile with inherited_from."""
        self._make_source(tmp_path, "base", {
            "inherited_from": None,  # will be ignored as None is default
            "display": {"compact": True},
        })
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )

        result = inherit_profile(["base"], "child")
        assert result.is_dir()
        assert (result / "config.yaml").exists()

        import yaml
        child_cfg = yaml.safe_load((result / "config.yaml").read_text())
        assert child_cfg["inherited_from"] == ["base"]

    def test_multi_parent_inherit(self, tmp_path, monkeypatch):
        """inherit_profile with multiple parents."""
        self._make_source(tmp_path, "alpha", {"display": {"compact": True}})
        self._make_source(tmp_path, "beta", {"agent": {"max_turns": 50}})
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )

        result = inherit_profile(["alpha", "beta"], "child")
        import yaml
        child_cfg = yaml.safe_load((result / "config.yaml").read_text())
        assert child_cfg["inherited_from"] == ["alpha", "beta"]

    def test_self_inherit_rejected(self, tmp_path, monkeypatch):
        """Cannot inherit from self."""
        self._make_source(tmp_path, "base", {})
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )
        with pytest.raises(ValueError, match="Cannot inherit from self"):
            inherit_profile(["base", "child"], "child")

    def test_missing_source_rejected(self, tmp_path, monkeypatch):
        """Non-existent source raises FileNotFoundError."""
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )
        with pytest.raises(FileNotFoundError):
            inherit_profile(["nonexistent"], "child")


class TestCreateProfileMultiSource:
    """Tests for create_profile with comma-separated multi-source clone."""

    def _make_source(self, tmp_path, name, config: dict):
        profile_dir = tmp_path / "profiles" / name
        profile_dir.mkdir(parents=True)
        import yaml
        (profile_dir / "config.yaml").write_text(
            yaml.dump(config, default_flow_style=False)
        )
        return profile_dir

    def test_multi_source_clone_merges_configs(self, tmp_path, monkeypatch):
        """create_profile with A,B → C produces merged flat config."""
        self._make_source(tmp_path, "alpha", {
            "display": {"compact": True, "bell": False},
        })
        self._make_source(tmp_path, "beta", {
            "agent": {"max_turns": 60},
        })
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )
        # Also mock _maybe_register_gateway_service
        monkeypatch.setattr(
            "hermes_cli.profiles._maybe_register_gateway_service",
            lambda _name: None,
        )

        result = create_profile(
            name="merged",
            clone_from="alpha,beta",
            clone_config=True,
            no_alias=True,
        )
        assert result.is_dir()

        import yaml
        cfg = yaml.safe_load((result / "config.yaml").read_text())
        # Both sources' keys present
        assert cfg["display"]["compact"] is True
        assert cfg["display"]["bell"] is False
        assert cfg["agent"]["max_turns"] == 60
        # No inherited_from pointing to the sources themselves
        assert "inherited_from" not in cfg

    def test_multi_source_clone_preserves_ancestors(self, tmp_path, monkeypatch):
        """clone A,B → C where A and B both inherit from Z — C inherits from Z."""
        self._make_source(tmp_path, "z", {"display": {"bell": True}})
        self._make_source(tmp_path, "alpha", {"inherited_from": "z", "display": {"compact": True}})
        self._make_source(tmp_path, "beta", {"inherited_from": "z", "agent": {"max_turns": 50}})
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )
        monkeypatch.setattr(
            "hermes_cli.profiles._maybe_register_gateway_service",
            lambda _name: None,
        )

        result = create_profile(
            name="merged",
            clone_from="alpha,beta",
            clone_config=True,
            no_alias=True,
        )

        import yaml
        cfg = yaml.safe_load((result / "config.yaml").read_text())
        # C inherits from Z (union of A and B's ancestors)
        assert cfg["inherited_from"] == ["z"]
        # But C does NOT inherit from alpha or beta
        assert "alpha" not in cfg.get("inherited_from", [])
        assert "beta" not in cfg.get("inherited_from", [])

    def test_single_source_clone_preserves_behavior(self, tmp_path, monkeypatch):
        """Single-source clone should still work exactly as before."""
        self._make_source(tmp_path, "source", {
            "display": {"compact": True},
        })
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )
        monkeypatch.setattr(
            "hermes_cli.profiles._maybe_register_gateway_service",
            lambda _name: None,
        )

        result = create_profile(
            name="target",
            clone_from="source",
            clone_config=True,
            no_alias=True,
        )
        assert result.is_dir()
        assert (result / "config.yaml").exists()

    def test_clone_self_rejected(self, tmp_path, monkeypatch):
        """Clone where source == target should be rejected."""
        # Make source_a and try to clone source_a,source_a → source_a
        self._make_source(tmp_path, "source_a", {"display": {"compact": True}})
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )
        with pytest.raises(ValueError, match="Cannot clone a profile onto itself"):
            create_profile(
                name="source_a",
                clone_from="source_a,target_b",
                clone_config=True,
            )


class TestInteractiveConflictResolution:
    """Tests for _resolve_conflicts_interactively with mocked input."""

    def _make_profile(self, tmp_path, name, config: dict):
        import yaml
        profile_dir = tmp_path / "profiles" / name
        profile_dir.mkdir(parents=True)
        (profile_dir / "config.yaml").write_text(
            yaml.dump(config, default_flow_style=False)
        )
        return profile_dir

    def test_user_picks_first(self, monkeypatch):
        """User picks option 1 for each conflict."""
        from hermes_cli.profiles import _resolve_conflicts_interactively
        conflicts = [
            ("display.compact", True, False, "alpha", "beta"),
        ]
        monkeypatch.setattr("builtins.input", lambda _: "1")
        result = _resolve_conflicts_interactively(conflicts)
        assert result == {"display.compact": True}

    def test_user_picks_second(self, monkeypatch):
        """User picks option 2."""
        from hermes_cli.profiles import _resolve_conflicts_interactively
        conflicts = [
            ("agent.max_turns", 50, 90, "alpha", "beta"),
        ]
        monkeypatch.setattr("builtins.input", lambda _: "2")
        result = _resolve_conflicts_interactively(conflicts)
        assert result == {"agent.max_turns": 90}

    def test_user_skips(self, monkeypatch):
        """User picks option 3 (skip)."""
        from hermes_cli.profiles import _resolve_conflicts_interactively
        conflicts = [
            ("display.skin", "dark", "light", "alpha", "beta"),
        ]
        monkeypatch.setattr("builtins.input", lambda _: "3")
        result = _resolve_conflicts_interactively(conflicts)
        assert result == {"display.skin": None}

    def test_empty_conflicts(self, monkeypatch):
        """No conflicts — returns empty dict without prompting."""
        from hermes_cli.profiles import _resolve_conflicts_interactively
        result = _resolve_conflicts_interactively([])
        assert result == {}

    def test_multiple_conflicts(self, monkeypatch):
        """Multiple conflicts resolved one by one."""
        from hermes_cli.profiles import _resolve_conflicts_interactively
        conflicts = [
            ("display.compact", True, False, "alpha", "beta"),
            ("agent.max_turns", 50, 90, "alpha", "beta"),
        ]
        choices = ["1", "2"]
        monkeypatch.setattr("builtins.input", lambda _: choices.pop(0))
        result = _resolve_conflicts_interactively(conflicts)
        assert result == {"display.compact": True, "agent.max_turns": 90}

    def test_inherit_profile_with_conflict_interactive(self, tmp_path, monkeypatch):
        """inherit_profile with conflicting parents prompts user, writes resolution."""
        self._make_profile(tmp_path, "alpha", {"display": {"compact": True}})
        self._make_profile(tmp_path, "beta", {"display": {"compact": False}})
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path / "profiles" / name,
        )
        monkeypatch.setattr(
            "hermes_cli.profiles._maybe_register_gateway_service",
            lambda _name: None,
        )
        monkeypatch.setattr("builtins.input", lambda _: "1")

        result = inherit_profile(["alpha", "beta"], "child")
        import yaml
        child_cfg = yaml.safe_load((result / "config.yaml").read_text())
        assert child_cfg["inherited_from"] == ["alpha", "beta"]
        # User chose option 1 (alpha's value)
        assert child_cfg["display"]["compact"] is True

    def test_apply_resolutions_deep(self):
        """_apply_resolutions handles dotted key paths."""
        from hermes_cli.profiles import _apply_resolutions
        config = {"display": {"compact": True, "bell": False}}
        resolutions = {"display.compact": False, "display.skin": "dark"}
        result = _apply_resolutions(config, resolutions)
        assert result["display"]["compact"] is False
        assert result["display"]["skin"] == "dark"
        assert result["display"]["bell"] is False

    def test_apply_resolutions_skip_none(self):
        """_apply_resolutions skips None values."""
        from hermes_cli.profiles import _apply_resolutions
        config = {"display": {"compact": True}}
        resolutions = {"display.compact": None}
        result = _apply_resolutions(config, resolutions)
        assert result["display"]["compact"] is True


class TestLoadCliConfigInheritedFrom:
    """Integration tests for load_cli_config with inherited_from."""

    def _make_profile(self, profiles_root, name, config: dict):
        import yaml
        profile_dir = profiles_root / name
        profile_dir.mkdir(parents=True)
        (profile_dir / "config.yaml").write_text(
            yaml.dump(config, default_flow_style=False)
        )
        return profile_dir

    def test_inherited_from_merged_into_cli_config(self, tmp_path, monkeypatch):
        """load_cli_config merges ancestor config under profile config."""
        import yaml
        profiles_root = tmp_path / "profiles"

        # Ancestor
        self._make_profile(profiles_root, "base", {
            "display": {"compact": True, "bell": False},
            "agent": {"max_turns": 50},
        })

        # Child with inherited_from
        child_dir = self._make_profile(profiles_root, "child", {
            "inherited_from": "base",
            "display": {"skin": "dark"},
        })

        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: profiles_root / name,
        )
        # Make _hermes_home point to child profile
        import cli
        monkeypatch.setattr(cli, "_hermes_home", child_dir)
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: child_dir,
        )

        # Re-import cli to get fresh load_cli_config
        import importlib
        importlib.reload(cli)
        cfg = cli.load_cli_config()

        # From ancestor
        assert cfg["display"]["bell"] is False
        assert cfg["agent"]["max_turns"] == 50
        # From child (overrides ancestor compact)
        assert cfg["display"]["skin"] == "dark"
        # Child doesn't set compact, so ancestor's value should be used
        # (but defaults may have their own compact value, so we check it exists)
        assert "compact" in cfg["display"]

    def test_no_inherited_from_unchanged(self, tmp_path, monkeypatch):
        """Profile without inherited_from loads normally."""
        import yaml
        profiles_root = tmp_path / "profiles"
        solo_dir = self._make_profile(profiles_root, "solo", {
            "display": {"compact": False},
        })

        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: profiles_root / name,
        )
        import cli
        monkeypatch.setattr(cli, "_hermes_home", solo_dir)
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: solo_dir,
        )

        import importlib
        importlib.reload(cli)
        cfg = cli.load_cli_config()
        assert cfg["display"]["compact"] is False

    def test_missing_ancestor_warns_not_crash(self, tmp_path, monkeypatch):
        """Missing ancestor profile logs warning, doesn't crash."""
        import yaml
        profiles_root = tmp_path / "profiles"

        child_dir = self._make_profile(profiles_root, "orphan", {
            "inherited_from": "nonexistent",
            "display": {"compact": False},
        })

        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: profiles_root / name if name != "nonexistent"
            else tmp_path / "profiles" / "nonexistent",
        )
        import cli
        monkeypatch.setattr(cli, "_hermes_home", child_dir)
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: child_dir,
        )

        import importlib
        importlib.reload(cli)
        # Should not raise — just warn and continue
        cfg = cli.load_cli_config()
        assert cfg["display"]["compact"] is False

"""Tests for Blank Slate setup mode (hermes_cli/setup.py).

Blank Slate is the third first-time setup option: everything off except the
bare minimum needed to run an agent (provider/model + file + terminal). These
tests pin the config the writers produce and the invariant that the toolset
resolver + tool-schema builder yield exactly the file/terminal tools.
"""


from hermes_cli.setup_quick import _blank_slate_minimal_toolsets, _blank_slate_minimize_config
from hermes_cli import setup_quick


class TestBlankSlateMinimalToolsets:


    def test_no_disabled_bundle_overlaps_kept_tools(self):
        """Invariant: ``disabled_toolsets`` is applied at *tool* granularity and
        a single tool can belong to several toolsets, so no disabled entry may
        share a tool with a kept toolset — it would silently strip that tool
        from the blank-slate agent (#57315, #58281).
        """
        from toolsets import resolve_toolset
        cfg = {}
        _blank_slate_minimal_toolsets(cfg)
        kept_tools = set()
        for ts in cfg["platform_toolsets"]["cli"]:
            kept_tools.update(resolve_toolset(ts))
        for ts in cfg["agent"]["disabled_toolsets"]:
            overlap = set(resolve_toolset(ts)) & kept_tools
            assert not overlap, (
                f"disabled toolset '{ts}' overlaps kept tools {sorted(overlap)}; "
                "it would silently strip them from the blank-slate agent"
            )


    def test_tool_schema_survives_disabled_toolsets_from_config(self, monkeypatch):
        """Regression: disabled_toolsets must not erase the minimal Blank Slate
        surface when passed to model_tools.  Before the fix, posture toolsets
        like ``coding`` in disabled_toolsets caused model_tools to subtract
        terminal, read_file, write_file, etc. (#57315).

        vision_analyze is additionally check_fn-gated on a resolvable vision
        backend; mock the requirement check so the toolset logic is exercised
        independent of the test host's provider credentials.
        """
        import model_tools
        from tools.registry import registry as _tool_registry
        _entry = _tool_registry.get_entry("vision_analyze")
        monkeypatch.setattr(_entry, "check_fn", lambda: True)
        # This test pins disabled_toolsets SUBTRACTION, not deferral policy —
        # assemble with the legacy everything-eager override so the expected
        # list stays deferral-independent (#97979 defers process_manage by
        # default, which would swap it for the three bridge tools here).
        from tools.tool_search import ToolSearchConfig
        _legacy = ToolSearchConfig.from_raw({"enabled": "on", "defer": []})
        monkeypatch.setattr("tools.tool_search.load_config", lambda: _legacy)
        monkeypatch.setattr("tools.tool_search.load_config_readonly", lambda: _legacy)
        from hermes_cli.tools_config import _get_platform_tools
        cfg = {}
        _blank_slate_minimal_toolsets(cfg)
        _blank_slate_minimize_config(cfg)
        enabled = sorted(_get_platform_tools(cfg, "cli"))
        disabled = cfg.get("agent", {}).get("disabled_toolsets") or []
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=enabled,
            disabled_toolsets=disabled,
            quiet_mode=True,
        )
        names = sorted(
            {(d.get("function") or {}).get("name") or d.get("name") for d in defs}
        )
        assert {"terminal", "read_file", "write_file", "patch", "search_files"} <= set(names)


class TestBlankSlateMinimizeConfig:
    def test_optional_features_turned_off(self):
        cfg = {}
        _blank_slate_minimize_config(cfg)
        assert cfg["compression"]["enabled"] is False
        assert cfg["memory"]["memory_enabled"] is False
        assert cfg["memory"]["user_profile_enabled"] is False
        assert cfg["checkpoints"]["enabled"] is False
        assert cfg["smart_model_routing"]["enabled"] is False


class TestBlankSlateFork:
    """The post-baseline fork: finish now vs walk through configurations."""

    def _patch_common(self, monkeypatch):
        import hermes_cli.setup as s
        # Neutralize side-effecting setup steps and I/O.
        monkeypatch.setattr(s, "setup_model_provider", lambda cfg, **k: None)
        monkeypatch.setattr(s, "setup_terminal_backend", lambda cfg, **k: None)
        monkeypatch.setattr(s, "save_config", lambda cfg: None)
        monkeypatch.setattr(s, "_print_setup_summary", lambda cfg, home: None)
        monkeypatch.setattr(s, "print_header", lambda *a, **k: None)
        monkeypatch.setattr(s, "print_info", lambda *a, **k: None)
        monkeypatch.setattr(s, "print_success", lambda *a, **k: None)
        monkeypatch.setattr(s, "print_warning", lambda *a, **k: None)

    def test_finish_now_skips_walkthrough(self, monkeypatch, tmp_path):
        import hermes_cli.setup as s
        self._patch_common(monkeypatch)
        # Fork prompt returns 0 = finish now.
        monkeypatch.setattr(s, "prompt_choice", lambda *a, **k: 0)
        walked = {"called": False}
        monkeypatch.setattr(setup_quick, "_blank_slate_walkthrough",
                            lambda cfg, home: walked.__setitem__("called", True))
        opted_out = {"value": None}
        monkeypatch.setattr("tools.skills_sync_bundled_ops.set_bundled_skills_opt_out",
                            lambda enabled: opted_out.__setitem__("value", enabled))

        cfg = {}
        setup_quick._run_blank_slate_setup(cfg, tmp_path, is_existing=False)

        # Minimal baseline was applied, walkthrough was NOT run.
        assert cfg["platform_toolsets"]["cli"] == ["file", "skills", "terminal", "vision"]
        assert walked["called"] is False
        # Finish-now path records the skill opt-out (no bundled skills).
        assert opted_out["value"] is True

    def test_finish_now_removes_installer_seeded_skills(self, monkeypatch, tmp_path):
        """Regression (#98027): the installer seeds the full bundled catalog BEFORE the
        setup wizard runs. Blank Slate used to write only the .no-bundled-skills marker
        (which blocks *future* seeding) and re-sync, leaving every installer-seeded skill
        on disk. End to end on a real skills dir: after an installer-style sync, finishing
        Blank Slate must leave only the essential skills, plus anything the user owns."""
        import hermes_cli.setup as s
        from unittest.mock import patch
        from agent.skill_utils import ESSENTIAL_SKILLS
        from tools.skills_sync import sync_skills
        self._patch_common(monkeypatch)
        monkeypatch.setattr(s, "prompt_choice", lambda *a, **k: 0)  # finish now

        bundled = tmp_path / "bundled"
        for n in ("alpha", "beta", *sorted(ESSENTIAL_SKILLS)):
            (bundled / n).mkdir(parents=True)
            (bundled / n / "SKILL.md").write_text(f"---\nname: {n}\n---\nbody {n}\n")
        home = tmp_path / "home"
        skills_dir = home / "skills"
        home.mkdir()
        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync._get_optional_dir", return_value=tmp_path / "optional-skills"), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", skills_dir / ".bundled_manifest"), \
             patch("tools.skills_sync.HERMES_HOME", home):
            sync_skills(quiet=True)  # what the installer does before `hermes setup`
            (skills_dir / "beta" / "SKILL.md").write_text("---\nname: beta\n---\nEDITED\n")
            (skills_dir / "mine").mkdir()
            (skills_dir / "mine" / "SKILL.md").write_text("---\nname: mine\n---\nlocal\n")

            setup_quick._run_blank_slate_setup({}, home, is_existing=False)

            assert (home / ".no-bundled-skills").exists()
            assert not (skills_dir / "alpha").exists()            # pristine bundled: removed
            assert "EDITED" in (skills_dir / "beta" / "SKILL.md").read_text()  # user-edited: kept
            assert (skills_dir / "mine" / "SKILL.md").exists()    # hand-written: kept
            for n in ESSENTIAL_SKILLS:                            # essential: kept
                assert (skills_dir / n / "SKILL.md").exists()

    def test_opt_in_path_does_not_remove_skills(self, monkeypatch):
        """Seeding (opt_out=False) must never call the destructive removal."""
        calls = []
        monkeypatch.setattr("tools.skills_sync_bundled_ops.set_bundled_skills_opt_out", lambda e: None)
        monkeypatch.setattr("tools.skills_sync_bundled_ops.remove_pristine_bundled_skills",
                            lambda dry_run=False: calls.append(dry_run) or {"removed": []})
        monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=False: {"copied": []})
        seen = {}
        setup_quick._set_bundled_skills_opt_out(False, "t", on_success=lambda r: seen.update(r))
        assert calls == []
        assert seen["removed"] == []

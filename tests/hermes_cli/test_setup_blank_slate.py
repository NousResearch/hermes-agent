"""Tests for Blank Slate setup mode (hermes_cli/setup.py).

Blank Slate is the third first-time setup option: everything off except the
bare minimum needed to run an agent (provider/model + file + terminal). These
tests pin the config the writers produce and the invariant that the toolset
resolver + tool-schema builder yield exactly the file/terminal tools.
"""

import pytest

from hermes_cli.setup import (
    _blank_slate_minimal_toolsets,
    _blank_slate_minimize_config,
)


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
        assert names == ["patch", "process", "read_file", "search_files",
                         "skill_manage", "skill_view", "skills_list",
                         "terminal", "vision_analyze", "write_file"]


class TestBlankSlateMinimizeConfig:
    def test_optional_features_turned_off(self):
        cfg = {}
        _blank_slate_minimize_config(cfg)
        assert cfg["compression"]["enabled"] is False
        assert cfg["memory"]["memory_enabled"] is False
        assert cfg["memory"]["user_profile_enabled"] is False
        assert cfg["checkpoints"]["enabled"] is False
        assert cfg["smart_model_routing"]["enabled"] is False
        assert cfg["session_reset"]["mode"] == "none"


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
        monkeypatch.setattr(s, "_blank_slate_walkthrough",
                            lambda cfg, home: walked.__setitem__("called", True))
        opted_out = {"value": None}
        monkeypatch.setattr("tools.skills_sync.set_bundled_skills_opt_out",
                            lambda enabled: opted_out.__setitem__("value", enabled))

        cfg = {}
        s._run_blank_slate_setup(cfg, tmp_path, is_existing=False)

        # Minimal baseline was applied, walkthrough was NOT run.
        assert cfg["platform_toolsets"]["cli"] == ["file", "skills", "terminal", "vision"]
        assert walked["called"] is False
        # Finish-now path records the skill opt-out (no bundled skills).
        assert opted_out["value"] is True


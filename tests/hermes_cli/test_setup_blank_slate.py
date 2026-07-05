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
    def test_only_file_and_terminal_enabled_for_cli(self):
        cfg = {}
        _blank_slate_minimal_toolsets(cfg)
        assert cfg["platform_toolsets"]["cli"] == ["file", "terminal"]

    def test_disabled_toolsets_excludes_kept_and_covers_known(self):
        cfg = {}
        _blank_slate_minimal_toolsets(cfg)
        disabled = set(cfg["agent"]["disabled_toolsets"])
        # The two kept toolsets must NOT be in the disabled list.
        assert "file" not in disabled
        assert "terminal" not in disabled
        # A representative spread of capabilities must be suppressed.
        for ts in ("web", "browser", "code_execution", "vision", "memory",
                   "delegation", "cronjob", "skills", "image_gen"):
            assert ts in disabled
        # The recovered non-configurable toolset that used to leak is suppressed.
        assert "kanban" in disabled

    def test_disabled_toolsets_excludes_posture_toolsets(self):
        """Posture toolsets (e.g. coding) are session-level selections made by
        agent/coding_context.py — not permanent user-facing disables.  Including
        them in disabled_toolsets causes model_tools to subtract their tools
        (terminal, read_file, …) from the minimal Blank Slate surface (#57315).
        """
        cfg = {}
        _blank_slate_minimal_toolsets(cfg)
        disabled = set(cfg["agent"]["disabled_toolsets"])
        assert "coding" not in disabled

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

    def test_resolver_yields_exactly_file_and_terminal(self):
        from hermes_cli.tools_config import _get_platform_tools
        cfg = {}
        _blank_slate_minimal_toolsets(cfg)
        _blank_slate_minimize_config(cfg)
        resolved = set(_get_platform_tools(cfg, "cli"))
        assert resolved == {"file", "terminal"}

    def test_tool_schema_builder_yields_only_file_and_terminal_tools(self):
        # End-to-end: the exact schema set the agent would send to the model.
        import model_tools
        from hermes_cli.tools_config import _get_platform_tools
        cfg = {}
        _blank_slate_minimal_toolsets(cfg)
        _blank_slate_minimize_config(cfg)
        enabled = sorted(_get_platform_tools(cfg, "cli"))
        defs = model_tools.get_tool_definitions(
            enabled_toolsets=enabled, disabled_toolsets=None, quiet_mode=True
        )
        names = sorted(
            {(d.get("function") or {}).get("name") or d.get("name") for d in defs}
        )
        assert names == ["patch", "process", "read_file", "search_files",
                         "terminal", "write_file"]

    def test_tool_schema_survives_disabled_toolsets_from_config(self):
        """Regression: disabled_toolsets must not erase the minimal Blank Slate
        surface when passed to model_tools.  Before the fix, posture toolsets
        like ``coding`` in disabled_toolsets caused model_tools to subtract
        terminal, read_file, write_file, etc. (#57315).
        """
        import model_tools
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
                         "terminal", "write_file"]


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

    def test_does_not_clobber_unrelated_keys(self):
        cfg = {"model": {"provider": "openrouter", "default": "x/y"}}
        _blank_slate_minimize_config(cfg)
        # Model config is untouched by the minimizer.
        assert cfg["model"]["provider"] == "openrouter"
        assert cfg["model"]["default"] == "x/y"


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
        assert cfg["platform_toolsets"]["cli"] == ["file", "terminal"]
        assert walked["called"] is False
        # Finish-now path records the skill opt-out (no bundled skills).
        assert opted_out["value"] is True

    def test_walkthrough_path_invokes_walkthrough(self, monkeypatch, tmp_path):
        import hermes_cli.setup as s
        self._patch_common(monkeypatch)
        # Fork prompt returns 1 = walk through.
        monkeypatch.setattr(s, "prompt_choice", lambda *a, **k: 1)
        walked = {"called": False}
        monkeypatch.setattr(s, "_blank_slate_walkthrough",
                            lambda cfg, home: walked.__setitem__("called", True))

        cfg = {}
        s._run_blank_slate_setup(cfg, tmp_path, is_existing=False)

        assert cfg["platform_toolsets"]["cli"] == ["file", "terminal"]
        assert walked["called"] is True

"""Profile opt-in toolsets (e.g. kanban) live in top-level config ``toolsets``.

Desktop feature toggles and ``tools.configure`` must write that list so
``_profile_has_kanban_toolset`` / skill environment gates agree with the UI.
Regression for #96969.
"""

from __future__ import annotations

from hermes_cli.tools_config import (
    PROFILE_OPT_IN_TOOLSETS,
    _apply_profile_toolset_change,
)


def test_kanban_is_a_profile_opt_in_toolset():
    assert "kanban" in PROFILE_OPT_IN_TOOLSETS


def test_enable_kanban_appends_top_level_toolsets(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("toolsets:\n  - hermes-cli\n", encoding="utf-8")

    cfg = {"toolsets": ["hermes-cli"]}
    _apply_profile_toolset_change(cfg, ["kanban"], "enable")
    assert cfg["toolsets"] == ["hermes-cli", "kanban"]

    _apply_profile_toolset_change(cfg, ["kanban"], "enable")
    assert cfg["toolsets"] == ["hermes-cli", "kanban"]

    from tools.kanban_tools import _profile_has_kanban_toolset
    from hermes_cli.config import save_config

    save_config(cfg)
    assert _profile_has_kanban_toolset() is True


def test_disable_kanban_removes_top_level_toolsets(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {"toolsets": ["hermes-cli", "kanban"]}
    _apply_profile_toolset_change(cfg, ["kanban"], "disable")
    assert cfg["toolsets"] == ["hermes-cli"]

    _apply_profile_toolset_change(cfg, ["kanban"], "disable")
    assert cfg["toolsets"] == ["hermes-cli"]

    from tools.kanban_tools import _profile_has_kanban_toolset
    from hermes_cli.config import save_config

    save_config(cfg)
    assert _profile_has_kanban_toolset() is False


def test_enable_seeds_toolsets_when_missing():
    cfg: dict = {}
    _apply_profile_toolset_change(cfg, ["kanban"], "enable")
    assert cfg["toolsets"] == ["kanban"]

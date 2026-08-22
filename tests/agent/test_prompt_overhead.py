from __future__ import annotations

from agent.prompt_overhead import resolve_prompt_overhead_modes


def test_prompt_overhead_defaults_are_full():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    modes = resolve_prompt_overhead_modes({}, platform="cli")

    assert DEFAULT_CONFIG["prompt_overhead"] == {
        "skill_index_mode": "full",
        "tool_schema_mode": "full",
        "platforms": {},
    }
    assert modes.skill_index_mode == "full"
    assert modes.tool_schema_mode == "full"
    assert modes.platform == "cli"


def test_platform_overrides_have_per_key_precedence():
    modes = resolve_prompt_overhead_modes(
        {
            "prompt_overhead": {
                "skill_index_mode": "compact",
                "tool_schema_mode": "minimal",
                "platforms": {
                    "FeiShu": {
                        "skill_index_mode": "minimal",
                    }
                },
            }
        },
        platform="feishu",
    )

    assert modes.skill_index_mode == "minimal"
    assert modes.tool_schema_mode == "minimal"


def test_invalid_values_fall_back_to_global_then_full():
    modes = resolve_prompt_overhead_modes(
        {
            "prompt_overhead": {
                "skill_index_mode": "unknown",
                "tool_schema_mode": "compact",
                "platforms": {
                    "slack": {
                        "skill_index_mode": "invalid",
                        "tool_schema_mode": "invalid",
                    }
                },
            }
        },
        platform="slack",
    )

    assert modes.skill_index_mode == "full"
    assert modes.tool_schema_mode == "compact"


def test_skill_index_modes_use_separate_cache_entries(monkeypatch, tmp_path):
    from agent import prompt_overhead
    from agent.prompt_builder import (
        build_skills_system_prompt,
        clear_skills_system_prompt_cache,
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skill_dir = tmp_path / "skills" / "tools" / "verbose-skill"
    skill_dir.mkdir(parents=True)
    description = (
        "Use when investigating a deliberately verbose workflow with several "
        "important constraints and a unique full-description tail"
    )
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: verbose-skill\ndescription: {description}\n---\n"
    )

    state = {"mode": "compact"}
    monkeypatch.setattr(
        prompt_overhead,
        "_load_config",
        lambda: {"prompt_overhead": {"skill_index_mode": state["mode"]}},
    )
    clear_skills_system_prompt_cache(clear_snapshot=True)
    try:
        compact = build_skills_system_prompt()
        state["mode"] = "minimal"
        minimal = build_skills_system_prompt()
        state["mode"] = "full"
        full = build_skills_system_prompt()
    finally:
        clear_skills_system_prompt_cache(clear_snapshot=True)

    assert "verbose-skill" in compact
    assert "unique full-description tail" not in compact
    assert "verbose-skill" in minimal
    assert description not in minimal
    assert "Use when investigating a deliberately verbose workflow" in full
    assert full != compact
    assert len(minimal) < len(compact) < len(full)

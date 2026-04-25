from __future__ import annotations

import json
from pathlib import Path

import agent.prompt_builder as prompt_builder
import agent.skill_utils as skill_utils
import tools.skills_tool as skills_tool


def _write_skill(skill_dir: Path, frontmatter: str, body: str = "Body.\n") -> Path:
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(frontmatter + "\n" + body, encoding="utf-8")
    return skill_md


def test_skill_view_resolves_frontmatter_name(monkeypatch, tmp_path):
    local_skills = tmp_path / "skills"
    _write_skill(
        local_skills / "mlops" / "models" / "audiocraft",
        "---\nname: audiocraft-audio-generation\ndescription: AudioCraft wrapper\n---\n",
    )

    monkeypatch.setattr(skills_tool, "SKILLS_DIR", local_skills)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [])

    result = json.loads(skills_tool.skill_view("audiocraft-audio-generation"))
    assert result["success"] is True
    assert result["name"] == "audiocraft-audio-generation"
    assert "AudioCraft wrapper" in result["description"]


def test_build_skills_system_prompt_prefers_specific_skills_and_normalizes_files_toolset(
    monkeypatch, tmp_path
):
    local_skills = tmp_path / "skills"
    _write_skill(
        local_skills / "research" / "research-paper-writing",
        "---\nname: research-paper-writing\ndescription: Research paper workflow\nmetadata:\n  hermes:\n    requires_toolsets: [terminal, files]\n---\n",
    )

    monkeypatch.setattr(prompt_builder, "get_skills_dir", lambda: local_skills)
    monkeypatch.setattr(prompt_builder, "get_all_skills_dirs", lambda: [local_skills])
    monkeypatch.setattr(prompt_builder, "iter_skill_index_files", skill_utils.iter_skill_index_files)
    monkeypatch.setattr(prompt_builder, "get_disabled_skill_names", lambda: set())
    prompt_builder._SKILLS_PROMPT_CACHE.clear()

    result = prompt_builder.build_skills_system_prompt(available_toolsets={"terminal", "file"})

    assert "research-paper-writing" in result
    assert "Prefer the most specific skill" in result
    assert "even partially relevant" not in result
    assert "Err on the side of loading" not in result

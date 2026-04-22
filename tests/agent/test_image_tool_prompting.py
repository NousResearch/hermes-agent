from agent.prompt_builder import OPENAI_MODEL_EXECUTION_GUIDANCE, build_skills_system_prompt
from tools.delegate_tool import DEFAULT_TOOLSETS


def test_openai_guidance_mentions_image_generate_for_image_requests():
    assert "image_generate" in OPENAI_MODEL_EXECUTION_GUIDANCE
    assert "Image generation requests" in OPENAI_MODEL_EXECUTION_GUIDANCE


def test_skills_guidance_mentions_image_generate(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills"
    sample_skill = skills_dir / "sample-skill"
    sample_skill.mkdir(parents=True)
    (sample_skill / "SKILL.md").write_text(
        "---\n"
        "name: sample-skill\n"
        "description: sample\n"
        "---\n\n"
        "# Sample\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("agent.prompt_builder.get_skills_dir", lambda: skills_dir)
    monkeypatch.setattr("agent.prompt_builder.get_all_skills_dirs", lambda: [skills_dir])

    prompt = build_skills_system_prompt(available_tools={"image_generate"})
    assert "use the image_generate tool" in prompt


def test_delegate_default_toolsets_include_image_gen():
    assert "image_gen" in DEFAULT_TOOLSETS

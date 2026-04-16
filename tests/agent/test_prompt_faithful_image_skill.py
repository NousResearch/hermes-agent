from pathlib import Path
from unittest.mock import patch

from agent.skill_commands import scan_skill_commands


def test_repo_bundled_prompt_faithful_image_skill_is_discoverable():
    project_skills_dir = Path(__file__).resolve().parents[2] / "skills"

    with patch("tools.skills_tool.SKILLS_DIR", project_skills_dir):
        commands = scan_skill_commands()

    assert "/prompt-faithful-image-generation" in commands
    assert commands["/prompt-faithful-image-generation"]["name"] == "prompt-faithful-image-generation"

"""
Tests for the Skill Curator skill (skills/software-development/skill-curator).

Covers:
- Trajectory parsing from JSONL transcripts
- Validation of in-repo SKILL.md standards (length, periods, marketing words, paths)
- Automated scaffolding of new skill directory structures
- CLI subprocess commands
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CURATOR_SCRIPT = (
    REPO_ROOT
    / "skills"
    / "software-development"
    / "skill-curator"
    / "scripts"
    / "curate_skill.py"
)

# Import module directly
sys.path.insert(0, str(CURATOR_SCRIPT.parent))
import curate_skill


@pytest.fixture
def sample_transcript(tmp_path):
    transcript_file = tmp_path / "sample_transcript.jsonl"
    events = [
        {"type": "USER_INPUT", "content": "Set up a new Redis caching layer and test it."},
        {
            "type": "PLANNER_RESPONSE",
            "content": "I will inspect the existing Redis configuration and write a test script.",
            "tool_calls": [
                {"name": "read_file", "args": {"path": "config/redis.conf"}},
                {"name": "terminal", "args": {"command": "pytest tests/test_redis.py"}},
            ],
        },
        {"type": "USER_INPUT", "content": "Great, now package this workflow into a skill."},
    ]
    with open(transcript_file, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return transcript_file


class TestCurateSkillCore:
    def test_parse_transcript(self, sample_transcript):
        data = curate_skill.parse_transcript(sample_transcript)
        assert len(data["user_prompts"]) == 2
        assert "Set up a new Redis caching layer and test it." in data["user_prompts"]
        assert len(data["tool_calls"]) == 2
        assert data["tool_calls"][0]["name"] == "read_file"
        assert data["tool_calls"][1]["name"] == "terminal"

    def test_validate_valid_skill(self, tmp_path):
        valid_dir = tmp_path / "my-test-skill"
        valid_dir.mkdir()
        valid_skill = valid_dir / "SKILL.md"
        valid_content = """---
name: my-test-skill
description: Run automated test workflows with predictable outputs.
version: 0.1.0
author: Contributor (handle), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Testing, Workflow]
    category: software-development
    related_skills: [hermes-agent-skill-authoring]
---

# My Test Skill

Valid skill content body.
"""
        valid_skill.write_text(valid_content, encoding="utf-8")
        is_valid, errors = curate_skill.validate_skill_file(valid_skill)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_rejects_long_description(self, tmp_path):
        bad_dir = tmp_path / "long-desc"
        bad_dir.mkdir()
        bad_skill = bad_dir / "SKILL.md"
        bad_content = """---
name: long-desc
description: This description is way too long and significantly exceeds sixty characters ceiling.
version: 0.1.0
author: Contributor, Hermes Agent
license: MIT
platforms: [linux, macos]
---

# Body
"""
        bad_skill.write_text(bad_content, encoding="utf-8")
        is_valid, errors = curate_skill.validate_skill_file(bad_skill)
        assert is_valid is False
        assert any("exceeds 60 chars" in e for e in errors)

    def test_validate_rejects_marketing_words(self, tmp_path):
        bad_dir = tmp_path / "marketing-skill"
        bad_dir.mkdir()
        bad_skill = bad_dir / "SKILL.md"
        bad_content = """---
name: marketing-skill
description: A powerful tool for running automated benchmarks.
version: 0.1.0
author: Contributor, Hermes Agent
license: MIT
platforms: [linux, macos]
---

# Body
"""
        bad_skill.write_text(bad_content, encoding="utf-8")
        is_valid, errors = curate_skill.validate_skill_file(bad_skill)
        assert is_valid is False
        assert any("Marketing word 'powerful' forbidden" in e for e in errors)

    def test_scaffold_skill_creates_structure(self, tmp_path):
        out_dir = tmp_path / "scaffolded-skill"
        skill_file = curate_skill.scaffold_skill(
            "scaffolded-skill",
            "software-development",
            description="Run scaffolded test workflows.",
            output_dir=out_dir,
        )
        assert skill_file.exists()
        assert (out_dir / "scripts").is_dir()
        assert (out_dir / "references").is_dir()

        is_valid, errors = curate_skill.validate_skill_file(skill_file)
        assert is_valid is True, f"Scaffolded skill failed validation: {errors}"


class TestCurateSkillCLI:
    def test_cli_extract_json(self, sample_transcript):
        res = subprocess.run(
            [
                sys.executable,
                str(CURATOR_SCRIPT),
                "extract",
                str(sample_transcript),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout)
        assert "user_prompts" in data
        assert len(data["tool_calls"]) == 2

    def test_cli_validate_success(self):
        curator_skill_md = CURATOR_SCRIPT.parent.parent / "SKILL.md"
        res = subprocess.run(
            [
                sys.executable,
                str(CURATOR_SCRIPT),
                "validate",
                str(curator_skill_md),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "is fully compliant" in res.stdout

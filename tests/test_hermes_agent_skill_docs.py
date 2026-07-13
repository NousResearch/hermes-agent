"""Validate local documentation links in the Hermes Agent skill index."""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = REPO_ROOT / "skills/autonomous-ai-agents/hermes-agent/SKILL.md"


def test_hermes_agent_skill_maps_to_existing_docs():
    skill = SKILL_PATH.read_text(encoding="utf-8")
    assert "~/.hermes/hermes-agent/website/docs/" not in skill

    paths = set(re.findall(r"`(website/docs/[^`]+)`", skill))
    paths = {path for path in paths if not path.endswith("/") and "<" not in path and "*" not in path}

    assert paths
    assert all((REPO_ROOT / path).is_file() for path in paths)

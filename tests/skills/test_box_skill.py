"""Tests for the bundled Box productivity skill (Hermes-native, CCG-first)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "productivity" / "box"
SKILL_MD = SKILL_DIR / "SKILL.md"
REFERENCES_DIR = SKILL_DIR / "references"
TEMPLATES_DIR = SKILL_DIR / "templates"

REQUIRED_REFERENCE_FILES = {
    "auth-and-setup.md",
    "cli-guide.md",
    "content-workflows.md",
    "search-and-ai.md",
    "bulk-operations.md",
    "webhooks-and-events.md",
    "rest-api.md",
    "sdk-development.md",
    "troubleshooting.md",
}

FORBIDDEN_FILES = {
    "UPSTREAM.md",
    "references/hermes-mcp-setup.md",
    "references/mcp-tool-patterns.md",
    "references/workflows.md",
    "references/box-cli.md",
    "references/ai-and-retrieval.md",
    "references/rest-calls.md",
}


def _parse_frontmatter(content: str) -> dict:
    from agent.skill_utils import parse_frontmatter

    fm, _ = parse_frontmatter(content)
    return fm


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict:
    return _parse_frontmatter(skill_text)


def test_skill_md_exists():
    assert SKILL_MD.is_file()


def test_frontmatter(frontmatter: dict):
    assert frontmatter.get("name") == "box"
    desc = frontmatter.get("description")
    assert isinstance(desc, str) and len(desc) > 0
    assert len(desc) <= 60, f"description is {len(desc)} chars: {desc!r}"
    assert desc.endswith(".")
    assert frontmatter.get("author") == "community"
    assert frontmatter.get("license") == "MIT"
    platforms = frontmatter.get("platforms")
    assert isinstance(platforms, list)
    assert set(platforms) >= {"linux", "macos", "windows"}


def test_prerequisites_env_vars(frontmatter: dict):
    prereqs = frontmatter.get("prerequisites") or {}
    env_vars = prereqs.get("env_vars") or []
    assert "BOX_CLIENT_ID" in env_vars
    assert "BOX_CLIENT_SECRET" in env_vars
    assert "BOX_ENTERPRISE_ID" in env_vars
    assert "box" in (prereqs.get("commands") or [])


def test_box_env_vars_registered_for_setup():
    from hermes_cli.config import OPTIONAL_ENV_VARS

    assert OPTIONAL_ENV_VARS["BOX_CLIENT_ID"]["category"] == "skill"
    assert OPTIONAL_ENV_VARS["BOX_CLIENT_ID"]["password"] is False
    assert OPTIONAL_ENV_VARS["BOX_CLIENT_SECRET"]["category"] == "skill"
    assert OPTIONAL_ENV_VARS["BOX_CLIENT_SECRET"]["password"] is True
    assert OPTIONAL_ENV_VARS["BOX_ENTERPRISE_ID"]["category"] == "skill"
    assert OPTIONAL_ENV_VARS["BOX_ENTERPRISE_ID"]["password"] is False


def test_required_reference_files_exist():
    for name in REQUIRED_REFERENCE_FILES:
        assert (REFERENCES_DIR / name).is_file(), f"missing references/{name}"


def test_forbidden_files_removed():
    for rel in FORBIDDEN_FILES:
        assert not (SKILL_DIR / rel).exists(), f"should not exist: {rel}"


def test_no_mcp_mentions_in_skill_tree():
    for path in [SKILL_MD, *REFERENCES_DIR.glob("*.md")]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\bmcp\b", text, re.IGNORECASE), f"MCP mention in {path.name}"


def test_all_skill_md_references_exist(skill_text: str):
    for name in re.findall(r"`references/([^`]+\.md)`", skill_text):
        assert (REFERENCES_DIR / name).is_file(), f"missing references/{name}"
    for name in re.findall(r"`templates/([^`]+)`", skill_text):
        assert (TEMPLATES_DIR / name).is_file(), f"missing templates/{name}"


def test_box_cli_examples_use_current_argument_shapes():
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [SKILL_MD, *REFERENCES_DIR.glob("*.md")]
    )

    assert (
        "box collaborations:create <FOLDER_ID> <SERVICE_ACCOUNT_EMAIL> editor"
        not in docs
    )
    assert (
        "box collaborations:create <FOLDER_ID> collaborator@example.com editor"
        not in docs
    )
    assert "box files:download <FILE_ID> ./local-copy" not in docs
    assert (
        "box files:versions:download <FILE_ID> <VERSION_ID> ./older.pdf"
        not in docs
    )
    assert "box folders:update <FOLDER_ID> --parent-id" not in docs
    assert "box request GET " not in docs
    assert " -d '{" not in docs
    assert "box files:metadata:get <FILE_ID> enterprise properties" not in docs
    assert 'TOKEN="<access_token>"' not in docs


def test_ccg_config_template():
    template = TEMPLATES_DIR / "ccg-config.json.example"
    assert template.is_file()
    data = json.loads(template.read_text(encoding="utf-8"))
    settings = data.get("boxAppSettings") or {}
    assert "clientID" in settings
    assert "clientSecret" in settings
    assert "enterpriseID" in data


def test_hermes_section_order(skill_text: str):
    for heading in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in skill_text

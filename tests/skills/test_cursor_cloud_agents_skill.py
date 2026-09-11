"""Contract tests for the optional Cursor Cloud Agents skill.

These tests validate the local skill asset and helper construction only. They
never contact Cursor or transmit credentials.
"""

import base64
import importlib.util
import re
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT
    / "optional-skills"
    / "autonomous-ai-agents"
    / "cursor-cloud-agents"
    / "SKILL.md"
)
SCRIPT_PATH = (
    REPO_ROOT
    / "optional-skills"
    / "autonomous-ai-agents"
    / "cursor-cloud-agents"
    / "scripts"
    / "cursor_cloud_agent.py"
)
REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


def _frontmatter_and_body():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---")
    match = re.search(r"\n---\s*\n", content[3:])
    assert match, "frontmatter must close with ---"
    fm_text = content[3 : match.start() + 3]
    body = content[match.end() + 3 :]
    fields = {}
    for line in fm_text.splitlines():
        field = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
        if field:
            fields[field.group(1)] = field.group(2).strip().strip('"')
    return fields, body


def _load_helper():
    spec = importlib.util.spec_from_file_location("cursor_cloud_agent_test", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_file_and_helper_exist():
    assert SKILL_PATH.is_file()
    assert SCRIPT_PATH.is_file()


def test_frontmatter_required_fields():
    fm, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == "cursor-cloud-agents"
    assert not fm["author"].startswith("Hermes Agent")


def test_description_hardline():
    fm, _ = _frontmatter_and_body()
    description = fm["description"]
    assert len(description) <= 60
    assert description.endswith(".")
    assert description.count(".") == 1


def test_required_sections_are_ordered():
    _, body = _frontmatter_and_body()
    positions = [body.find(section) for section in REQUIRED_SECTIONS]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)


def test_steps_have_completion_criteria():
    _, body = _frontmatter_and_body()
    steps = re.findall(
        r"^### \d+\..*?(?=^### \d+\.|^## )",
        body,
        re.MULTILINE | re.DOTALL,
    )
    assert len(steps) == 8
    assert all("Completion criterion:" in step for step in steps)


def test_related_skills_resolve_in_repo():
    fm, _ = _frontmatter_and_body()
    related = re.search(r"related_skills:\s*\[(.*?)\]", SKILL_PATH.read_text())
    assert related
    for name in (item.strip() for item in related.group(1).split(",")):
        hits = list(REPO_ROOT.glob(f"skills/*/{name}/SKILL.md"))
        hits += list(REPO_ROOT.glob(f"optional-skills/*/{name}/SKILL.md"))
        assert hits, f"related skill does not resolve: {name}"


def test_skill_references_native_tools_and_no_local_paths():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert "`terminal`" in content
    assert "/Users/" not in content
    assert "/home/" not in content
    assert not re.search(r"[A-Z]:\\\\Users", content)


def test_helper_builds_safe_launch_payload(monkeypatch):
    helper = _load_helper()
    args = helper.build_parser().parse_args(
        [
            "launch",
            "--repo",
            "https://github.com/acme/demo",
            "--ref",
            "main",
            "--prompt",
            "Implement the approved change",
            "--auto-create-pr",
        ]
    )
    payload = helper.create_payload(args)
    assert payload == {
        "prompt": {"text": "Implement the approved change"},
        "repos": [{"url": "https://github.com/acme/demo", "startingRef": "main"}],
        "autoCreatePR": True,
    }
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)


def test_helper_uses_basic_auth_without_leaking_key(monkeypatch):
    helper = _load_helper()
    secret = "test-cursor-key"
    monkeypatch.setenv("CURSOR_API_KEY", secret)
    response = Mock()
    response.read.return_value = b'{"items": []}'
    response.headers = {"Content-Type": "application/json"}
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    with patch.object(helper, "urlopen", return_value=response) as mocked:
        assert helper.api_request("GET", "/v1/models") == {"items": []}
    request = mocked.call_args.args[0]
    encoded = request.get_header("Authorization").removeprefix("Basic ")
    assert base64.b64decode(encoded).decode() == secret + ":"
    assert secret not in repr(request)


def test_helper_uses_fixed_cursor_api_host(monkeypatch):
    helper = _load_helper()
    monkeypatch.setenv("CURSOR_API_KEY", "test-cursor-key")
    monkeypatch.setenv("CURSOR_API_BASE_URL", "https://attacker.example")
    response = Mock()
    response.read.return_value = b'{"items": []}'
    response.headers = {"Content-Type": "application/json"}
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    with patch.object(helper, "urlopen", return_value=response) as mocked:
        helper.api_request("GET", "/v1/models")
    assert mocked.call_args.args[0].full_url == "https://api.cursor.com/v1/models"


def test_helper_rejects_invalid_repository_urls():
    helper = _load_helper()
    with pytest.raises(Exception):
        helper.require_https_github_repo("https://example.com/acme/demo")
    with pytest.raises(Exception):
        helper.require_https_github_repo("https://github.com/acme")
    assert helper.require_https_github_repo("https://github.com/acme/demo.git") == "https://github.com/acme/demo.git"

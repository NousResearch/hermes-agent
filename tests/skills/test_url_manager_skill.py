"""
Tests for url-manager skill — behavioral, no source-text inspection.

Uses stdlib + pytest + unittest.mock. No live network calls.

Run: pytest tests/test_url_manager_skill.py -q
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Paths relative to skill directory ──
SKILL_DIR = Path(__file__).resolve().parent.parent.parent / "skills" / "productivity" / "url-manager"
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPTS_DIR = SKILL_DIR / "scripts"
FOOTPRINTS_PY = SCRIPTS_DIR / "footprints.py"


# ──────────────────────────────────────────────
# SKILL.md structure tests
# ──────────────────────────────────────────────


def test_skill_md_exists():
    """SKILL.md must be present in the skill directory."""
    assert SKILL_MD.exists(), f"SKILL.md not found at {SKILL_MD}"


def test_skill_md_has_required_sections():
    """SKILL.md must include all required sections per AGENTS.md standard."""
    content = SKILL_MD.read_text()
    required = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    for section in required:
        assert section in content, f"Missing required section: {section}"


def test_skill_md_has_frontmatter():
    """SKILL.md must have valid YAML frontmatter with required fields."""
    content = SKILL_MD.read_text()
    assert content.startswith("---"), "SKILL.md must start with YAML frontmatter"
    end = content.find("---", 3)
    assert end > 0, "SKILL.md frontmatter must have closing ---"
    frontmatter = content[3:end].strip()
    assert "name:" in frontmatter, "Frontmatter missing 'name'"
    assert "description:" in frontmatter, "Frontmatter missing 'description'"
    assert "author:" in frontmatter, "Frontmatter metadata missing 'author'"
    assert "version:" in frontmatter, "Frontmatter metadata missing 'version'"


def test_skill_md_description_length():
    """SKILL.md description must be ≤60 characters per AGENTS.md."""
    content = SKILL_MD.read_text()
    end = content.find("---", 3)
    frontmatter = content[3:end].strip()
    import re
    match = re.search(r"description:\s*(.+)", frontmatter)
    assert match, "Frontmatter missing 'description'"
    desc = match.group(1).strip()
    assert len(desc) <= 60, f"Description is {len(desc)} chars, must be ≤60: {desc}"


def test_skill_md_no_orphan_references():
    """SKILL.md must not reference files that are not shipped with the skill."""
    content = SKILL_MD.read_text()
    import re
    ref_links = re.findall(r"\]\(((?:\.\.?/)?references/[^)]+)\)", content)
    for link in ref_links:
        target = SKILL_DIR / link
        assert target.exists(), f"SKILL.md references missing file: {link}"


# ──────────────────────────────────────────────
# footprints.py existence tests
# ──────────────────────────────────────────────


def test_footprints_py_exists():
    """footprints.py must be shipped with the skill."""
    assert FOOTPRINTS_PY.exists(), f"footprints.py not found at {FOOTPRINTS_PY}"


def test_footprints_py_imports():
    """footprints.py must import as a module without errors (behavioral, no source parsing)."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    for key in list(sys.modules.keys()):
        if "footprints" in key:
            del sys.modules[key]
    try:
        import footprints
        assert footprints is not None
        # Verify it's a real module, not empty
        assert hasattr(footprints, "_get_token")
        assert hasattr(footprints, "api")
        assert hasattr(footprints, "agent_register")
    finally:
        sys.path.remove(str(SCRIPTS_DIR))


# ──────────────────────────────────────────────
# Token management logic tests (mocked)
# ──────────────────────────────────────────────


@pytest.fixture
def footprints_module():
    """
    Import footprints.py as a module for testing.
    The module has side-effects on import (reads env vars, checks files).
    We isolate it in a test fixture.
    """
    sys.path.insert(0, str(SCRIPTS_DIR))
    for key in list(sys.modules.keys()):
        if "footprints" in key:
            del sys.modules[key]
    import footprints

    yield footprints
    sys.path.remove(str(SCRIPTS_DIR))


def test_token_from_env(footprints_module, monkeypatch):
    """Token should be read from FOOTPRINTS_TOKEN env var first."""
    monkeypatch.setenv("FOOTPRINTS_TOKEN", "FA_test_token_123")
    token_file = footprints_module.TOKEN_FILE
    saved = None
    if os.path.exists(token_file):
        with open(token_file) as f:
            saved = f.read()
        os.remove(token_file)

    try:
        token = footprints_module._get_token()
        assert token == "FA_test_token_123"
    finally:
        if saved:
            os.makedirs(os.path.dirname(token_file), exist_ok=True)
            with open(token_file, "w") as f:
                f.write(saved)


def test_token_from_file(footprints_module, monkeypatch):
    """Token should fall back to .token file when env var is empty."""
    monkeypatch.delenv("FOOTPRINTS_TOKEN", raising=False)

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("FA_file_token_456")
        token_path = f.name

    monkeypatch.setattr(footprints_module, "TOKEN_FILE", token_path)

    try:
        token = footprints_module._get_token()
        assert token == "FA_file_token_456"
    finally:
        os.unlink(token_path)


def test_get_token_returns_empty_without_credentials(footprints_module, monkeypatch):
    """When no token in env or file, _get_token must return '' — never auto-register."""
    monkeypatch.delenv("FOOTPRINTS_TOKEN", raising=False)
    monkeypatch.setattr(footprints_module, "TOKEN_FILE", "/tmp/__nonexistent_token__")

    # _get_token must return empty string WITHOUT calling _raw_api
    mock_raw = MagicMock(return_value={"token": "should_not_be_called"})
    with patch.object(footprints_module, "_raw_api", mock_raw):
        token = footprints_module._get_token()
        assert token == "", "_get_token must return '' when no credentials exist"
        # Verify _raw_api was NEVER called (no auto-register!)
        mock_raw.assert_not_called()


def test_api_returns_error_without_token(footprints_module, monkeypatch):
    """api() without token must return error — no silent network call to /register."""
    monkeypatch.delenv("FOOTPRINTS_TOKEN", raising=False)
    monkeypatch.setattr(footprints_module, "TOKEN_FILE", "/tmp/__nonexistent_token__")

    # Patch _raw_api to detect any unintended /register call
    mock_raw = MagicMock(return_value={"ok": True})
    with patch.object(footprints_module, "_raw_api", mock_raw):
        result = footprints_module.api("/me")
        assert "error" in result, "api() without token must return error"
        # Must NOT have attempted any network call
        mock_raw.assert_not_called()


def test_api_adds_authorization(footprints_module, monkeypatch):
    """api() must attach Bearer token to requests."""
    monkeypatch.setenv("FOOTPRINTS_TOKEN", "FA_test_auth")

    def mock_raw_api(path, method="GET", data=None, no_auth=False):
        return {"ok": True}

    with patch.object(footprints_module, "_raw_api", side_effect=mock_raw_api):
        with patch.object(footprints_module, "_get_token", return_value="FA_test_auth"):
            result = footprints_module.api("/me")
            assert "error" not in result


def test_raw_api_no_auth_returns_error_without_token(footprints_module):
    """_raw_api must return error when no token and no_auth=False."""
    result = footprints_module._raw_api("/me", no_auth=False)
    assert "error" in result


def test_raw_api_register_no_auth_ok(footprints_module, monkeypatch):
    """_raw_api with no_auth=True should NOT require token (used by agent_register)."""
    monkeypatch.delenv("FOOTPRINTS_TOKEN", raising=False)
    import io

    mock_resp = io.BytesIO(json.dumps({"token": "FA_new"}).encode())
    mock_resp.status = 200

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = footprints_module._raw_api("/register", method="POST", no_auth=True)
        assert "token" in result


# ──────────────────────────────────────────────
# Command presence tests (behavioral, not source inspection)
# ──────────────────────────────────────────────


def test_all_commands_exist_in_module(footprints_module):
    """Every required command must exist as a callable in the imported module."""
    expected_funcs = [
        "add",
        "get_collection",
        "search",
        "list_collections",
        "update_collection",
        "batch_update_collections",
        "categories",
        "create_category",
        "category_sets",
        "create_category_set",
        "tags",
        "content_types",
        "create_shared_category",
        "create_invite_link",
        "join_shared_category",
        "add_to_shared_category",
        "remove_from_shared_category",
        "copy_collection",
        "me",
        "agent_magic_link",
        "agent_register",
    ]

    mod_attrs = set(dir(footprints_module))
    missing = [f for f in expected_funcs if f not in mod_attrs]
    assert not missing, f"Commands missing from footprints.py: {missing}"

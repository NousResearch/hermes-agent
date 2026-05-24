"""Tests for tools.domain_knowledge_loader."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the tool functions directly (not via registry to avoid side effects)
from tools.domain_knowledge_loader import (
    _VALID_DOMAINS,
    check_requirements,
    load_domain_knowledge,
    _resolve_vault_path,
)


@pytest.fixture
def mock_vault(tmp_path: Path) -> Path:
    """Create a temporary vault with domain notes."""
    vault = tmp_path / "vault"
    vault.mkdir()
    domains_dir = vault / "domains"
    domains_dir.mkdir()
    for d in ["frontend", "backend"]:
        (domains_dir / d).mkdir()

    # Create a frontend note
    (domains_dir / "frontend" / "react-hooks.md").write_text(
        "---\ntitle: React Hooks\norigin_project: proj-a\n---\nReact hooks are functions...\n"
    )

    return vault


def test_valid_domains_constant() -> None:
    assert "frontend" in _VALID_DOMAINS
    assert "backend" in _VALID_DOMAINS
    assert "business" in _VALID_DOMAINS
    assert "marketing" in _VALID_DOMAINS
    assert "finance" in _VALID_DOMAINS
    assert "people" in _VALID_DOMAINS
    assert "invalid-domain" not in _VALID_DOMAINS
    assert len(_VALID_DOMAINS) == 14


def test_check_requirements_missing() -> None:
    """With no vault anywhere, check_requirements should return False."""
    with patch.object(Path, "home", return_value=Path("/nonexistent")):
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": "/nonexistent"}, clear=False):
            # Clear the env var first
            with patch.dict(os.environ, {}, clear=False):
                if "OBSIDIAN_VAULT_PATH" in os.environ:
                    del os.environ["OBSIDIAN_VAULT_PATH"]
                assert check_requirements() is False


def test_load_domain_knowledge_valid(mock_vault: Path) -> None:
    with patch("tools.domain_knowledge_loader._resolve_vault_path", return_value=mock_vault):
        result_str = load_domain_knowledge(["frontend"])
        result = json.loads(result_str)
        assert result["success"] is True
        assert result["domains_valid"] == ["frontend"]
        assert result["domains_invalid"] == []
        assert result["notes_loaded"] == 1
        assert len(result["notes"]) == 1
        assert result["notes"][0]["title"] == "react-hooks"


def test_load_domain_knowledge_invalid_domain(mock_vault: Path) -> None:
    with patch("tools.domain_knowledge_loader._resolve_vault_path", return_value=mock_vault):
        result_str = load_domain_knowledge(["nonexistent-domain"])
        result = json.loads(result_str)
        assert result["success"] is True
        assert result["domains_valid"] == []
        assert result["domains_invalid"] == ["nonexistent-domain"]
        assert result["notes_loaded"] == 0


def test_load_domain_knowledge_mixed(mock_vault: Path) -> None:
    with patch("tools.domain_knowledge_loader._resolve_vault_path", return_value=mock_vault):
        result_str = load_domain_knowledge(["frontend", "invalid", "backend"])
        result = json.loads(result_str)
        assert result["domains_valid"] == ["frontend", "backend"]
        assert result["domains_invalid"] == ["invalid"]


def test_load_domain_knowledge_not_list(mock_vault: Path) -> None:
    with patch("tools.domain_knowledge_loader._resolve_vault_path", return_value=mock_vault):
        result_str = load_domain_knowledge("not-a-list")  # type: ignore
        result = json.loads(result_str)
        assert result["success"] is False
        assert "error" in result


def test_load_domain_knowledge_empty_dir(mock_vault: Path) -> None:
    """A domain dir with no notes (only README) should return 0 loaded."""
    with patch("tools.domain_knowledge_loader._resolve_vault_path", return_value=mock_vault):
        result_str = load_domain_knowledge(["security"])
        result = json.loads(result_str)
        assert result["success"] is True
        assert result["notes_loaded"] == 0


def test_resolve_vault_path_env() -> None:
    custom = "/tmp/custom-vault"
    with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": custom}):
        with patch("pathlib.Path.exists", return_value=True):
            result = _resolve_vault_path()
            assert str(result) == custom


def test_resolve_vault_path_standalone() -> None:
    with patch.dict(os.environ, {}, clear=True):
        def fake_exists(self) -> bool:
            if "ObsidianVault" in str(self):
                return True
            return False
        with patch("pathlib.Path.exists", fake_exists):
            with patch("pathlib.Path.home", return_value=Path("/home/user")):
                result = _resolve_vault_path()
                assert "ObsidianVault" in str(result)

"""Tests for external skill directory filtering."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestExternalDirConfig:
    """Test ExternalDirConfig parsing."""

    def test_parse_string_entry(self):
        """Test parsing a simple string entry."""
        from agent.skill_external_dirs import parse_external_dir_config
        
        config = parse_external_dir_config("/path/to/skills")
        
        assert config is not None
        assert config.path == "/path/to/skills"
        assert config.include == ["*"]
        assert config.exclude == []
        assert config.category_map == {}
        assert config.enabled is True
        
    def test_parse_dict_entry(self):
        """Test parsing a dict entry."""
        from agent.skill_external_dirs import parse_external_dir_config
        
        entry = {
            "path": "/path/to/skills",
            "include": ["anthropic-*", "mcp-*"],
            "exclude": ["*test*"],
            "category_map": {"anthropic-*": "anthropic-tools"},
            "enabled": True,
        }
        
        config = parse_external_dir_config(entry)
        
        assert config is not None
        assert config.path == "/path/to/skills"
        assert config.include == ["anthropic-*", "mcp-*"]
        assert config.exclude == ["*test*"]
        assert config.category_map == {"anthropic-*": "anthropic-tools"}
        assert config.enabled is True
        
    def test_parse_empty_string(self):
        """Test parsing an empty string returns None."""
        from agent.skill_external_dirs import parse_external_dir_config
        
        config = parse_external_dir_config("")
        assert config is None
        
    def test_parse_dict_without_path(self):
        """Test parsing a dict without path returns None."""
        from agent.skill_external_dirs import parse_external_dir_config
        
        config = parse_external_dir_config({"include": ["*"]})
        assert config is None
        
    def test_parse_disabled_entry(self):
        """Test parsing a disabled entry."""
        from agent.skill_external_dirs import parse_external_dir_config
        
        entry = {
            "path": "/path/to/skills",
            "enabled": False,
        }
        
        config = parse_external_dir_config(entry)
        
        assert config is not None
        assert config.enabled is False


class TestMatchesAny:
    """Test the matches_any function."""

    def test_matches_exact(self):
        """Test exact match."""
        from agent.skill_external_dirs import matches_any
        
        assert matches_any("anthropic-tools", ["anthropic-tools"]) is True
        
    def test_matches_glob(self):
        """Test glob pattern matching."""
        from agent.skill_external_dirs import matches_any
        
        assert matches_any("anthropic-tools", ["anthropic-*"]) is True
        assert matches_any("mcp-server", ["mcp-*"]) is True
        assert matches_any("test-skill", ["*test*"]) is True
        
    def test_no_match(self):
        """Test no match case."""
        from agent.skill_external_dirs import matches_any
        
        assert matches_any("anthropic-tools", ["mcp-*"]) is False
        assert matches_any("test-skill", ["prod-*"]) is False


class TestMatchesAllWords:
    """Test the matches_all_words function."""

    def test_matches_all_words(self):
        """Test matching all words in a phrase."""
        from agent.skill_external_dirs import matches_all_words
        
        assert matches_all_words("deploy to server", ["deploy", "server"]) is True
        assert matches_all_words("deploy to server", ["deploy", "database"]) is False
        
    def test_matches_glob_pattern(self):
        """Test matching with glob patterns."""
        from agent.skill_external_dirs import matches_all_words
        
        assert matches_all_words("anthropic-tools", ["anthropic-*"]) is True
        assert matches_all_words("mcp-server", ["mcp-*"]) is True


class TestDetermineCategory:
    """Test category determination from category_map."""

    def test_determine_category_match(self):
        """Test category determination with matching pattern."""
        from agent.skill_external_dirs import determine_category
        
        category_map = {
            "anthropic-*": "anthropic-tools",
            "mcp-*": "mcp-servers",
        }
        
        assert determine_category("anthropic-tools", category_map) == "anthropic-tools"
        assert determine_category("mcp-server", category_map) == "mcp-servers"
        
    def test_determine_category_no_match(self):
        """Test category determination with no matching pattern."""
        from agent.skill_external_dirs import determine_category
        
        category_map = {
            "anthropic-*": "anthropic-tools",
        }
        
        assert determine_category("mcp-server", category_map) == "external"


class TestExternalDirsIntegration:
    """Integration tests for external_dirs with skill_utils."""

    def test_simple_list_format(self):
        """Test backward-compatible simple list format."""
        from agent.skill_external_dirs import parse_external_dir_config
        
        # Simple string format should work
        config = parse_external_dir_config("/path/to/skills")
        assert config is not None
        assert config.path == "/path/to/skills"
        
    def test_advanced_dict_format(self):
        """Test advanced dict format with filtering."""
        from agent.skill_external_dirs import parse_external_dir_config
        
        entry = {
            "path": "/path/to/skills",
            "include": ["anthropic-*", "mcp-*"],
            "exclude": ["*test*"],
            "category_map": {"anthropic-*": "anthropic-tools"},
        }
        
        config = parse_external_dir_config(entry)
        assert config is not None
        assert config.include == ["anthropic-*", "mcp-*"]
        assert config.exclude == ["*test*"]
        
    def test_filtering_logic(self):
        """Test the filtering logic with include/exclude patterns."""
        from agent.skill_external_dirs import matches_any
        
        include = ["anthropic-*", "mcp-*"]
        exclude = ["*test*"]
        
        # Should be included
        assert matches_any("anthropic-tools", include) is True
        assert matches_any("mcp-server", include) is True
        
        # Should be excluded
        assert matches_any("test-skill", exclude) is True
        
        # Should be included and not excluded
        assert matches_any("anthropic-tools", include) is True
        assert not matches_any("anthropic-tools", exclude)

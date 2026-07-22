"""Tests for trigger-based skill auto-loading."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestTriggerIndex:
    """Test the TriggerIndex class."""

    def test_normalize(self):
        """Test trigger normalization."""
        from agent.skill_trigger_index import TriggerIndex
        
        index = TriggerIndex()
        
        # Basic normalization
        assert index._normalize("Deploy to SRV1") == "deploy to srv1"
        assert index._normalize("Create Incus VM!") == "create incus vm"
        assert index._normalize("  Backup Infrastructure  ") == "backup infrastructure"
        
    def test_match_exact_phrase(self):
        """Test exact phrase matching."""
        from agent.skill_trigger_index import TriggerIndex
        
        index = TriggerIndex()
        index.index = {
            "deploy to srv1": ["rapidwebs-infra-deployment"],
            "backup infrastructure": ["rapidwebs-backup-restore"],
            "create incus vm": ["incus-vm-management"],
        }
        index._built = True
        
        # Exact matches
        assert index.match("deploy to srv1") == ["rapidwebs-infra-deployment"]
        assert index.match("backup infrastructure") == ["rapidwebs-backup-restore"]
        assert index.match("create incus vm") == ["incus-vm-management"]
        
    def test_match_partial_phrase(self):
        """Test partial phrase matching."""
        from agent.skill_trigger_index import TriggerIndex
        
        index = TriggerIndex()
        index.index = {
            "deploy to srv1": ["rapidwebs-infra-deployment"],
            "backup infrastructure": ["rapidwebs-backup-restore"],
        }
        index._built = True
        
        # Partial matches (trigger phrase appears in input)
        assert index.match("I need to deploy to srv1 now") == ["rapidwebs-infra-deployment"]
        assert index.match("Can you backup infrastructure?") == ["rapidwebs-backup-restore"]
        
    def test_match_no_match(self):
        """Test no match case."""
        from agent.skill_trigger_index import TriggerIndex
        
        index = TriggerIndex()
        index.index = {
            "deploy to srv1": ["rapidwebs-infra-deployment"],
        }
        index._built = True
        
        # No match
        assert index.match("unrelated input") == []
        assert index.match("something else entirely") == []
        
    def test_match_multiple_triggers(self):
        """Test matching multiple triggers."""
        from agent.skill_trigger_index import TriggerIndex
        
        index = TriggerIndex()
        index.index = {
            "deploy to srv1": ["rapidwebs-infra-deployment"],
            "backup infrastructure": ["rapidwebs-backup-restore"],
        }
        index._built = True
        
        # Multiple matches
        result = index.match("deploy to srv1 and backup infrastructure")
        assert "rapidwebs-infra-deployment" in result
        assert "rapidwebs-backup-restore" in result
        
    def test_match_deduplication(self):
        """Test that duplicate matches are deduplicated."""
        from agent.skill_trigger_index import TriggerIndex
        
        index = TriggerIndex()
        index.index = {
            "deploy": ["skill-a", "skill-b"],
        }
        index._built = True
        
        # Single trigger matches multiple skills
        result = index.match("deploy")
        assert len(result) == 2
        assert "skill-a" in result
        assert "skill-b" in result


class TestAutoLoadSkills:
    """Test the auto_load_skills_for_turn function."""

    def test_auto_load_disabled(self):
        """Test auto-loading when disabled in config."""
        from agent.skill_trigger_index import auto_load_skills_for_turn
        
        with patch("hermes_cli.config.load_config") as mock_config:
            mock_config.return_value = {
                "skills": {"auto_load": {"enabled": False}}
            }
            
            result = auto_load_skills_for_turn("deploy to srv1")
            assert result == ""
            
    def test_auto_load_no_match(self):
        """Test auto-loading when no triggers match."""
        from agent.skill_trigger_index import auto_load_skills_for_turn
        
        with patch("hermes_cli.config.load_config") as mock_config:
            mock_config.return_value = {
                "skills": {"auto_load": {"enabled": True}}
            }
            
            with patch("agent.skill_trigger_index.match_skills_for_input") as mock_match:
                mock_match.return_value = []
                
                result = auto_load_skills_for_turn("unrelated input")
                assert result == ""
                
    def test_auto_load_with_match(self):
        """Test auto-loading when triggers match."""
        from agent.skill_trigger_index import auto_load_skills_for_turn
        
        with patch("hermes_cli.config.load_config") as mock_config:
            mock_config.return_value = {
                "skills": {"auto_load": {"enabled": True}}
            }
            
            with patch("agent.skill_trigger_index.match_skills_for_input") as mock_match:
                mock_match.return_value = ["test-skill"]
                
                with patch("tools.skills_tool.skill_view") as mock_view:
                    mock_view.return_value = json.dumps({
                        "success": True,
                        "content": "Test skill content"
                    })
                    
                    result = auto_load_skills_for_turn("test input")
                    assert "Auto-loaded Skill: test-skill" in result
                    assert "Test skill content" in result


class TestSkillTriggerIndexBuild:
    """Test building the trigger index from SKILL.md files."""

    def test_build_index(self):
        """Test building index from SKILL.md files."""
        from agent.skill_trigger_index import TriggerIndex
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test SKILL.md with triggers
            skill_dir = Path(tmpdir) / "test-skill"
            skill_dir.mkdir()
            
            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text('''---
name: test-skill
triggers:
  - "test trigger"
  - "another trigger"
---

# Test Skill

This is a test skill.
''')
            
            index = TriggerIndex(skills_dir=Path(tmpdir))
            index.build()
            
            # Check that triggers were indexed
            assert "test trigger" in index.index
            assert "another trigger" in index.index
            assert "test-skill" in index.skill_triggers
            
    def test_build_index_excludes_dirs(self):
        """Test that excluded directories are skipped."""
        from agent.skill_trigger_index import TriggerIndex
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create excluded directory
            node_modules = Path(tmpdir) / "node_modules"
            node_modules.mkdir()
            
            skill_dir = node_modules / "test-skill"
            skill_dir.mkdir()
            
            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text('''---
name: test-skill
triggers:
  - "test trigger"
---

# Test Skill
''')
            
            index = TriggerIndex(skills_dir=Path(tmpdir))
            index.build()
            
            # Check that excluded directory was skipped
            assert "test trigger" not in index.index


class TestGetTriggerIndexStats:
    """Test the get_trigger_index_stats function."""

    def test_get_stats(self):
        """Test getting trigger index statistics."""
        from agent.skill_trigger_index import get_trigger_index_stats, TriggerIndex
        
        with patch("agent.skill_trigger_index.get_trigger_index") as mock_get:
            mock_index = TriggerIndex()
            mock_index.index = {
                "trigger1": ["skill1"],
                "trigger2": ["skill2", "skill3"],
            }
            mock_index.skill_triggers = {
                "skill1": ["trigger1"],
                "skill2": ["trigger2"],
                "skill3": ["trigger2"],
            }
            mock_index._built = True
            mock_get.return_value = mock_index
            
            stats = get_trigger_index_stats()
            
            assert stats["total_triggers"] == 2
            assert stats["total_skills"] == 3
            assert "trigger1" in stats["triggers"]
            assert "trigger2" in stats["triggers"]

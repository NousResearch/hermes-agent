"""Tests for batch curator commands."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


class TestBatchPin:
    """Test batch pin commands."""

    def test_pin_batch_by_names(self):
        """Test pinning skills by comma-separated names."""
        from hermes_cli.curator import _cmd_pin_batch
        
        args = argparse.Namespace(
            batch="skill-a,skill-b,skill-c",
            by_usage=None,
            by_category=None,
            dry_run=False,
        )
        
        with patch("tools.skill_usage.is_agent_created") as mock_is_agent:
            mock_is_agent.return_value = True
            
            with patch("tools.skill_usage.set_pinned") as mock_set_pinned:
                result = _cmd_pin_batch(args)
                
                assert result == 0
                assert mock_set_pinned.call_count == 3
                
    def test_pin_batch_by_usage(self):
        """Test pinning skills by usage threshold."""
        from hermes_cli.curator import _cmd_pin_batch
        
        args = argparse.Namespace(
            batch=None,
            by_usage=20,
            by_category=None,
            dry_run=False,
        )
        
        with patch("tools.skill_usage.agent_created_report") as mock_report:
            mock_report.return_value = [
                {"name": "high-usage", "use_count": 25},
                {"name": "low-usage", "use_count": 5},
            ]
            
            with patch("tools.skill_usage.is_agent_created") as mock_is_agent:
                mock_is_agent.return_value = True
                
                with patch("tools.skill_usage.set_pinned") as mock_set_pinned:
                    result = _cmd_pin_batch(args)
                    
                    assert result == 0
                    # Only high-usage should be pinned
                    assert mock_set_pinned.call_count == 1
                    mock_set_pinned.assert_called_with("high-usage", True)
                    
    def test_pin_batch_dry_run(self):
        """Test dry run mode."""
        from hermes_cli.curator import _cmd_pin_batch
        
        args = argparse.Namespace(
            batch="skill-a,skill-b",
            by_usage=None,
            by_category=None,
            dry_run=True,
        )
        
        with patch("tools.skill_usage.is_agent_created") as mock_is_agent:
            mock_is_agent.return_value = True
            
            with patch("tools.skill_usage.set_pinned") as mock_set_pinned:
                result = _cmd_pin_batch(args)
                
                assert result == 0
                # No actual pinning in dry run
                assert mock_set_pinned.call_count == 0
                
    def test_pin_batch_skips_bundled(self):
        """Test that bundled skills are skipped."""
        from hermes_cli.curator import _cmd_pin_batch
        
        args = argparse.Namespace(
            batch="bundled-skill,agent-skill",
            by_usage=None,
            by_category=None,
            dry_run=False,
        )
        
        with patch("tools.skill_usage.is_agent_created") as mock_is_agent:
            def side_effect(name):
                return name == "agent-skill"
            mock_is_agent.side_effect = side_effect
            
            with patch("tools.skill_usage.set_pinned") as mock_set_pinned:
                result = _cmd_pin_batch(args)
                
                assert result == 0
                # Only agent-skill should be pinned
                assert mock_set_pinned.call_count == 1
                mock_set_pinned.assert_called_with("agent-skill", True)


class TestBatchUnpin:
    """Test batch unpin commands."""

    def test_unpin_batch_by_names(self):
        """Test unpinning skills by comma-separated names."""
        from hermes_cli.curator import _cmd_unpin_batch
        
        args = argparse.Namespace(
            batch="skill-a,skill-b",
            by_usage=None,
            by_category=None,
            dry_run=False,
        )
        
        with patch("tools.skill_usage.is_agent_created") as mock_is_agent:
            mock_is_agent.return_value = True
            
            with patch("tools.skill_usage.get_record") as mock_get_record:
                mock_get_record.return_value = {"pinned": True}
                
                with patch("tools.skill_usage.set_pinned") as mock_set_pinned:
                    result = _cmd_unpin_batch(args)
                    
                    assert result == 0
                    assert mock_set_pinned.call_count == 2
                    
    def test_unpin_batch_skips_not_pinned(self):
        """Test that unpinned skills are skipped."""
        from hermes_cli.curator import _cmd_unpin_batch
        
        args = argparse.Namespace(
            batch="skill-a,skill-b",
            by_usage=None,
            by_category=None,
            dry_run=False,
        )
        
        with patch("tools.skill_usage.is_agent_created") as mock_is_agent:
            mock_is_agent.return_value = True
            
            with patch("tools.skill_usage.get_record") as mock_get_record:
                def side_effect(name):
                    return {"pinned": name == "skill-a"}
                mock_get_record.side_effect = side_effect
                
                with patch("tools.skill_usage.set_pinned") as mock_set_pinned:
                    result = _cmd_unpin_batch(args)
                    
                    assert result == 0
                    # Only skill-a should be unpinned
                    assert mock_set_pinned.call_count == 1
                    mock_set_pinned.assert_called_with("skill-a", False)


class TestBatchArchive:
    """Test batch archive commands."""

    def test_archive_batch_by_names(self):
        """Test archiving skills by comma-separated names."""
        from hermes_cli.curator import _cmd_archive_batch
        
        args = argparse.Namespace(
            batch="skill-a,skill-b",
            by_usage=None,
            stale=None,
            dry_run=False,
            yes=True,
        )
        
        with patch("tools.skill_usage.is_agent_created") as mock_is_agent:
            mock_is_agent.return_value = True
            
            with patch("tools.skill_usage.get_record") as mock_get_record:
                mock_get_record.return_value = {"pinned": False, "state": "active"}
                
                with patch("tools.skill_usage.archive_skill") as mock_archive:
                    mock_archive.return_value = (True, "archived")
                    
                    result = _cmd_archive_batch(args)
                    
                    assert result == 0
                    assert mock_archive.call_count == 2
                    
    def test_archive_batch_skips_pinned(self):
        """Test that pinned skills are skipped."""
        from hermes_cli.curator import _cmd_archive_batch
        
        args = argparse.Namespace(
            batch="skill-a,skill-b",
            by_usage=None,
            stale=None,
            dry_run=False,
            yes=True,
        )
        
        with patch("tools.skill_usage.is_agent_created") as mock_is_agent:
            mock_is_agent.return_value = True
            
            with patch("tools.skill_usage.get_record") as mock_get_record:
                def side_effect(name):
                    return {"pinned": name == "skill-a", "state": "active"}
                mock_get_record.side_effect = side_effect
                
                with patch("tools.skill_usage.archive_skill") as mock_archive:
                    mock_archive.return_value = (True, "archived")
                    
                    result = _cmd_archive_batch(args)
                    
                    assert result == 0
                    # Only skill-b should be archived
                    assert mock_archive.call_count == 1
                    mock_archive.assert_called_with("skill-b")
                    
    def test_archive_batch_by_stale(self):
        """Test archiving skills by stale days."""
        from hermes_cli.curator import _cmd_archive_batch
        from hermes_cli.curator import _idle_days
        
        args = argparse.Namespace(
            batch=None,
            by_usage=None,
            stale=30,
            dry_run=False,
            yes=True,
        )
        
        with patch("tools.skill_usage.agent_created_report") as mock_report:
            mock_report.return_value = [
                {"name": "stale-skill", "last_activity_at": "2025-01-01T00:00:00"},
                {"name": "fresh-skill", "last_activity_at": "2026-07-20T00:00:00"},
            ]
            
            with patch("tools.skill_usage.is_agent_created") as mock_is_agent:
                mock_is_agent.return_value = True
                
                with patch("tools.skill_usage.get_record") as mock_get_record:
                    mock_get_record.return_value = {"pinned": False, "state": "active"}
                    
                    with patch("tools.skill_usage.archive_skill") as mock_archive:
                        mock_archive.return_value = (True, "archived")
                        
                        result = _cmd_archive_batch(args)
                        
                        assert result == 0


class TestUsageFilter:
    """Test usage filter command."""

    def test_usage_filter_min_usage(self):
        """Test filtering by minimum usage."""
        from hermes_cli.curator import _cmd_usage_batch
        
        args = argparse.Namespace(
            sort="activity",
            provenance=None,
            min_usage=10,
            max_usage=None,
            json=False,
        )
        
        with patch("tools.skill_usage.usage_report") as mock_report:
            mock_report.return_value = [
                {"name": "high-usage", "use_count": 15, "activity_count": 15},
                {"name": "low-usage", "use_count": 5, "activity_count": 5},
            ]
            
            result = _cmd_usage_batch(args)
            
            assert result == 0
            
    def test_usage_filter_json_output(self):
        """Test JSON output format."""
        from hermes_cli.curator import _cmd_usage_batch
        
        args = argparse.Namespace(
            sort="activity",
            provenance=None,
            min_usage=None,
            max_usage=None,
            json=True,
        )
        
        with patch("tools.skill_usage.usage_report") as mock_report:
            mock_report.return_value = [
                {"name": "test-skill", "use_count": 10, "activity_count": 10},
            ]
            
            result = _cmd_usage_batch(args)
            
            assert result == 0


class TestRegisterCLI:
    """Test CLI registration for batch commands."""

    def test_batch_commands_registered(self):
        """Test that batch commands are registered in the CLI."""
        from hermes_cli.curator import register_cli
        
        parser = argparse.ArgumentParser(prog="hermes curator")
        register_cli(parser)
        
        # Test that batch commands exist
        args = parser.parse_args(["pin-batch", "--batch", "skill-a"])
        assert hasattr(args, "func")
        
        args = parser.parse_args(["unpin-batch", "--batch", "skill-a"])
        assert hasattr(args, "func")
        
        args = parser.parse_args(["archive-batch", "--batch", "skill-a", "-y"])
        assert hasattr(args, "func")
        
        args = parser.parse_args(["usage-filter"])
        assert hasattr(args, "func")

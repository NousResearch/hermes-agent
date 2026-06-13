"""Tests for project-local skill discovery (.agent_context/skills/)."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project with .agent_context/skills/ structure."""
    project_dir = tmp_path / "test-project"
    project_dir.mkdir()
    
    # Create the project-local skills directory
    skills_dir = project_dir / ".agent_context" / "skills"
    skills_dir.mkdir(parents=True)
    
    # Create a test skill
    test_skill_dir = skills_dir / "test-project-skill"
    test_skill_dir.mkdir()
    
    skill_md = test_skill_dir / "SKILL.md"
    skill_md.write_text(
        '---\n'
        'name: test-project-skill\n'
        'description: "Test skill from project-local path"\n'
        'version: 1.0.0\n'
        '---\n\n'
        '# Test Project Skill\n\n'
        'This skill was discovered from `.agent_context/skills/`.\n'
    )
    
    return project_dir


@pytest.fixture
def temp_project_no_skills(tmp_path):
    """Create a temporary project without .agent_context/skills/."""
    project_dir = tmp_path / "test-project-no-skills"
    project_dir.mkdir()
    return project_dir


class TestGetProjectSkillsDir:
    """Tests for get_project_skills_dir() function."""
    
    def test_returns_path_when_exists(self, temp_project):
        """Should return the path when .agent_context/skills/ exists."""
        from agent.skill_utils import get_project_skills_dir
        
        with patch('agent.skill_utils.Path.cwd', return_value=temp_project):
            result = get_project_skills_dir()
            
        assert result is not None
        assert result == temp_project / ".agent_context" / "skills"
        assert result.exists()
    
    def test_returns_none_when_not_exists(self, temp_project_no_skills):
        """Should return None when .agent_context/skills/ doesn't exist."""
        from agent.skill_utils import get_project_skills_dir
        
        with patch('agent.skill_utils.Path.cwd', return_value=temp_project_no_skills):
            result = get_project_skills_dir()
            
        assert result is None
    
    def test_handles_cwd_error(self):
        """Should handle OSError when cwd() fails."""
        from agent.skill_utils import get_project_skills_dir
        
        with patch('agent.skill_utils.Path.cwd', side_effect=OSError("Permission denied")):
            result = get_project_skills_dir()
            
        assert result is None
    
    def test_handles_runtime_error(self):
        """Should handle RuntimeError when cwd() fails."""
        from agent.skill_utils import get_project_skills_dir
        
        with patch('agent.skill_utils.Path.cwd', side_effect=RuntimeError("No cwd")):
            result = get_project_skills_dir()
            
        assert result is None


class TestGetAllSkillsDirs:
    """Tests for get_all_skills_dirs() with project-local skills."""
    
    def test_includes_project_skills_when_exists(self, temp_project):
        """Should include project-local skills dir when it exists."""
        from agent.skill_utils import get_all_skills_dirs, get_skills_dir
        
        with patch('agent.skill_utils.Path.cwd', return_value=temp_project):
            dirs = get_all_skills_dirs()
        
        # Should include global skills dir
        assert get_skills_dir() in dirs
        
        # Should include project-local skills dir
        project_skills = temp_project / ".agent_context" / "skills"
        assert project_skills in dirs
    
    def test_excludes_project_skills_when_not_exists(self, temp_project_no_skills):
        """Should not include project-local skills dir when it doesn't exist."""
        from agent.skill_utils import get_all_skills_dirs, get_skills_dir
        
        with patch('agent.skill_utils.Path.cwd', return_value=temp_project_no_skills):
            dirs = get_all_skills_dirs()
        
        # Should include global skills dir
        assert get_skills_dir() in dirs
        
        # Should NOT include project-local skills dir
        project_skills = temp_project_no_skills / ".agent_context" / "skills"
        assert project_skills not in dirs
    
    def test_no_duplicates(self, temp_project):
        """Should not have duplicate entries in the skills dirs list."""
        from agent.skill_utils import get_all_skills_dirs
        
        with patch('agent.skill_utils.Path.cwd', return_value=temp_project):
            dirs = get_all_skills_dirs()
        
        # Check for duplicates
        assert len(dirs) == len(set(dirs))
    
    def test_order_preserved(self, temp_project):
        """Global skills dir should come first, then project-local."""
        from agent.skill_utils import get_all_skills_dirs, get_skills_dir
        
        with patch('agent.skill_utils.Path.cwd', return_value=temp_project):
            dirs = get_all_skills_dirs()
        
        # Global should be first
        assert dirs[0] == get_skills_dir()
        
        # Project-local should be after global
        project_skills = temp_project / ".agent_context" / "skills"
        if project_skills in dirs:
            global_idx = dirs.index(get_skills_dir())
            project_idx = dirs.index(project_skills)
            assert project_idx > global_idx


class TestSkillDiscovery:
    """Integration tests for skill discovery from project-local path."""
    
    def test_skill_view_loads_project_skill(self, temp_project):
        """Should be able to load a skill from project-local path."""
        from tools.skills_tool import skill_view
        
        with patch('agent.skill_utils.Path.cwd', return_value=temp_project):
            result = skill_view('test-project-skill')
        
        # Parse the JSON result
        import json
        data = json.loads(result)
        
        assert data['success'] is True
        assert data['name'] == 'test-project-skill'
        assert 'test skill from project-local path' in data['description'].lower()
    
    def test_skill_not_found_without_project_dir(self, temp_project_no_skills):
        """Should not find project-local skill when dir doesn't exist."""
        from tools.skills_tool import skill_view
        
        with patch('agent.skill_utils.Path.cwd', return_value=temp_project_no_skills):
            result = skill_view('test-project-skill')
        
        import json
        data = json.loads(result)
        
        assert data['success'] is False
        assert 'not found' in data['error'].lower()


class TestSkillListing:
    """Tests for skill listing with project-local skills."""
    
    def test_find_all_skills_includes_project(self, temp_project):
        """_find_all_skills should include skills from project-local path."""
        from tools.skills_tool import _find_all_skills
        
        with patch('agent.skill_utils.Path.cwd', return_value=temp_project):
            skills = _find_all_skills()
        
        # Find our test skill
        test_skill = None
        for skill in skills:
            if skill.get('name') == 'test-project-skill':
                test_skill = skill
                break
        
        assert test_skill is not None
        assert 'test skill from project-local path' in test_skill.get('description', '').lower()

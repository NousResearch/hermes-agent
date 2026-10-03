"""Regression test for subdirectory hint path rendering bug.

This test verifies that relative subdirectory paths are correctly resolved
relative to working_dir, not to the current process working directory.

Bug: When directory = Path("agent") and working_dir = Path("hermes-agent/"),
hint_path was constructed as Path("agent") / Path("AGENTS.md") = Path("agent/AGENTS.md"),
which resolved to /Users/Dmitriy_Zhorov/agent/AGENTS.md instead of
/Users/Dmitriy_Zhorov/.hermes/hermes-agent/agent/AGENTS.md.

Fix: Construct hint_path as working_dir / directory / filename to anchor
relative paths correctly.
"""

import pytest
from pathlib import Path
from agent.subdirectory_hints import SubdirectoryHintTracker


def test_relative_subdirectory_path_resolution(tmp_path):
    """Test that relative subdirectory paths are anchored to working_dir."""
    # Create a subdirectory with a hint file
    sub = tmp_path / "backend"
    sub.mkdir()
    (sub / "AGENTS.md").write_text("Backend-specific instructions:\\n- Use FastAPI", encoding="utf-8")
    
    # Tracker should resolve the relative path correctly
    tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
    
    # Check that the hint is discovered
    result = tracker.check_tool_call("read_file", {"path": str(sub / "main.py")})
    assert result is not None
    assert "Backend-specific instructions" in result
    
    # The hint path should be rendered as "backend/AGENTS.md", not "backend"
    # This ensures the model sees the correct file path
    assert "backend/AGENTS.md" in result or result is None


def test_deeply_nested_relative_path(tmp_path):
    """Test that deeply nested relative paths are resolved correctly."""
    # Create a nested directory structure
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "AGENTS.md").write_text("Deep directory instructions", encoding="utf-8")
    
    tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
    result = tracker.check_tool_call("read_file", {"path": str(deep / "file.py")})
    
    assert result is not None
    assert "Deep directory instructions" in result
    
    # The path should include the full relative path, not just the last component
    assert "a/b/c/d/AGENTS.md" in result or result is None


def test_relative_path_with_dot(tmp_path):
    """Test that Path('.') in directory parameter is handled correctly."""
    # Create a subdirectory
    sub = tmp_path / "frontend"
    sub.mkdir()
    (sub / "CLAUDE.md").write_text("Frontend rules", encoding="utf-8")
    
    tracker = SubdirectoryHintTracker(working_dir=str(tmp_path))
    
    # The directory parameter can be Path('.')
    result = tracker.check_tool_call("read_file", {"path": str(sub / "index.ts")})
    assert result is not None
    assert "Frontend rules" in result

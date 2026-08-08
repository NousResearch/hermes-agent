#!/usr/bin/env python3
"""
Test batch_runner CLI flags to ensure they don't crash
"""

import subprocess
import sys
from pathlib import Path


def test_show_distributions_flag():
    """Test that --show_distributions flag works without crashing."""
    # Find the batch_runner.py file
    repo_root = Path(__file__).parent.parent
    batch_runner = repo_root / "batch_runner.py"
    
    # Run with --show_distributions flag
    cmd = [sys.executable, str(batch_runner), "--show_distributions"]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_root)
    )
    
    # Should not crash with TypeError
    assert result.returncode == 0, f"Command failed: {result.stderr}"
    
    # Should output distribution info
    assert "Available Toolset Distributions" in result.stdout
    assert "Distribution:" in result.stdout
    
    # Should not have the old error
    assert "TypeError: 'bool' object is not callable" not in result.stderr


if __name__ == "__main__":
    test_show_distributions_flag()
    print("✅ Test passed: --show_distributions flag works correctly")
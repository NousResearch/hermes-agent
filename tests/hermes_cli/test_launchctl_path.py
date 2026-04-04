"""Tests for launchctl path resolution."""

import os
import sys
import pytest
from unittest.mock import patch

# Skip on non-macOS
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


def test_get_launchctl_path_uses_which_first():
    """Should use shutil.which() result when available."""
    from hermes_cli.gateway import get_launchctl_path
    
    # Clear the cache
    get_launchctl_path.cache_clear()
    
    with patch("shutil.which", return_value="/opt/custom/bin/launchctl"):
        result = get_launchctl_path()
        assert result == "/opt/custom/bin/launchctl"
    
    get_launchctl_path.cache_clear()


def test_get_launchctl_path_fallback_to_bin():
    """Should fall back to /bin/launchctl when which() fails."""
    from hermes_cli.gateway import get_launchctl_path
    
    get_launchctl_path.cache_clear()
    
    with patch("shutil.which", return_value=None), \
         patch("os.path.isfile", return_value=True), \
         patch("os.access", return_value=True):
        result = get_launchctl_path()
        assert result in ("/bin/launchctl", "/usr/bin/launchctl")
    
    get_launchctl_path.cache_clear()


def test_get_launchctl_path_caches_result():
    """Should cache the result and not call which() repeatedly."""
    from hermes_cli.gateway import get_launchctl_path
    
    get_launchctl_path.cache_clear()
    
    with patch("shutil.which", return_value="/test/launchctl") as mock_which:
        # Call twice
        result1 = get_launchctl_path()
        result2 = get_launchctl_path()
        
        # Should only call which once due to caching
        assert mock_which.call_count == 1
        assert result1 == result2 == "/test/launchctl"
    
    get_launchctl_path.cache_clear()

"""Email platform toolset hardening tests."""
from __future__ import annotations

from unittest.mock import patch

from hermes_cli.tools_config import _get_platform_tools


def test_email_toolsets_restricted_by_default():
    config = {
        "platform_toolsets": {
            "email": [
                "terminal",
                "browser",
                "web",
                "memory",
                "session_search",
                "todo",
                "clarify",
                "search",
                "vision",
                "skills",
                "messaging",
            ]
        }
    }

    enabled = _get_platform_tools(config, "email")

    assert "memory" in enabled
    assert "session_search" in enabled
    assert "terminal" not in enabled
    assert "browser" not in enabled
    assert "web" not in enabled


def test_email_toolset_restriction_can_be_disabled():
    config = {
        "platform_toolsets": {
            "email": ["terminal", "memory", "search"]
        }
    }

    with patch.dict("os.environ", {"EMAIL_RESTRICT_TOOLS": "0"}):
        enabled = _get_platform_tools(config, "email")

    assert "terminal" in enabled
    assert "memory" in enabled
    assert "search" in enabled

"""Unit tests verifying setup.status and setup.runtime_check profile scoping (#77006)."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tui_gateway.server import _methods, _profile_scoped


def test_setup_status_and_runtime_check_are_profile_scoped(tmp_path, monkeypatch):
    """Verify setup.status and setup.runtime_check resolve credentials under params['profile']."""
    # Launch profile dir (empty)
    launch_home = tmp_path / "launch_profile"
    launch_home.mkdir()

    # Target profile B dir (has custom config/credentials)
    profile_b_home = tmp_path / "profiles" / "profile_b"
    profile_b_home.mkdir(parents=True)
    (profile_b_home / ".env").write_text("OPENAI_API_KEY=sk-test-key-123\n", encoding="utf-8")
    (profile_b_home / "config.yaml").write_text("provider: openai-api\nmodel:\n  default: 'openai-api/gpt-4o'\n", encoding="utf-8")

    # Set launch profile as default HERMES_HOME
    token = set_hermes_home_override(launch_home)
    try:
        def mock_get_profile_dir(name):
            if name == "profile_b":
                return str(profile_b_home)
            return str(tmp_path / "profiles" / name)

        with patch("hermes_cli.profiles.get_profile_dir", side_effect=mock_get_profile_dir):
            setup_status = _methods["setup.status"]
            setup_runtime_check = _methods["setup.runtime_check"]

            # 1. Without profile param: evaluates launch profile (empty -> not configured / no key)
            res_status_launch = setup_status(1, {})
            assert res_status_launch.get("result", {}).get("provider_configured") is False

            res_check_launch = setup_runtime_check(2, {})
            assert res_check_launch.get("result", {}).get("ok") is False

            # 2. With profile="profile_b": evaluates profile_b context (configured -> ok=True)
            res_status_b = setup_status(3, {"profile": "profile_b"})
            assert res_status_b.get("result", {}).get("provider_configured") is True

            res_check_b = setup_runtime_check(4, {"profile": "profile_b"})
            assert res_check_b.get("result", {}).get("ok") is True
            assert res_check_b.get("result", {}).get("provider") == "openai-api"
    finally:
        reset_hermes_home_override(token)

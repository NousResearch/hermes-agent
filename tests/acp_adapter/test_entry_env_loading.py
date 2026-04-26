"""Tests for acp_adapter/entry.py project_env loading.

Covers secondary finding in https://github.com/NousResearch/hermes-agent/issues/15914:
acp_adapter/entry.py was the only entry point missing project_env in load_hermes_dotenv().
"""

import json
import sys
from pathlib import Path
from unittest import mock


def test_load_env_passes_project_env(monkeypatch, tmp_path):
    """Verify _load_env passes project_env to load_hermes_dotenv."""
    # Set up fake hermes home
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Mock load_hermes_dotenv to capture what args were passed
    from unittest.mock import MagicMock
    from hermes_cli.env_loader import load_hermes_dotenv as real_loader

    mock_calls = []
    def mock_loader(*, hermes_home=None, project_env=None):
        mock_calls.append({"hermes_home": hermes_home, "project_env": project_env})
        return real_loader(hermes_home=hermes_home, project_env=project_env)

    # Patch at the module where entry.py imports it
    import hermes_cli.env_loader
    original = hermes_cli.env_loader.load_hermes_dotenv
    hermes_cli.env_loader.load_hermes_dotenv = mock_loader

    try:
        # Import and call _load_env from acp_adapter.entry
        import importlib
        import acp_adapter.entry as entry_module
        # Reload to pick up patched env_loader
        importlib.reload(entry_module)
        entry_module._load_env()

        assert len(mock_calls) == 1, f"Expected 1 call, got {len(mock_calls)}"
        call = mock_calls[0]
        assert call["hermes_home"] is not None, "hermes_home not passed"
        assert call["project_env"] is not None, "project_env not passed (was missing before fix)"
        # project_env should be the project root's .env (Path type)
        assert isinstance(call["project_env"], Path), "project_env should be a Path"
        assert call["project_env"].name == ".env", "project_env should point to .env"
    finally:
        hermes_cli.env_loader.load_hermes_dotenv = original
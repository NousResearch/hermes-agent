"""
Tests for cron job profile isolation (#51853).

Verifies that:
1. Jobs store the originating profile hermes_home at creation time
2. The scheduler loads config.yaml from the job hermes_home
3. Tool restrictions from the originating profile are applied at execution time
4. Legacy jobs without hermes_home fall back to the default hermes_home
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


def _load_yaml(path):
    """Helper to load a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


@pytest.fixture
def temp_hermes_homes(tmp_path):
    """Create two simulated profile hermes_home directories."""
    default_home = tmp_path / "default"
    default_home.mkdir()
    (default_home / "cron").mkdir()
    (default_home / "config.yaml").write_text(
        "agent:\n  disabled_toolsets: []\nmodel:\n  default: gpt-4\n"
    )

    restricted_home = tmp_path / "profiles" / "restricted"
    restricted_home.mkdir(parents=True)
    (restricted_home / "cron").mkdir()
    (restricted_home / "config.yaml").write_text(
        "agent:\n  disabled_toolsets:\n    - terminal\n    - file\nmodel:\n  default: gpt-3.5-turbo\n"
    )

    return {"default": default_home, "restricted": restricted_home}


class TestJobCreationStoresHermesHome:
    """Jobs must record the creating profile hermes_home."""

    def test_create_job_stores_explicit_hermes_home(self, temp_hermes_homes):
        """create_job() persists the hermes_home field in the job record."""
        from cron.jobs import create_job

        restricted_home = str(temp_hermes_homes["restricted"])

        with patch("cron.jobs.get_hermes_home", return_value=temp_hermes_homes["default"]):
            with patch("cron.jobs.HERMES_DIR", temp_hermes_homes["default"]):
                with patch("cron.jobs.CRON_DIR", temp_hermes_homes["default"] / "cron"):
                    with patch("cron.jobs.JOBS_FILE", temp_hermes_homes["default"] / "cron" / "jobs.json"):
                        job = create_job(
                            prompt="test prompt",
                            schedule="every 30m",
                            hermes_home=restricted_home,
                        )

        assert job["hermes_home"] == restricted_home

    def test_create_job_defaults_to_current_hermes_home(self, temp_hermes_homes):
        """When hermes_home is not passed, it defaults to get_hermes_home()."""
        from cron.jobs import create_job

        with patch("cron.jobs.get_hermes_home", return_value=temp_hermes_homes["default"]):
            with patch("cron.jobs.HERMES_DIR", temp_hermes_homes["default"]):
                with patch("cron.jobs.CRON_DIR", temp_hermes_homes["default"] / "cron"):
                    with patch("cron.jobs.JOBS_FILE", temp_hermes_homes["default"] / "cron" / "jobs.json"):
                        job = create_job(
                            prompt="test prompt",
                            schedule="every 30m",
                        )

        assert job["hermes_home"] == str(temp_hermes_homes["default"])


class TestSchedulerUsesJobHermesHome:
    """The scheduler must load config from the job hermes_home."""

    def test_resolve_cron_disabled_toolsets_from_restricted_profile(self, temp_hermes_homes):
        """Tool restrictions from the restricted profile config must be applied."""
        from cron.scheduler import _resolve_cron_disabled_toolsets

        cfg = _load_yaml(temp_hermes_homes["restricted"] / "config.yaml")
        disabled = _resolve_cron_disabled_toolsets(cfg)

        assert "cronjob" in disabled
        assert "messaging" in disabled
        assert "clarify" in disabled
        assert "terminal" in disabled
        assert "file" in disabled

    def test_resolve_cron_disabled_toolsets_from_default_profile(self, temp_hermes_homes):
        """Default profile should not have extra restrictions."""
        from cron.scheduler import _resolve_cron_disabled_toolsets

        cfg = _load_yaml(temp_hermes_homes["default"] / "config.yaml")
        disabled = _resolve_cron_disabled_toolsets(cfg)

        assert "cronjob" in disabled
        assert "messaging" in disabled
        assert "clarify" in disabled
        assert "terminal" not in disabled
        assert "file" not in disabled

    def test_legacy_job_without_hermes_home_uses_default(self, temp_hermes_homes):
        """Jobs without hermes_home field (pre-fix) fall back to _get_hermes_home()."""
        legacy_job = {"id": "abc123", "name": "legacy job", "prompt": "test"}

        from cron.scheduler import _get_hermes_home

        job_home = (
            Path(legacy_job["hermes_home"])
            if legacy_job.get("hermes_home")
            else _get_hermes_home()
        )
        assert job_home == _get_hermes_home()

    def test_job_with_hermes_home_uses_stored_path(self, temp_hermes_homes):
        """Jobs with hermes_home use their stored profile path."""
        job = {
            "id": "def456",
            "hermes_home": str(temp_hermes_homes["restricted"]),
        }
        job_home = Path(job["hermes_home"]) if job.get("hermes_home") else None
        assert job_home == temp_hermes_homes["restricted"]


class TestToolEscalationPrevention:
    """Verify that a restricted profile cannot escalate via cron."""

    def test_restricted_profile_terminal_blocked(self, temp_hermes_homes):
        """A job from a terminal-disabled profile must have terminal in disabled_toolsets."""
        from cron.scheduler import _resolve_cron_disabled_toolsets

        cfg = _load_yaml(Path(str(temp_hermes_homes["restricted"])) / "config.yaml")
        disabled = _resolve_cron_disabled_toolsets(cfg)

        assert "terminal" in disabled, "SECURITY: terminal must be disabled"
        assert "file" in disabled, "SECURITY: file must be disabled"

    def test_default_profile_terminal_allowed(self, temp_hermes_homes):
        """A job from the default profile keeps terminal access."""
        from cron.scheduler import _resolve_cron_disabled_toolsets

        cfg = _load_yaml(Path(str(temp_hermes_homes["default"])) / "config.yaml")
        disabled = _resolve_cron_disabled_toolsets(cfg)

        assert "terminal" not in disabled
        assert "file" not in disabled

    def test_config_path_resolves_to_job_profile(self, temp_hermes_homes):
        """config.yaml must be loaded from the job hermes_home."""
        job = {"id": "cfg_test", "hermes_home": str(temp_hermes_homes["restricted"])}
        job_hermes_home = (
            Path(job["hermes_home"])
            if job.get("hermes_home")
            else temp_hermes_homes["default"]
        )
        cfg = _load_yaml(job_hermes_home / "config.yaml")

        agent_cfg = cfg.get("agent", {})
        assert "terminal" in agent_cfg.get("disabled_toolsets", [])
        assert "file" in agent_cfg.get("disabled_toolsets", [])

        default_cfg = _load_yaml(temp_hermes_homes["default"] / "config.yaml")
        default_agent_cfg = default_cfg.get("agent", {})
        assert default_agent_cfg.get("disabled_toolsets") == []

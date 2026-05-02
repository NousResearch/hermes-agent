"""Tests for load_hermes_dotenv 12-factor env precedence (#18705)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


class TestLoadHermesDotenvOverride:
    """Verify HERMES_DOTENV_OVERRIDE toggle behavior."""

    def test_default_no_override(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """By default, .env should NOT override existing env vars (12-factor)."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_PRECEDENCE_KEY=from_dotenv\n")

        # Simulate a runtime-injected secret.
        monkeypatch.setenv("TEST_PRECEDENCE_KEY", "from_runtime")
        # Ensure the toggle is not set.
        monkeypatch.delenv("HERMES_DOTENV_OVERRIDE", raising=False)

        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv(hermes_home=tmp_path)

        assert os.environ["TEST_PRECEDENCE_KEY"] == "from_runtime"

    def test_override_opt_in(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """HERMES_DOTENV_OVERRIDE=1 restores legacy override behavior."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_PRECEDENCE_KEY=from_dotenv\n")

        monkeypatch.setenv("TEST_PRECEDENCE_KEY", "from_runtime")
        monkeypatch.setenv("HERMES_DOTENV_OVERRIDE", "1")

        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv(hermes_home=tmp_path)

        assert os.environ["TEST_PRECEDENCE_KEY"] == "from_dotenv"

    def test_no_runtime_var_fills_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no runtime var exists, .env should fill the value."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_FILL_KEY=from_dotenv\n")

        monkeypatch.delenv("TEST_FILL_KEY", raising=False)
        monkeypatch.delenv("HERMES_DOTENV_OVERRIDE", raising=False)

        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv(hermes_home=tmp_path)

        assert os.environ["TEST_FILL_KEY"] == "from_dotenv"

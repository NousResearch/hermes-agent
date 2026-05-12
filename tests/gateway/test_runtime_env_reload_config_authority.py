"""Regression tests for gateway per-turn env reload preserving config authority.

Issue #19158: startup bridges config.yaml agent.max_turns into
HERMES_MAX_ITERATIONS, but a later per-turn load_dotenv(..., override=True)
can restore a stale .env HERMES_MAX_ITERATIONS value before the next turn.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import yaml

from gateway import run as gateway_run


def test_reload_runtime_env_preserves_config_max_turns(tmp_path: Path, monkeypatch) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"agent": {"max_turns": 9000}}),
        encoding="utf-8",
    )
    (hermes_home / ".env").write_text(
        "HERMES_MAX_ITERATIONS=90\nOPENROUTER_API_KEY=fresh-key\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "9000")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    gateway_run._reload_runtime_env_preserving_config_authority()

    assert os.environ["OPENROUTER_API_KEY"] == "fresh-key"
    assert os.environ["HERMES_MAX_ITERATIONS"] == "9000"


def test_reload_runtime_env_keeps_env_max_iterations_when_config_omits_key(
    tmp_path: Path, monkeypatch
) -> None:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(yaml.safe_dump({"agent": {}}), encoding="utf-8")
    (hermes_home / ".env").write_text("HERMES_MAX_ITERATIONS=123\n", encoding="utf-8")

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)

    gateway_run._reload_runtime_env_preserving_config_authority()

    assert os.environ["HERMES_MAX_ITERATIONS"] == "123"


def test_max_iterations_env_invalid_value_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "not-a-number")

    with patch.object(gateway_run.logger, "warning") as warning:
        assert gateway_run._max_iterations_env() == 90

    warning.assert_called_once()
    assert "Invalid HERMES_MAX_ITERATIONS" in warning.call_args[0][0]


def test_max_iterations_env_strips_whitespace_and_uses_valid_value(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", " 42 ")

    with patch.object(gateway_run.logger, "warning") as warning:
        assert gateway_run._max_iterations_env() == 42

    warning.assert_not_called()

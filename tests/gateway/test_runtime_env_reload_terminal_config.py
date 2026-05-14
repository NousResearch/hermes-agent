"""Regression tests for gateway .env reload preserving terminal config authority."""

import os

import yaml


def test_runtime_env_reload_reapplies_terminal_backend_from_config(monkeypatch, tmp_path):
    """A stale .env TERMINAL_ENV must not override config.yaml after reload.

    Gateway sessions reload ~/.hermes/.env between turns to pick up fresh
    credentials.  If that reload restores TERMINAL_ENV=docker, terminal tools
    can keep executing in Docker even though config.yaml says local.  The reload
    path must re-bridge terminal.backend from config.yaml after loading .env.
    """
    from gateway import run as gateway_run

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {"max_turns": 90},
                "terminal": {
                    "backend": "local",
                    "cwd": ".",
                    "timeout": 180,
                    "docker_mount_cwd_to_workspace": False,
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_load_dotenv(*args, **kwargs):
        os.environ["TERMINAL_ENV"] = "docker"
        os.environ["TERMINAL_DOCKER_IMAGE"] = "nikolaik/python-nodejs:python3.11-nodejs20"
        os.environ["TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE"] = "true"

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "load_hermes_dotenv", fake_load_dotenv)
    monkeypatch.setenv("TERMINAL_ENV", "local")

    gateway_run._reload_runtime_env_preserving_config_authority()

    assert os.environ["TERMINAL_ENV"] == "local"
    assert os.environ["TERMINAL_TIMEOUT"] == "180"
    assert os.environ["TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE"] == "False"

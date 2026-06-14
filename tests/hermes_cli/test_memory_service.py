"""Tests for `hermes memory service ...` lifecycle helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import memory_setup
from hermes_cli.subcommands.memory import build_memory_parser


@pytest.fixture
def hindsight_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "hindsight").mkdir(parents=True)
    (hermes_home / "hindsight" / "config.json").write_text(
        json.dumps({"mode": "local_external", "api_url": "http://127.0.0.1:9177"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def test_memory_parser_accepts_service_install_hindsight():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_memory_parser(subparsers, cmd_memory=lambda args: None)

    args = parser.parse_args(["memory", "service", "install", "hindsight"])

    assert args.memory_command == "service"
    assert args.service_command == "install"
    assert args.provider == "hindsight"


def test_memory_command_routes_service_subcommand():
    args = SimpleNamespace(memory_command="service", service_command="status", provider="hindsight")

    with patch("hermes_cli.memory_service.memory_service_command") as service_command:
        memory_setup.memory_command(args)

    service_command.assert_called_once_with(args)


def test_hindsight_plan_reads_profile_config(hindsight_home):
    from hermes_cli.memory_service import build_hindsight_service_plan

    plan = build_hindsight_service_plan(hermes_home=hindsight_home, executable=Path("/opt/hindsight-api"))

    assert plan.provider == "hindsight"
    assert plan.mode == "local_external"
    assert plan.host == "127.0.0.1"
    assert plan.port == 9177
    assert plan.config_path == hindsight_home / "hindsight" / "config.json"
    assert plan.wrapper_path == hindsight_home / "services" / "hindsight" / "start-hindsight.sh"
    assert plan.stdout_path == hindsight_home / "logs" / "hindsight-service.log"
    assert plan.stderr_path == hindsight_home / "logs" / "hindsight-service.err.log"


def test_hindsight_plan_rejects_non_external_modes(tmp_path, monkeypatch):
    from hermes_cli.memory_service import UnsupportedMemoryService

    hermes_home = tmp_path / ".hermes"
    (hermes_home / "hindsight").mkdir(parents=True)
    (hermes_home / "hindsight" / "config.json").write_text(
        json.dumps({"mode": "cloud", "api_url": "https://api.example.test"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli.memory_service import build_hindsight_service_plan

    with pytest.raises(UnsupportedMemoryService, match="local_external"):
        build_hindsight_service_plan(hermes_home=hermes_home, executable=Path("/opt/hindsight-api"))


def test_launchd_wrapper_uses_paths_without_embedding_secrets(hindsight_home):
    from hermes_cli.memory_service import build_hindsight_service_plan, render_launchd_wrapper

    plan = build_hindsight_service_plan(
        hermes_home=hindsight_home,
        executable=Path("/opt/hindsight/bin/hindsight-api"),
        env_file=Path("/Users/test/.hindsight/profiles/hermes.env"),
    )

    rendered = render_launchd_wrapper(plan)

    assert str(hindsight_home) in rendered
    assert "/Users/test/.hindsight/profiles/hermes.env" in rendered
    assert "exec \"$HINDSIGHT_API\" --host \"$HOST\" --port \"$PORT\"" in rendered
    assert "API_KEY=" not in rendered
    assert "TOKEN=" not in rendered
    assert "SECRET=" not in rendered


def test_launchd_plist_points_to_wrapper_and_logs(hindsight_home):
    from hermes_cli.memory_service import build_hindsight_service_plan, render_launchd_plist

    plan = build_hindsight_service_plan(hermes_home=hindsight_home, executable=Path("/opt/hindsight-api"))

    plist = render_launchd_plist(plan)

    assert "<string>ai.hermes.hindsight</string>" in plist
    assert f"<string>{plan.wrapper_path}</string>" in plist
    assert f"<string>{plan.stdout_path}</string>" in plist
    assert f"<string>{plan.stderr_path}</string>" in plist
    assert "HERMES_HOME" in plist
    assert str(hindsight_home) in plist


def test_named_profile_launchd_label_is_profile_scoped(tmp_path):
    from hermes_cli.memory_service import build_hindsight_service_plan

    hermes_home = tmp_path / "profiles" / "kairos-orchestrator"
    (hermes_home / "hindsight").mkdir(parents=True)
    (hermes_home / "hindsight" / "config.json").write_text(
        json.dumps({"mode": "local_external", "api_url": "http://127.0.0.1:9177"}),
        encoding="utf-8",
    )

    plan = build_hindsight_service_plan(hermes_home=hermes_home, executable=Path("/opt/hindsight-api"))

    assert plan.label == "ai.hermes.hindsight.kairos-orchestrator"

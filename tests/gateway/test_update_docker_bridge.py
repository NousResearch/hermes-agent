"""Regression tests for the Docker-aware Telegram /update handoff."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import Platform
from gateway.slash_commands import GatewaySlashCommandsMixin


def _event() -> SimpleNamespace:
    return SimpleNamespace(
        source=SimpleNamespace(
            platform=Platform.TELEGRAM,
            chat_id="probe-chat",
            chat_type="private",
            user_id="probe-user",
            thread_id=None,
        ),
        message_id="probe-message",
    )


def _runner() -> GatewaySlashCommandsMixin:
    runner = object.__new__(GatewaySlashCommandsMixin)
    runner._UPDATE_ALLOWED_PLATFORMS = {Platform.TELEGRAM}
    return runner


def _fake_gateway_file(tmp_path: Path) -> str:
    gateway_dir = tmp_path / "project" / "gateway"
    gateway_dir.mkdir(parents=True)
    gateway_file = gateway_dir / "slash_commands.py"
    gateway_file.touch()
    return str(gateway_file)


def test_docker_update_records_an_atomic_host_request(tmp_path, monkeypatch):
    request_path = tmp_path / "hermes" / ".host_update_request.json"
    request_path.parent.mkdir()
    monkeypatch.setenv("HERMES_HOST_UPDATE_REQUEST_FILE", str(request_path))
    monkeypatch.setenv("HERMES_UPDATE_AGENT_NAME", "Harry HermesBot")

    with (
        patch("gateway.run._hermes_home", request_path.parent),
        patch("gateway.run._resolve_hermes_bin", return_value=None),
        patch("gateway.slash_commands.__file__", _fake_gateway_file(tmp_path)),
        patch("hermes_cli.config.is_managed", return_value=False),
        patch("hermes_cli.config.detect_install_method", return_value="docker"),
    ):
        response = asyncio.run(_runner()._handle_update_command(_event()))

    assert response == (
        "⏳ Sichere Docker-Aktualisierung angefordert. Harry HermesBot "
        "aktualisiert sich im Hintergrund und startet danach neu."
    )
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    assert payload["schema"] == 1
    assert payload["action"] == "update"
    assert payload["platform"] == "telegram"
    assert payload["chat_id"] == "probe-chat"
    assert payload["chat_type"] == "private"
    assert payload["user_id"] == "probe-user"
    assert payload["message_id"] == "probe-message"
    assert isinstance(payload["timestamp"], str) and payload["timestamp"]
    assert request_path.stat().st_mode & 0o777 == 0o600
    assert not list(request_path.parent.glob(".host_update_request.*.tmp"))


def test_docker_update_fails_closed_without_a_host_watcher(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_HOST_UPDATE_REQUEST_FILE", raising=False)

    with (
        patch("gateway.run._hermes_home", tmp_path),
        patch("gateway.slash_commands.__file__", _fake_gateway_file(tmp_path)),
        patch("hermes_cli.config.is_managed", return_value=False),
        patch("hermes_cli.config.detect_install_method", return_value="docker"),
    ):
        response = asyncio.run(_runner()._handle_update_command(_event()))

    assert "Docker-Update ist auf diesem Host nicht eingerichtet" in response

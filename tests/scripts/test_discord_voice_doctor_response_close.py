"""Discord diagnostics release user, login, and guild responses."""

import importlib.util
import io
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests


@pytest.fixture
def doctor():
    path = Path(__file__).resolve().parents[2] / "scripts/discord-voice-doctor.py"
    spec = importlib.util.spec_from_file_location("voice_doctor_response_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _response(body, status=200):
    response = requests.Response()
    response.status_code = status
    response._content = body
    response.raw = io.BytesIO(body)
    response.close = Mock(wraps=response.close)
    return response


@pytest.mark.parametrize("login_status,guild_status", [(200, 200), (200, 503), (401, 200)])
def test_permission_responses_closed(doctor, monkeypatch, login_status, guild_status):
    responses = [_response(b'{"username": "test-bot"}', login_status)]
    if login_status == 200:
        responses.append(_response(b"[]", guild_status))
    monkeypatch.setattr(requests, "get", Mock(side_effect=responses))

    assert doctor.check_bot_permissions("test-token") is (login_status == 200)
    for response in responses:
        response.close.assert_called_once_with()


def test_user_lookup_response_closed(doctor, monkeypatch):
    import hermes_cli.env_loader

    monkeypatch.setattr(hermes_cli.env_loader, "load_hermes_dotenv", lambda **kwargs: None)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "123456789")
    response = _response(b'{"username": "test-user"}')
    monkeypatch.setattr(requests, "get", Mock(return_value=response))

    doctor.check_env_vars()
    response.close.assert_called_once_with()

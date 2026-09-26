import importlib.util
import json
from pathlib import Path

import requests


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _load_doctor(monkeypatch, hermes_home):
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "discord-voice-doctor.py"
    spec = importlib.util.spec_from_file_location("discord_voice_doctor", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _guild_with_required_permissions():
    required_permissions = (1 << 10) | (1 << 11) | (1 << 20) | (1 << 21)
    return {
        "id": "guild-1",
        "name": "Alfred server",
        "permissions": str(required_permissions),
    }


def _write_config_with_channel_prompt(hermes_home, channel_id):
    (hermes_home / "config.yaml").write_text(
        "discord:\n"
        "  channel_prompts:\n"
        f"    '{channel_id}': Drive Mode prompt\n",
        encoding="utf-8",
    )


def test_check_bot_permissions_fails_when_bot_has_no_guilds(monkeypatch, tmp_path, capsys):
    doctor = _load_doctor(monkeypatch, tmp_path)

    def fake_get(url, headers, timeout):
        if url.endswith("/users/@me"):
            return _Response(200, {"username": "Alfred Drive"})
        if url.endswith("/users/@me/guilds"):
            return _Response(200, [])
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)

    assert doctor.check_bot_permissions("test-token") is False
    output = capsys.readouterr().out
    assert "Guilds" in output
    assert "0 guild(s)" in output
    assert "invite it before voice testing" in output


def test_check_bot_permissions_passes_when_no_configured_channels_exist(monkeypatch, tmp_path):
    doctor = _load_doctor(monkeypatch, tmp_path)

    def fake_get(url, headers, timeout):
        if url.endswith("/users/@me"):
            return _Response(200, {"username": "Alfred Drive"})
        if url.endswith("/users/@me/guilds"):
            return _Response(200, [_guild_with_required_permissions()])
        if "/channels/" in url:
            raise AssertionError(f"unexpected channel check: {url}")
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)

    assert doctor.check_bot_permissions("test-token") is True


def test_check_bot_permissions_fails_when_configured_channel_is_forbidden(
    monkeypatch,
    tmp_path,
    capsys,
):
    doctor = _load_doctor(monkeypatch, tmp_path)
    _write_config_with_channel_prompt(tmp_path, "1234567890")

    def fake_get(url, headers, timeout):
        if url.endswith("/users/@me"):
            return _Response(200, {"username": "Alfred Drive"})
        if url.endswith("/users/@me/guilds"):
            return _Response(200, [_guild_with_required_permissions()])
        if url.endswith("/channels/1234567890"):
            return _Response(403, {"message": "Missing Access"})
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)

    assert doctor.check_bot_permissions("test-token") is False
    output = capsys.readouterr().out
    assert "Configured channel 1234567890" in output
    assert "Missing Access" in output
    assert "add the Alfred Drive bot role" in output


def test_check_bot_permissions_passes_when_configured_channels_are_accessible(
    monkeypatch,
    tmp_path,
    capsys,
):
    doctor = _load_doctor(monkeypatch, tmp_path)
    _write_config_with_channel_prompt(tmp_path, "1234567890")
    (tmp_path / "gateway_voice_mode.json").write_text(
        json.dumps({"discord:9876543210": "off"}),
        encoding="utf-8",
    )

    checked_channel_ids = set()

    def fake_get(url, headers, timeout):
        if url.endswith("/users/@me"):
            return _Response(200, {"username": "Alfred Drive"})
        if url.endswith("/users/@me/guilds"):
            return _Response(200, [_guild_with_required_permissions()])
        if "/channels/" in url:
            channel_id = url.rsplit("/", 1)[-1]
            checked_channel_ids.add(channel_id)
            return _Response(200, {"id": channel_id, "name": f"channel-{channel_id}"})
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(requests, "get", fake_get)

    assert doctor.check_bot_permissions("test-token") is True
    assert checked_channel_ids == {"1234567890", "9876543210"}
    output = capsys.readouterr().out
    assert "Configured channel 1234567890" in output
    assert "Configured channel 9876543210" in output
    assert "accessible" in output

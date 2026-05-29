import io
import json
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from plugins.memory.kynver.agentos_bridge import (
    KynverAgentOSClient,
    KynverAgentOSConfig,
    KynverAgentOSError,
    _load_profile_env,
    agentos_enabled,
    load_kynver_agentos_config,
    redact,
)


def test_load_profile_env_parses_dotenv(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# comment\nKYNVER_API_KEY='secret'\nKYNVER_AGENT_OS_SLUG=forge\n",
        encoding="utf-8",
    )

    assert _load_profile_env(env_path) == {
        "KYNVER_API_KEY": "secret",
        "KYNVER_AGENT_OS_SLUG": "forge",
    }


def test_load_config_defaults_enabled_when_configured(monkeypatch, tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("KYNVER_API_KEY=profile-key\nKYNVER_AGENT_OS_SLUG=forge\n", encoding="utf-8")
    monkeypatch.setattr("plugins.memory.kynver.agentos_bridge._active_env_path", lambda: env_path)

    cfg = load_kynver_agentos_config({"KYNVER_FETCH_TIMEOUT_MS": "2500", "KYNVER_OBSERVER_TIMEOUT_MS": "1000"})

    assert cfg.configured is True
    assert cfg.enabled is True
    assert cfg.mode == "enabled"
    assert cfg.timeout == 2.5
    assert cfg.side_effect_timeout == 1.0


def test_load_config_uses_short_default_timeouts():
    cfg = load_kynver_agentos_config(
        {"KYNVER_API_KEY": "key", "KYNVER_AGENT_OS_SLUG": "forge"}
    )

    assert cfg.timeout <= 3.0
    assert cfg.side_effect_timeout <= 3.0


def test_disabled_mode_is_opt_out():
    assert not agentos_enabled(
        {
            "KYNVER_API_KEY": "key",
            "KYNVER_AGENT_OS_SLUG": "forge",
            "KYNVER_AGENTOS_MODE": "disabled",
        }
    )


def test_disabled_escape_hatches_are_loaded():
    cfg = load_kynver_agentos_config(
        {
            "KYNVER_API_KEY": "key",
            "KYNVER_AGENT_OS_SLUG": "forge",
            "KYNVER_TASKS_DISABLED": "1",
            "KYNVER_SKILLS_DISABLED": "true",
            "KYNVER_SESSION_SYNC_DISABLED": "yes",
            "KYNVER_TODO_MIRROR_DISABLED": "on",
        }
    )

    assert cfg.tasks_disabled is True
    assert cfg.skills_disabled is True
    assert cfg.session_sync_disabled is True
    assert cfg.todo_mirror_disabled is True


class _FakeResponse:
    def __init__(self, payload: str):
        self.payload = payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def test_request_sends_bearer_and_parses_json(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["timeout"] = timeout
        seen["auth"] = req.headers.get("Authorization")
        seen["method"] = req.get_method()
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse('{"ok": true}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = KynverAgentOSClient(
        KynverAgentOSConfig(api_url="https://example.test", api_key="secret", slug="forge", timeout=3)
    )

    result = client.post("/sessions", {"channel": "cli"})

    assert result == {"ok": True}
    assert seen == {
        "url": "https://example.test/api/agent-os/forge/sessions",
        "timeout": 3,
        "auth": "Bearer secret",
        "method": "POST",
        "body": {"channel": "cli"},
    }


def test_request_accepts_per_call_timeout(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["timeout"] = timeout
        return _FakeResponse('{"ok": true}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = KynverAgentOSClient(
        KynverAgentOSConfig(api_url="https://example.test", api_key="secret", slug="forge", timeout=10)
    )

    assert client.post("/sessions/session-1/events", {"event": {"summary": "turn"}}, timeout=1.25) == {"ok": True}
    assert seen["timeout"] == 1.25


def test_request_redacts_http_error(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url,
            401,
            "Unauthorized",
            hdrs=Message(),
            fp=io.BytesIO(b"bad Bearer secret-token api_key=abc123"),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = KynverAgentOSClient(KynverAgentOSConfig(api_url="https://example.test", api_key="secret", slug="forge"))

    with pytest.raises(KynverAgentOSError) as exc:
        client.get("/stats")

    msg = str(exc.value)
    assert "secret-token" not in msg
    assert "abc123" not in msg
    assert "[REDACTED]" in msg


def test_redact_covers_json_and_header_secret_forms():
    raw = (
        '{"apiKey": "json-secret", "token": "tok-secret", '
        '"Authorization": "Bearer auth-secret"}\n'
        "x-api-key: header-secret\n"
        "Authorization: Bearer bearer-secret\n"
        "password=pw-secret"
    )

    msg = redact(raw)

    for secret in ("json-secret", "tok-secret", "auth-secret", "header-secret", "bearer-secret", "pw-secret"):
        assert secret not in msg
    assert "[REDACTED]" in msg

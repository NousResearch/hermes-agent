import json
import io
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from plugins.memory.kynver.agentos_bridge import (
    KynverAgentOSClient,
    KynverAgentOSConfig,
    KynverAgentOSError,
    _load_profile_env,
    agentos_available,
    load_kynver_agentos_config,
    probe_agentos_health,
    redact,
)


def test_load_profile_env_parses_basic_dotenv(tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# comment\nKYNVER_API_URL=https://example.test/\nKYNVER_API_KEY='secret'\nKYNVER_AGENT_OS_SLUG=forge\n",
        encoding="utf-8",
    )

    assert _load_profile_env(env_path) == {
        "KYNVER_API_URL": "https://example.test/",
        "KYNVER_API_KEY": "secret",
        "KYNVER_AGENT_OS_SLUG": "forge",
    }


def test_load_config_merges_profile_and_process_env(monkeypatch, tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text("KYNVER_API_KEY=profile-key\nKYNVER_AGENT_OS_SLUG=forge\n", encoding="utf-8")
    monkeypatch.setattr("plugins.memory.kynver.agentos_bridge._active_env_path", lambda: env_path)

    cfg = load_kynver_agentos_config({"KYNVER_API_KEY": "process-key", "KYNVER_FETCH_TIMEOUT_MS": "2500"})

    assert cfg.api_key == "process-key"
    assert cfg.slug == "forge"
    assert cfg.timeout == 2.5


def test_agentos_available_requires_credentials():
    assert not agentos_available({})
    assert agentos_available(
        {
            "KYNVER_API_URL": "https://example.test",
            "KYNVER_API_KEY": "key",
            "KYNVER_AGENT_OS_SLUG": "forge",
        }
    )


def test_api_path_rejects_traversal():
    client = KynverAgentOSClient(
        KynverAgentOSConfig(api_url="https://example.test", api_key="key", slug="forge")
    )
    with pytest.raises(KynverAgentOSError):
        client.api_path("../secrets")


def test_http_error_redacts_bearer_token(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url,
            401,
            "Unauthorized",
            hdrs=Message(),
            fp=io.BytesIO(b"bad Bearer secret-token token=abc123"),
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


def test_probe_agentos_health_success():
    client = MagicMock()
    client.config.enabled = True
    client.get.return_value = {"ok": True}
    assert probe_agentos_health(client) is True


def test_probe_agentos_health_failure():
    client = MagicMock()
    client.config.enabled = True
    client.get.side_effect = RuntimeError("down")
    assert probe_agentos_health(client) is False

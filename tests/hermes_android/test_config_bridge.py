import json
import time
from pathlib import Path

from hermes_android.auth_bridge import (
    clear_provider_auth_bundle,
    provider_env_key,
    provider_env_keys,
    read_provider_api_key,
    read_provider_auth_bundle,
    write_provider_api_key,
    write_provider_auth_bundle,
)
from hermes_android.config_bridge import read_runtime_config, write_runtime_config


def test_write_runtime_config_updates_model_section(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    updated = write_runtime_config(
        provider="openrouter",
        model="anthropic/claude-sonnet-4",
        base_url="https://openrouter.ai/api/v1",
    )

    assert updated["model"]["provider"] == "openrouter"
    assert updated["model"]["default"] == "anthropic/claude-sonnet-4"
    assert updated["model"]["base_url"] == "https://openrouter.ai/api/v1"
    assert read_runtime_config()["model"]["provider"] == "openrouter"


def test_auth_bridge_reads_and_writes_provider_api_key(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = write_provider_api_key("openrouter", "sk-test")

    assert result == {"provider": "openrouter", "env_key": "OPENROUTER_API_KEY", "saved": True}
    assert provider_env_key("openrouter") == "OPENROUTER_API_KEY"
    assert read_provider_api_key("openrouter") == "sk-test"


def test_auth_bridge_supports_chatgpt_web_session_and_access_tokens(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = write_provider_auth_bundle(
        "chatgpt-web",
        access_token="chatgpt-access",
        session_token="chatgpt-session",
    )
    bundle = read_provider_auth_bundle("chatgpt-web")

    assert result["provider"] == "chatgpt-web"
    assert bundle["configured"] is True
    assert bundle["api_key"] == "chatgpt-access"
    assert bundle["access_token"] == "chatgpt-access"
    assert bundle["session_token"] == "chatgpt-session"

    clear_result = clear_provider_auth_bundle("chatgpt-web")
    cleared = read_provider_auth_bundle("chatgpt-web")
    assert clear_result["cleared"] is True
    assert cleared["configured"] is False
    assert cleared["api_key"] == ""
    assert cleared["session_token"] == ""


def test_auth_bridge_supports_anthropic_gemini_and_zai_bundles(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    write_provider_auth_bundle("anthropic", api_key="anthropic-key", access_token="anthropic-oauth")
    write_provider_auth_bundle("gemini", api_key="gemini-key")
    write_provider_auth_bundle("zai", api_key="glm-key")
    write_provider_auth_bundle("alibaba", api_key="dashscope-key")

    anthropic_bundle = read_provider_auth_bundle("anthropic")
    gemini_bundle = read_provider_auth_bundle("gemini")
    zai_bundle = read_provider_auth_bundle("zai")
    alibaba_bundle = read_provider_auth_bundle("alibaba")

    assert anthropic_bundle["configured"] is True
    assert anthropic_bundle["api_key"] == "anthropic-key"
    assert anthropic_bundle["access_token"] == "anthropic-oauth"
    assert gemini_bundle["configured"] is True
    assert gemini_bundle["api_key"] == "gemini-key"
    assert zai_bundle["configured"] is True
    assert zai_bundle["api_key"] == "glm-key"
    assert alibaba_bundle["configured"] is True
    assert alibaba_bundle["api_key"] == "dashscope-key"
    assert provider_env_key("alibaba") == "DASHSCOPE_API_KEY"


def test_auth_bridge_recognizes_zai_cli_env_aliases(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    assert provider_env_key("zai") == "GLM_API_KEY"
    assert provider_env_keys("zai") == ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY")

    (hermes_home / ".env").write_text("ZAI_API_KEY=zai-fallback-key\n", encoding="utf-8")
    assert read_provider_api_key("zai") == "zai-fallback-key"
    assert read_provider_auth_bundle("zai")["api_key"] == "zai-fallback-key"

    write_provider_api_key("zai", "primary-glm-key")
    assert read_provider_api_key("zai") == "primary-glm-key"

    clear_provider_auth_bundle("zai")
    cleared = (hermes_home / ".env").read_text(encoding="utf-8")
    assert "GLM_API_KEY=" in cleared
    assert "ZAI_API_KEY=" in cleared
    assert "Z_AI_API_KEY=" in cleared



def test_auth_bridge_supports_qwen_oauth_bundle_via_home_scoped_cli_file(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(tmp_path))

    result = write_provider_auth_bundle(
        "qwen-oauth",
        access_token="qwen-access",
        refresh_token="qwen-refresh",
        base_url="https://portal.qwen.ai/v1",
    )
    bundle = read_provider_auth_bundle("qwen-oauth")
    auth_file = Path(tmp_path) / ".qwen" / "oauth_creds.json"

    assert result["provider"] == "qwen-oauth"
    assert auth_file.exists()
    saved = json.loads(auth_file.read_text(encoding="utf-8"))
    assert saved["access_token"] == "qwen-access"
    assert saved["refresh_token"] == "qwen-refresh"
    assert bundle["configured"] is True
    assert bundle["api_key"] == "qwen-access"
    assert bundle["refresh_token"] == "qwen-refresh"
    assert bundle["base_url"] == "https://portal.qwen.ai/v1"

    clear_provider_auth_bundle("qwen-oauth")
    assert not auth_file.exists()


def test_auth_bridge_refreshes_expiring_qwen_oauth_bundle(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(tmp_path))

    auth_file = Path(tmp_path) / ".qwen" / "oauth_creds.json"
    auth_file.parent.mkdir()
    auth_file.write_text(
        json.dumps(
            {
                "access_token": "old-access",
                "refresh_token": "old-refresh",
                "token_type": "Bearer",
                "resource_url": "portal.qwen.ai",
                "expiry_date": int((time.time() - 60) * 1000),
            }
        ),
        encoding="utf-8",
    )

    def fake_refresh(tokens):
        assert tokens["refresh_token"] == "old-refresh"
        refreshed = {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "token_type": "Bearer",
            "resource_url": "portal.qwen.ai",
            "expiry_date": int((time.time() + 3600) * 1000),
        }
        auth_file.write_text(json.dumps(refreshed), encoding="utf-8")
        return refreshed

    monkeypatch.setattr("hermes_android.auth_bridge._refresh_qwen_tokens", fake_refresh)

    bundle = read_provider_auth_bundle("qwen-oauth")

    assert bundle["configured"] is True
    assert bundle["api_key"] == "new-access"
    assert bundle["access_token"] == "new-access"
    assert bundle["refresh_token"] == "new-refresh"
    assert bundle["refresh_error"] == ""
    saved = json.loads(auth_file.read_text(encoding="utf-8"))
    assert saved["access_token"] == "new-access"

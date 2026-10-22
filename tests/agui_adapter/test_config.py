"""Integration tests for AG-UI config.yaml propagation."""

import pytest

from agui_adapter.entry import _load_listener_config
from agui_adapter.session import AgentConfig
from hermes_cli import config as config_module
from hermes_cli.config import save_config


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Use a real, isolated config.yaml and keep loader caches test-local."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()
    yield tmp_path
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()


def test_agui_config_and_secret_metadata_are_registered():
    assert "agui" in config_module.DEFAULT_CONFIG
    assert "agui" in config_module._KNOWN_ROOT_KEYS

    token_metadata = config_module.OPTIONAL_ENV_VARS["HERMES_AGUI_SESSION_TOKEN"]
    assert token_metadata["password"] is True


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("port", p, 8000)
        for p in (True, 1.5, float("inf"), float("nan"), -1, 65536, [], {})
    ]
    + [("host", h, "127.0.0.1") for h in ("", " ", [], {}, False)]
    + [("port", 0, 0), ("port", "0", 0), ("port", 9123, 9123)],
)
def test_listener_settings_come_from_config_yaml(
    isolated_hermes_home, monkeypatch, caplog, key, value, expected
):
    import uvicorn
    from agui_adapter import entry, server

    save_config({"agui": {"host": "127.0.0.1", "port": 0, key: value}})
    monkeypatch.setenv("HERMES_AGUI_HOST", "legacy.example.com")
    monkeypatch.setenv("HERMES_AGUI_PORT", "9999")
    monkeypatch.setenv("PORT", "7777")

    listener = {"host": "127.0.0.1", "port": 0, key: expected}
    assert _load_listener_config() == (listener["host"], listener["port"])
    invalid = value != expected or type(value) is not type(expected)
    if invalid and value != "0":
        assert f"agui.{key}" in caplog.text

    def bind(app, **kwargs):
        assert (kwargs["host"], kwargs["port"]) == (listener["host"], listener["port"])
        config = uvicorn.Config(app, **kwargs)
        if config.port == 0:
            with config.bind_socket() as sock:
                assert sock.getsockname()[1] > 0

    monkeypatch.setattr(uvicorn, "run", bind)
    entry.main()
    if invalid and value != "0":
        monkeypatch.setattr(
            server,
            "create_app",
            lambda **kw: pytest.fail("invalid override reached app"),
        )
        with pytest.raises(SystemExit, match=f"agui.{key}"):
            entry.main(**{key: value})
    else:
        entry.main(**{key: value})


def test_agent_settings_come_from_config_yaml(isolated_hermes_home, monkeypatch):
    save_config(
        {
            "agui": {
                "base_url": "http://localhost:4010/v1",
                "model": "configured-model",
                "provider": "custom",
                "api_mode": "chat_completions",
                "toolsets": ["hermes-acp", "web"],
            }
        }
    )
    monkeypatch.setenv("HERMES_AGUI_BASE_URL", "https://legacy.invalid/v1")
    monkeypatch.setenv("HERMES_AGUI_MODEL", "legacy-model")
    monkeypatch.setenv("HERMES_AGUI_PROVIDER", "legacy-provider")
    monkeypatch.setenv("HERMES_AGUI_API_MODE", "legacy-mode")
    monkeypatch.setenv("HERMES_AGUI_TOOLSETS", "terminal")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-from-credential-env")

    config = AgentConfig()

    assert config.base_url == "http://localhost:4010/v1"
    assert config.model == "configured-model"
    assert config.provider == "custom"
    assert config.api_mode == "chat_completions"
    assert config.enabled_toolsets == ["hermes-acp", "web"]
    assert config.api_key == "secret-from-credential-env"


def test_agent_settings_use_adapter_defaults_without_user_config(isolated_hermes_home):
    config = AgentConfig()

    assert config.base_url is None
    assert config.model == ""
    assert config.provider is None
    assert config.api_mode is None
    assert config.enabled_toolsets == ["hermes-acp"]


def test_explicit_endpoint_uses_profile_scoped_credential(
    isolated_hermes_home, monkeypatch
):
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    save_config({"agui": {"base_url": "http://localhost:4010/v1"}})
    monkeypatch.setenv("OPENAI_API_KEY", "launch-profile-credential")
    token = set_secret_scope({"OPENAI_API_KEY": "owning-profile-credential"})
    try:
        config = AgentConfig()
    finally:
        reset_secret_scope(token)
    assert config.api_key == "owning-profile-credential"


def test_lifecycle_defaults_have_matching_runtime_readers(isolated_hermes_home):
    from agui_adapter.continuation import frontend_timeout

    assert (
        frontend_timeout()
        == config_module.DEFAULT_CONFIG["agui"]["frontend_tool_timeout"]
    )


def test_frontend_timeout_reads_real_config(isolated_hermes_home):
    from agui_adapter.continuation import frontend_timeout

    save_config({"agui": {"frontend_tool_timeout": 27}})
    assert frontend_timeout() == 27


def test_conversation_limits_read_real_config(isolated_hermes_home):
    from agui_adapter.conversations import conversation_limits

    save_config({"agui": {"max_conversations": 3, "conversation_idle_timeout": 23}})
    assert conversation_limits() == (3, 23)


@pytest.mark.parametrize(
    "key,value",
    [
        ("frontend_tool_timeout", True),
        ("frontend_tool_timeout", None),
        ("frontend_tool_timeout", 0),
        ("frontend_tool_timeout", float("inf")),
        ("conversation_idle_timeout", None),
        ("conversation_idle_timeout", False),
        ("conversation_idle_timeout", -1),
        ("max_conversations", True),
        ("max_conversations", 0),
        ("max_conversations", 1.5),
    ],
)
def test_invalid_lifecycle_config_refuses_startup(isolated_hermes_home, key, value):
    from agui_adapter.server import create_app

    save_config({"agui": {key: value}})
    with pytest.raises(ValueError, match=key):
        create_app()

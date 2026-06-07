import importlib

from agent.cursor import (
    CONNECT_END_STREAM_FLAG,
    CURSOR_AGENT_RUN_PATH,
    CURSOR_API_URL,
    CURSOR_CLIENT_VERSION,
    CURSOR_GET_USABLE_MODELS_PATH,
)


from agent.cursor.constants import normalize_cursor_model_id


def test_normalize_cursor_model_id_maps_legacy_composer_names() -> None:
    assert normalize_cursor_model_id("cursor-composer-2.5") == "composer-2.5"
    assert normalize_cursor_model_id("composer-2.5") == "composer-2.5"
    assert normalize_cursor_model_id("claude-4.6-opus-high") == "claude-4.6-opus-high"


def test_cursor_constants_exist() -> None:
    assert CURSOR_API_URL == "https://api2.cursor.sh"
    assert CURSOR_CLIENT_VERSION == "cli-2026.01.09-231024f"
    assert CURSOR_AGENT_RUN_PATH == "/agent.v1.AgentService/Run"
    assert CURSOR_GET_USABLE_MODELS_PATH == "/agent.v1.AgentService/GetUsableModels"
    assert CONNECT_END_STREAM_FLAG == 0b00000010


def test_cursor_client_version_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("CURSOR_CLIENT_VERSION", "cli-test-version")
    constants = importlib.import_module("agent.cursor.constants")
    constants = importlib.reload(constants)

    assert constants.CURSOR_CLIENT_VERSION == "cli-test-version"

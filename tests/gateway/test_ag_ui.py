"""
Tests for the AG-UI protocol adapter.

Covers:
- Platform enum registration
- Config loading from env vars
- Protocol event serialisation
- Adapter initialisation
- Auth helper
- Health / capabilities endpoints
"""

import json
import os
import pytest
from unittest.mock import patch

from gateway.config import Platform, PlatformConfig, GatewayConfig, load_gateway_config
from gateway.platforms.ag_ui_protocol import (
    AGUIEventType,
    AGUIRunStartedEvent,
    AGUIRunFinishedEvent,
    AGUIRunErrorEvent,
    AGUITextMessageStartEvent,
    AGUITextMessageContentEvent,
    AGUITextMessageEndEvent,
    AGUIToolCallStartEvent,
    AGUIToolCallArgsEvent,
    AGUIToolCallEndEvent,
    AGUIToolCallResultEvent,
    AGUIStateSnapshotEvent,
    AGUIRunAgentInput,
    AGUIMessage,
)
from gateway.platforms.ag_ui_server import AGUIServerAdapter, check_ag_ui_requirements


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------

def test_platform_enum_exists():
    assert Platform.AG_UI.value == "ag_ui"


def test_platform_enum_identity():
    assert Platform("ag_ui") is Platform.AG_UI


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

def test_check_requirements_aiohttp_present():
    assert check_ag_ui_requirements() is True


def test_check_requirements_no_aiohttp():
    import gateway.platforms.ag_ui_server as mod
    original = mod.AIOHTTP_AVAILABLE
    mod.AIOHTTP_AVAILABLE = False
    try:
        assert check_ag_ui_requirements() is False
    finally:
        mod.AIOHTTP_AVAILABLE = original


# ---------------------------------------------------------------------------
# Config loading from env vars
# ---------------------------------------------------------------------------

def test_env_ag_ui_enabled():
    with patch.dict(os.environ, {"AG_UI_ENABLED": "true"}, clear=False):
        config = GatewayConfig()
        from gateway.config import _apply_env_overrides
        _apply_env_overrides(config)
        assert Platform.AG_UI in config.platforms
        assert config.platforms[Platform.AG_UI].enabled is True


def test_env_ag_ui_key_enables():
    with patch.dict(os.environ, {"AG_UI_KEY": "mysecretkey1234567"}, clear=False):
        config = GatewayConfig()
        from gateway.config import _apply_env_overrides
        _apply_env_overrides(config)
        assert Platform.AG_UI in config.platforms
        assert config.platforms[Platform.AG_UI].token == "mysecretkey1234567"


def test_env_ag_ui_port():
    with patch.dict(os.environ, {"AG_UI_ENABLED": "true", "AG_UI_PORT": "9000"}, clear=False):
        config = GatewayConfig()
        from gateway.config import _apply_env_overrides
        _apply_env_overrides(config)
        assert config.platforms[Platform.AG_UI].extra.get("port") == 9000


def test_env_ag_ui_host():
    with patch.dict(os.environ, {"AG_UI_ENABLED": "true", "AG_UI_HOST": "127.0.0.1"}, clear=False):
        config = GatewayConfig()
        from gateway.config import _apply_env_overrides
        _apply_env_overrides(config)
        assert config.platforms[Platform.AG_UI].extra.get("host") == "127.0.0.1"


# ---------------------------------------------------------------------------
# Adapter initialisation
# ---------------------------------------------------------------------------

def test_adapter_init_defaults():
    config = PlatformConfig()
    adapter = AGUIServerAdapter(config)
    assert adapter._host == "0.0.0.0"
    assert adapter._port == 8643
    assert adapter._api_key is None


def test_adapter_init_custom_port():
    config = PlatformConfig()
    config.extra["port"] = 9999
    config.extra["host"] = "127.0.0.1"
    adapter = AGUIServerAdapter(config)
    assert adapter._port == 9999
    assert adapter._host == "127.0.0.1"


def test_adapter_init_api_key():
    config = PlatformConfig()
    config.token = "supersecretkey"
    adapter = AGUIServerAdapter(config)
    assert adapter._api_key == "supersecretkey"


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def test_auth_no_key_always_passes():
    from unittest.mock import MagicMock
    config = PlatformConfig()
    adapter = AGUIServerAdapter(config)
    request = MagicMock()
    assert adapter._check_auth(request) is True


def test_auth_bearer_token_valid():
    from unittest.mock import MagicMock
    config = PlatformConfig()
    config.token = "mytoken123"
    adapter = AGUIServerAdapter(config)
    request = MagicMock()
    request.headers = {"Authorization": "Bearer mytoken123"}
    assert adapter._check_auth(request) is True


def test_auth_bearer_token_invalid():
    from unittest.mock import MagicMock
    config = PlatformConfig()
    config.token = "mytoken123"
    adapter = AGUIServerAdapter(config)
    request = MagicMock()
    request.headers = {"Authorization": "Bearer wrongtoken"}
    assert adapter._check_auth(request) is False


def test_auth_x_api_key_valid():
    from unittest.mock import MagicMock
    config = PlatformConfig()
    config.token = "mytoken123"
    adapter = AGUIServerAdapter(config)
    request = MagicMock()
    request.headers = {"X-API-Key": "mytoken123"}
    assert adapter._check_auth(request) is True


# ---------------------------------------------------------------------------
# Protocol event serialisation
# ---------------------------------------------------------------------------

def test_run_started_event_serialises():
    event = AGUIRunStartedEvent(thread_id="t1", run_id="r1")
    data = json.loads(event.model_dump_json(by_alias=True, exclude_none=True))
    assert data["type"] == "RUN_STARTED"
    assert data["threadId"] == "t1"
    assert data["runId"] == "r1"
    assert "timestamp" in data


def test_run_finished_event_serialises():
    event = AGUIRunFinishedEvent(thread_id="t1", run_id="r1")
    data = json.loads(event.model_dump_json(by_alias=True, exclude_none=True))
    assert data["type"] == "RUN_FINISHED"


def test_run_error_event_serialises():
    event = AGUIRunErrorEvent(message="something broke", code="TEST_ERROR")
    data = json.loads(event.model_dump_json(by_alias=True, exclude_none=True))
    assert data["type"] == "RUN_ERROR"
    assert data["message"] == "something broke"
    assert data["code"] == "TEST_ERROR"


def test_text_message_events_serialise():
    start = AGUITextMessageStartEvent(message_id="m1", role="assistant")
    content = AGUITextMessageContentEvent(message_id="m1", delta="hello")
    end = AGUITextMessageEndEvent(message_id="m1")
    for event in (start, content, end):
        data = json.loads(event.model_dump_json(by_alias=True, exclude_none=True))
        assert data["messageId"] == "m1"


def test_tool_call_events_serialise():
    start = AGUIToolCallStartEvent(
        tool_call_id="tc1", tool_call_name="web_search", parent_message_id="m1"
    )
    args = AGUIToolCallArgsEvent(tool_call_id="tc1", delta='{"query": "test"}')
    end = AGUIToolCallEndEvent(tool_call_id="tc1")
    result = AGUIToolCallResultEvent(
        message_id="m2", tool_call_id="tc1", content="search results"
    )
    for event in (start, args, end, result):
        data = json.loads(event.model_dump_json(by_alias=True, exclude_none=True))
        assert data["toolCallId"] == "tc1"


def test_state_snapshot_event_serialises():
    event = AGUIStateSnapshotEvent(snapshot={"status": "waiting_for_approval", "cmd": "rm -rf"})
    data = json.loads(event.model_dump_json(by_alias=True, exclude_none=True))
    assert data["type"] == "STATE_SNAPSHOT"
    assert data["snapshot"]["status"] == "waiting_for_approval"


# ---------------------------------------------------------------------------
# SSE line format
# ---------------------------------------------------------------------------

def test_sse_line_format():
    config = PlatformConfig()
    adapter = AGUIServerAdapter(config)
    event = AGUIRunStartedEvent(thread_id="t1", run_id="r1")
    line = adapter._sse_line(event)
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    payload = json.loads(line[6:])
    assert payload["type"] == "RUN_STARTED"


# ---------------------------------------------------------------------------
# RunAgentInput parsing
# ---------------------------------------------------------------------------

def test_run_agent_input_parses_camel_case():
    raw = {
        "threadId": "thread-abc",
        "runId": "run-xyz",
        "messages": [{"id": "msg1", "role": "user", "content": "hello"}],
    }
    inp = AGUIRunAgentInput.model_validate(raw)
    assert inp.thread_id == "thread-abc"
    assert inp.run_id == "run-xyz"
    assert inp.messages[0].role == "user"


def test_run_agent_input_empty_messages():
    raw = {"threadId": "t1", "runId": "r1"}
    inp = AGUIRunAgentInput.model_validate(raw)
    assert inp.messages == []
    assert inp.state is None

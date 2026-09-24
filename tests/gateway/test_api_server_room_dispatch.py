"""Compatibility seams for extracted RoomLink dispatch handling."""

import json
from unittest.mock import MagicMock

import pytest

from gateway.platforms import api_server
from gateway.platforms import api_server_room_dispatch as room_dispatch








@pytest.mark.asyncio
async def test_non_room_run_body_passes_through_unchanged():
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._room_grant_token = MagicMock(return_value="")
    request = object()
    body = {"input": "ordinary run"}

    normalized, error = await adapter._normalize_room_dispatch(request, body)

    assert normalized is body
    assert error is None
    adapter._room_grant_token.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_room_dispatch_rejects_extra_fields_before_grant_verification():
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._room_grant_token = MagicMock(return_value="room-grant")
    request = object()
    body = {
        "input": "room prompt",
        "hosted_room_dispatch": {},
        "unexpected": True,
    }

    normalized, error = await adapter._normalize_room_dispatch(request, body)

    assert normalized is body
    assert error.status == 400
    assert json.loads(error.text)["error"]["code"] == "invalid_room_dispatch"


def test_room_dispatch_surfaces_typed_postgresql_activation_as_503() -> None:
    """A selected-PG hosted-room dispatch must not mask the activation failure as a
    generic 403 policy refusal: it surfaces the canonical session_db_unavailable 503."""
    from gateway.platforms.api_server import _openai_error
    from state_store_runtime_readiness import (
        PostgreSQLRuntimeActivationError, RuntimeActivationReport)

    exc = PostgreSQLRuntimeActivationError(RuntimeActivationReport(
        selected_backend="postgresql", profile_home="/tmp/pg-home", profile_name="pg",
        tenant_schema="hermes_tenant_x", supported_capabilities=(),
        missing_capabilities=("tui-api-session-runtime",), raw_state_db_openers=(),
    ))
    response = room_dispatch._room_dispatch_error(exc, _openai_error=_openai_error)

    payload = json.loads(response.text)
    assert response.status == 503
    assert payload["error"]["code"] == "session_db_unavailable"
    assert payload["error"]["type"] == "service_unavailable_error"
    assert "tui-api-session-runtime" in payload["diagnostic"]["missing_capabilities"]

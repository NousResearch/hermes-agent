"""Extraction seams and lifecycle coverage for API run idempotency."""

from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms import api_server
from gateway.platforms.api_server_run_idempotency import (
    RunIdempotencyStore as ExtractedRunIdempotencyStore,
)


def test_run_idempotency_store_remains_reexported_from_api_server():
    assert api_server.RunIdempotencyStore is ExtractedRunIdempotencyStore


@pytest.mark.asyncio
async def test_api_server_constructor_uses_legacy_run_store_monkeypatch(monkeypatch):
    store = MagicMock()
    store_factory = MagicMock(return_value=store)
    monkeypatch.setattr(api_server, "RunIdempotencyStore", store_factory)

    adapter = api_server.APIServerAdapter(PlatformConfig(enabled=True))
    try:
        store_factory.assert_called_once_with()
        assert adapter._run_idempotency_store is store
    finally:
        await adapter.disconnect()
    store.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_disconnect_tolerates_bare_fixture_without_run_idempotency_store():
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter.platform = Platform.API_SERVER
    adapter._mark_disconnected = MagicMock()
    adapter._close_cached_session_dbs = MagicMock()
    adapter._response_store = MagicMock()
    adapter._site = None
    adapter._runner = None
    adapter._app = object()

    assert not hasattr(adapter, "_run_idempotency_store")
    await adapter.disconnect()

    adapter._mark_disconnected.assert_called_once_with()
    adapter._response_store.close.assert_called_once_with()
    adapter._close_cached_session_dbs.assert_called_once_with()
    assert adapter._app is None

"""Matrix and DingTalk take the cross-gateway credential lock like every other unique-credential adapter.

Without it, two gateways on one Matrix account run two sync loops and both answer every message,
and two gateways on one DingTalk app split its Stream callbacks between them.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.status as gateway_status
import plugins.platforms.matrix.adapter as matrix_mod
from gateway.config import PlatformConfig
from gateway.status import _get_scope_lock_path
from tests.gateway.test_matrix import _make_fake_mautrix

_FOREIGN_HOLDER = {"pid": 4242, "profile": "other"}


def _matrix_adapter():
    return matrix_mod.MatrixAdapter(PlatformConfig(enabled=True, token="syt_test_token", extra={
        "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}))


def _fake_matrix_client():
    client = MagicMock()
    client.mxid = "@bot:example.org"
    client.device_id = "DEV"
    client.crypto = None
    client.whoami = AsyncMock(return_value=MagicMock(user_id="@bot:example.org", device_id="DEV"))
    client.sync = AsyncMock(return_value={"rooms": {"join": {}}})
    client.api = MagicMock()
    client.api.session.close = AsyncMock()
    return client


class TestMatrixAccountLock:

    @pytest.mark.asyncio
    async def test_connect_refused_when_account_held_by_another_gateway(self, monkeypatch):
        monkeypatch.setattr(gateway_status, "acquire_scoped_lock",
                            lambda scope, identity, metadata=None: (False, dict(_FOREIGN_HOLDER)))
        adapter = _matrix_adapter()
        adapter._encryption = False
        mods = _make_fake_mautrix()
        client = _fake_matrix_client()
        mods["mautrix.client"].Client = MagicMock(return_value=client)
        sync_loop = AsyncMock()
        with patch.dict("sys.modules", mods), patch.object(adapter, "_sync_loop", sync_loop), \
                patch.object(matrix_mod, "_create_matrix_session", return_value=MagicMock()):
            assert await adapter.connect() is False
        assert adapter._fatal_error_code == "matrix-account_lock"
        assert "other" in adapter._fatal_error_message
        client.sync.assert_not_awaited()
        sync_loop.assert_not_called()

    @pytest.mark.asyncio
    async def test_lock_held_while_connected_and_released_on_disconnect(self):
        adapter = _matrix_adapter()
        adapter._encryption = False
        mods = _make_fake_mautrix()
        mods["mautrix.client"].Client = MagicMock(return_value=_fake_matrix_client())
        lock_path = _get_scope_lock_path("matrix-account", "https://matrix.example.org|@bot:example.org")
        with patch.dict("sys.modules", mods), \
                patch.object(matrix_mod, "_create_matrix_session", return_value=MagicMock()), \
                patch.object(adapter, "_refresh_dm_cache", AsyncMock()), \
                patch.object(adapter, "_sync_loop", AsyncMock(return_value=None)):
            assert await adapter.connect() is True
            assert lock_path.exists()
            await adapter.disconnect()
        assert not lock_path.exists()

    @pytest.mark.asyncio
    async def test_case_variant_homeserver_cannot_start_second_gateway(self, monkeypatch):
        # DNS hostnames are case-insensitive; a different spelling is still the same Matrix account.
        held = set()

        def acquire(scope, identity, metadata=None):
            key = (scope, identity)
            if key in held:
                return False, dict(_FOREIGN_HOLDER)
            held.add(key)
            return True, None

        monkeypatch.setattr(gateway_status, "acquire_scoped_lock", acquire)
        first = _matrix_adapter()
        second = matrix_mod.MatrixAdapter(PlatformConfig(enabled=True, token="another_token", extra={
            "homeserver": "https://MATRIX.EXAMPLE.ORG", "user_id": "@bot:example.org"}))
        first._encryption = second._encryption = False
        mods = _make_fake_mautrix()
        mods["mautrix.client"].Client = MagicMock(side_effect=[_fake_matrix_client(), _fake_matrix_client()])
        with patch.dict("sys.modules", mods), \
                patch.object(matrix_mod, "_create_matrix_session", return_value=MagicMock()), \
                patch.object(matrix_mod.MatrixAdapter, "_refresh_dm_cache", AsyncMock()), \
                patch.object(matrix_mod.MatrixAdapter, "_sync_loop", AsyncMock(return_value=None)):
            try:
                assert await first.connect() is True
                assert await second.connect() is False
                assert second._fatal_error_code == "matrix-account_lock"
            finally:
                await first.disconnect()
                await second.disconnect()



def _dingtalk_adapter(monkeypatch):
    import plugins.platforms.dingtalk.adapter as dingtalk
    monkeypatch.setattr(dingtalk, "DINGTALK_STREAM_AVAILABLE", True)
    monkeypatch.setattr(dingtalk, "HTTPX_AVAILABLE", True)
    monkeypatch.setattr(dingtalk, "CARD_SDK_AVAILABLE", False)
    stream_client_cls = MagicMock()
    monkeypatch.setattr(dingtalk, "dingtalk_stream", MagicMock(DingTalkStreamClient=stream_client_cls))
    monkeypatch.setattr(dingtalk.DingTalkAdapter, "_run_stream", AsyncMock(return_value=None))
    adapter = dingtalk.DingTalkAdapter(PlatformConfig(enabled=True, extra={
        "client_id": "ding-app", "client_secret": "s3cret"}))
    return adapter, stream_client_cls


class TestDingTalkAppLock:

    @pytest.mark.asyncio
    async def test_connect_refused_when_app_held_by_another_gateway(self, monkeypatch):
        monkeypatch.setattr(gateway_status, "acquire_scoped_lock",
                            lambda scope, identity, metadata=None: (False, dict(_FOREIGN_HOLDER)))
        adapter, stream_client_cls = _dingtalk_adapter(monkeypatch)
        assert await adapter.connect() is False
        assert adapter._fatal_error_code == "dingtalk-client-id_lock"
        assert "other" in adapter._fatal_error_message
        stream_client_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_lock_held_while_connected_and_released_on_disconnect(self, monkeypatch):
        adapter, _ = _dingtalk_adapter(monkeypatch)
        lock_path = _get_scope_lock_path("dingtalk-client-id", "ding-app")
        assert await adapter.connect() is True
        assert lock_path.exists()
        await adapter.disconnect()
        assert not lock_path.exists()

    @pytest.mark.asyncio
    async def test_failed_connect_releases_lock(self, monkeypatch):
        adapter, stream_client_cls = _dingtalk_adapter(monkeypatch)
        stream_client_cls.side_effect = RuntimeError("bad credentials")
        assert await adapter.connect() is False
        assert not _get_scope_lock_path("dingtalk-client-id", "ding-app").exists()

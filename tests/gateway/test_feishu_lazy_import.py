"""Regression coverage for deferred Feishu SDK loading."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, patch


def _feishu_adapter_module():
    """Import the adapter with the Windows app-data root available in CI."""
    with patch.dict(os.environ, {"LOCALAPPDATA": tempfile.gettempdir()}):
        from plugins.platforms.feishu import adapter

    return adapter


def test_configured_feishu_dependency_check_does_not_load_sdk():
    """Gateway configuration can validate Feishu without importing its SDK."""
    feishu_adapter = _feishu_adapter_module()

    with (
        patch.object(feishu_adapter, "FEISHU_AVAILABLE", False),
        patch("tools.lazy_deps.ensure", autospec=True) as ensure,
    ):
        assert feishu_adapter.check_feishu_requirements() is True
        assert feishu_adapter.FEISHU_AVAILABLE is False

    ensure.assert_called_once_with("platform.feishu", prompt=False)


def test_feishu_connect_loads_sdk_on_worker_thread():
    """The first SDK import is deferred until a configured adapter connects."""
    from gateway.config import PlatformConfig
    feishu_adapter = _feishu_adapter_module()

    adapter = feishu_adapter.FeishuAdapter(
        PlatformConfig(
            extra={
                "app_id": "cli_test",
                "app_secret": "secret_test",
                "connection_mode": "websocket",
            }
        )
    )

    with (
        patch.object(feishu_adapter, "FEISHU_AVAILABLE", False),
        patch.object(feishu_adapter, "_load_lark_oapi", return_value=True) as load_sdk,
        patch.object(feishu_adapter.asyncio, "to_thread", new_callable=AsyncMock, return_value=True) as to_thread,
        patch.object(adapter, "_connect_with_retry", new_callable=AsyncMock),
        patch.object(feishu_adapter, "acquire_scoped_lock", return_value=(True, {})),
    ):
        assert asyncio.run(adapter.connect()) is True

    to_thread.assert_awaited_once_with(load_sdk)

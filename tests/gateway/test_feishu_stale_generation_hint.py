"""A cron delivery that misses Feishu names a restart, not hermes setup, when the
generation is already committed (#140620)."""

import asyncio
import os
import tempfile
from unittest.mock import patch


def _feishu_adapter_module():
    """Import the adapter with the Windows app-data root available in CI."""
    with patch.dict(os.environ, {"LOCALAPPDATA": tempfile.gettempdir()}):
        from plugins.platforms.feishu import adapter

    return adapter


def _deliver(feishu_adapter, stale):
    with (
        patch.object(feishu_adapter, "FEISHU_AVAILABLE", False),
        patch.object(feishu_adapter, "_load_lark_oapi", return_value=False),
        patch.object(feishu_adapter, "stale_dependency_generation", return_value=stale),
    ):
        return asyncio.run(feishu_adapter._standalone_send(None, "chat-1", "hello"))


def test_stale_generation_blames_the_process_not_setup():
    result = _deliver(_feishu_adapter_module(), stale=True)

    assert "Restart the gateway" in result["error"]
    assert "hermes setup" not in result["error"]


def test_genuinely_missing_dependency_keeps_the_setup_advice():
    result = _deliver(_feishu_adapter_module(), stale=False)

    assert "hermes setup" in result["error"]

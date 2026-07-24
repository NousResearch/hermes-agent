"""Regression tests for secret-safe Yuanbao refresh logging."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, patch

from agent.redact import mask_secret
from gateway.platforms.yuanbao import SignManager


def test_force_refresh_masks_app_key_and_preserves_result(caplog) -> None:
    app_key = "yuanbao-app-key-ABCDEFGH-SECRET99"
    fetched = {
        "token": "signed-token",
        "bot_id": "bot-1",
        "duration": 3600,
        "product": "yuanbao",
        "source": "test",
    }

    SignManager._cache.clear()
    SignManager.clear_locks()
    try:
        with (
            patch.object(SignManager, "fetch", new=AsyncMock(return_value=fetched)),
            caplog.at_level(logging.WARNING),
        ):
            result = asyncio.run(
                SignManager.force_refresh(
                    app_key,
                    "app-secret",
                    "https://yuanbao.example",
                )
            )

        assert result["token"] == "signed-token"
        assert result["bot_id"] == "bot-1"
        assert app_key not in caplog.text
        assert f"****{app_key[-4:]}" not in caplog.text
        assert mask_secret(app_key) in caplog.text
    finally:
        SignManager._cache.clear()
        SignManager.clear_locks()

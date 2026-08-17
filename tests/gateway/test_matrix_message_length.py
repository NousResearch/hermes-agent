"""Tests for Matrix outbound message length configuration (#53026)."""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _make_adapter(**extra):
    from plugins.platforms.matrix.adapter import MatrixAdapter

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.example.org",
            "user_id": "@bot:example.org",
            **extra,
        },
    )
    return MatrixAdapter(config)


class TestMatrixMaxMessageLength:
    def test_default_limit_is_16000(self):
        adapter = _make_adapter()
        assert adapter.max_message_length == 16000
        assert adapter._split_threshold == 15900

    def test_extra_override(self):
        adapter = _make_adapter(max_message_length=12000)
        assert adapter.max_message_length == 12000
        assert adapter._split_threshold == 11900

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MATRIX_MAX_MESSAGE_LENGTH", "20000")
        adapter = _make_adapter()
        assert adapter.max_message_length == 20000



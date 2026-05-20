"""Tests for BasePlatformAdapter.build_delivery_metadata default behavior.

The default implementation is a no-op — returns base_metadata unchanged
(or an empty dict if None). Subclasses can override to enrich the
metadata dict passed to adapter.send() during cron delivery.

These tests exercise the default; subclass-override coverage belongs
in the override's own test suite.
"""
from __future__ import annotations

import inspect

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


class _StubAdapter(BasePlatformAdapter):
    """Minimal concrete adapter that does NOT override build_delivery_metadata.

    Used here to exercise the default implementation. Override-side coverage
    is the responsibility of whichever adapter overrides the method.
    """

    def __init__(self):
        # Platform.TELEGRAM is arbitrary — we only need a valid Platform
        # enum value to satisfy BasePlatformAdapter.__init__; the choice
        # has no effect on build_delivery_metadata's behavior.
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def get_chat_info(self, chat_id):  # pragma: no cover — abstract impl
        return None


def test_default_returns_base_metadata_unchanged():
    """The default implementation is a no-op — returns base_metadata as-is."""
    adapter = _StubAdapter()
    base = {"thread_id": "t-1"}
    result = adapter.build_delivery_metadata(
        job={"id": "j", "name": "n"}, status_hint="ok", base_metadata=base
    )
    assert result == base


def test_default_returns_none_when_base_metadata_is_none():
    """base_metadata=None returns None — preserves the pre-hook adapter.send(metadata=None) contract."""
    adapter = _StubAdapter()
    result = adapter.build_delivery_metadata(
        job={"id": "j", "name": "n"}, status_hint="ok", base_metadata=None
    )
    assert result is None


def test_default_returns_copy_not_reference():
    """The default returns a copy — caller mutations don't affect base_metadata."""
    adapter = _StubAdapter()
    base = {"thread_id": "t-1"}
    result = adapter.build_delivery_metadata(
        job={"id": "j", "name": "n"}, status_hint="ok", base_metadata=base
    )
    result["thread_id"] = "mutated"
    assert base["thread_id"] == "t-1"


def test_signature_is_stable():
    """Method signature: (self, job, status_hint='ok', base_metadata=None)."""
    sig = inspect.signature(BasePlatformAdapter.build_delivery_metadata)
    params = list(sig.parameters)
    assert params == ["self", "job", "status_hint", "base_metadata"]
    assert sig.parameters["status_hint"].default == "ok"
    assert sig.parameters["base_metadata"].default is None


def test_default_is_status_hint_agnostic():
    """The default hook returns identical output regardless of status_hint.

    Locks in the contract that the default implementation does NOT
    interpret status_hint. Overriding adapters may interpret it as
    they wish.
    """
    adapter = _StubAdapter()
    base = {"thread_id": "t-1"}
    ok_result = adapter.build_delivery_metadata(
        job={"id": "j", "name": "n"}, status_hint="ok", base_metadata=base
    )
    error_result = adapter.build_delivery_metadata(
        job={"id": "j", "name": "n"}, status_hint="error", base_metadata=base
    )
    assert ok_result == error_result == base

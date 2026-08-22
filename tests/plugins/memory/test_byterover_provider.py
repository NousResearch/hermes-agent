"""Tests for the ByteRover memory provider config gates."""

import pytest

from plugins.memory.byterover import ByteRoverMemoryProvider


def test_auto_extract_false_skips_sync_turn(monkeypatch):
    calls = []
    provider = ByteRoverMemoryProvider({"auto_extract": False})
    provider.initialize("session-1")

    monkeypatch.setattr("plugins.memory.byterover._run_brv", lambda *args, **kwargs: calls.append((args, kwargs)))

    provider.sync_turn("please remember this detail", "acknowledged")

    assert calls == []
    assert provider._sync_thread is None


def test_memory_write_propagates_backend_failure(monkeypatch):
    provider = ByteRoverMemoryProvider({"auto_extract": True})
    provider._auto_extract = True

    def fail(*args, **kwargs):
        raise RuntimeError("byterover unavailable")

    monkeypatch.setattr("plugins.memory.byterover._run_brv", fail)

    with pytest.raises(RuntimeError, match="byterover unavailable"):
        provider.on_memory_write("add", "memory", "remember this")


def test_memory_write_propagates_unsuccessful_cli_result(monkeypatch):
    provider = ByteRoverMemoryProvider({"auto_extract": True})
    provider._auto_extract = True

    monkeypatch.setattr(
        "plugins.memory.byterover._run_brv",
        lambda *args, **kwargs: {"success": False, "error": "backend rejected"},
    )

    with pytest.raises(RuntimeError, match="backend rejected"):
        provider.on_memory_write("add", "memory", "remember this")



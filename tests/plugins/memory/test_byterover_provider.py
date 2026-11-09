"""Tests for the ByteRover memory provider config gates."""

from plugins.memory.byterover import ByteRoverMemoryProvider


def test_auto_extract_false_skips_on_turn_complete(monkeypatch):
    calls = []
    provider = ByteRoverMemoryProvider({"auto_extract": False})
    provider.initialize("session-1")

    monkeypatch.setattr("plugins.memory.byterover._run_brv", lambda *args, **kwargs: calls.append((args, kwargs)))

    provider.on_turn_complete("please remember this detail", "acknowledged", should_store=True)

    assert calls == []
    assert provider._sync_thread is None


def test_sync_turn_no_longer_curates(monkeypatch):
    """The cross-dialogue write moved to on_turn_complete; sync_turn only counts turns."""
    calls = []
    provider = ByteRoverMemoryProvider({"auto_extract": True})
    provider.initialize("session-1")

    monkeypatch.setattr("plugins.memory.byterover._run_brv", lambda *args, **kwargs: calls.append((args, kwargs)))

    provider.sync_turn("a substantive user message about vector stores", "acknowledged")

    assert calls == []
    assert provider._sync_thread is None
    assert provider._turn_count == 1



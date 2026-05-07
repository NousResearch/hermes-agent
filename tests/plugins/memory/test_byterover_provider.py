"""Tests for the ByteRover memory provider config gates."""

from plugins.memory.byterover import ByteRoverMemoryProvider


def test_auto_extract_false_skips_sync_turn(monkeypatch):
    calls = []
    provider = ByteRoverMemoryProvider({"auto_extract": False})
    provider.initialize("session-1")

    monkeypatch.setattr("plugins.memory.byterover._run_brv", lambda *args, **kwargs: calls.append((args, kwargs)))

    provider.sync_turn("please remember this detail", "acknowledged")

    assert calls == []
    assert provider._sync_thread is None


def test_auto_extract_false_skips_memory_write(monkeypatch):
    calls = []
    provider = ByteRoverMemoryProvider({"auto_extract": "false"})
    provider.initialize("session-1")

    monkeypatch.setattr("plugins.memory.byterover._run_brv", lambda *args, **kwargs: calls.append((args, kwargs)))

    provider.on_memory_write("add", "user", "User prefers concise responses")

    assert calls == []


def test_auto_extract_false_skips_pre_compress(monkeypatch):
    calls = []
    provider = ByteRoverMemoryProvider({"auto_extract": "off"})
    provider.initialize("session-1")

    monkeypatch.setattr("plugins.memory.byterover._run_brv", lambda *args, **kwargs: calls.append((args, kwargs)))

    result = provider.on_pre_compress([
        {"role": "user", "content": "remember this"},
        {"role": "assistant", "content": "stored"},
    ])

    assert result == ""
    assert calls == []


def test_auto_extract_false_keeps_explicit_curate_tool(monkeypatch):
    calls = []
    provider = ByteRoverMemoryProvider({"auto_extract": False})
    provider.initialize("session-1")

    def fake_run(args, **kwargs):
        calls.append(args)
        return {"success": True, "output": "ok"}

    monkeypatch.setattr("plugins.memory.byterover._run_brv", fake_run)

    result = provider.handle_tool_call("brv_curate", {"content": "Important project fact"})

    assert "Memory curated successfully" in result
    assert calls == [["curate", "--", "Important project fact"]]

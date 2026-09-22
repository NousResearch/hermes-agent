"""Tests for the Memvid single-file .mv2 memory provider."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from plugins.memory.memvid import MemvidMemoryProvider


def test_available_requires_cli_and_mv2_file(monkeypatch, tmp_path):
    brain = tmp_path / "mind.mv2"
    brain.write_bytes(b"mv2")
    monkeypatch.setattr("plugins.memory.memvid.shutil.which", lambda name: "/usr/bin/memvid")

    provider = MemvidMemoryProvider({"file_path": str(brain)})

    assert provider.is_available() is True


def test_unavailable_without_single_mv2_file(monkeypatch, tmp_path):
    not_mv2 = tmp_path / "mind.sqlite"
    not_mv2.write_text("nope")
    monkeypatch.setattr("plugins.memory.memvid.shutil.which", lambda name: "/usr/bin/memvid")

    provider = MemvidMemoryProvider({"file_path": str(not_mv2)})

    assert provider.is_available() is False


def test_prefetch_searches_and_limits_lines(monkeypatch, tmp_path):
    brain = tmp_path / "mind.mv2"
    brain.write_bytes(b"mv2")
    calls = []
    monkeypatch.setattr("plugins.memory.memvid.shutil.which", lambda name: "/usr/bin/memvid")

    def fake_run(executable, args, timeout=10):
        calls.append((executable, args, timeout))
        return {"success": True, "output": "one\ntwo\nthree"}

    monkeypatch.setattr("plugins.memory.memvid._run_memvid", fake_run)
    provider = MemvidMemoryProvider({"file_path": str(brain), "prefetch_top_k": 2})

    assert provider.prefetch("auth bug") == "## Memvid recall\none\ntwo"
    assert calls == [("/usr/bin/memvid", ["find", str(brain), "auth bug"], 10)]


def test_tool_dispatch_ask_and_stats(monkeypatch, tmp_path):
    brain = tmp_path / "mind.mv2"
    brain.write_bytes(b"mv2")
    monkeypatch.setattr("plugins.memory.memvid.shutil.which", lambda name: "/usr/bin/memvid")
    monkeypatch.setattr(
        "plugins.memory.memvid._run_memvid",
        lambda executable, args, timeout=10: {"success": True, "output": " ".join(args), "timeout": timeout},
    )
    provider = MemvidMemoryProvider({"file_path": str(brain)})

    ask = json.loads(provider.handle_tool_call("memvid_ask", {"question": "why jwt?"}))
    stats = json.loads(provider.handle_tool_call("memvid_stats", {}))

    assert ask["output"] == f"ask {brain} why jwt?"
    assert ask["timeout"] == 30
    assert stats["output"] == f"stats {brain}"


def test_register_calls_register_memory_provider():
    from plugins.memory.memvid import register

    ctx = MagicMock()
    register(ctx)

    ctx.register_memory_provider.assert_called_once()
    assert ctx.register_memory_provider.call_args[0][0].name == "memvid"

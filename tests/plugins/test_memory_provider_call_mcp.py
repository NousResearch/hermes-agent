"""Exclusive memory providers share the native plugin MCP capability gate."""

import pytest

from plugins.memory import load_memory_provider


def test_loaded_provider_uses_native_mcp_context(tmp_path, monkeypatch):
    from tools import mcp_tool_handlers

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n  provider: native-memory\n"
        "plugins:\n  entries:\n    native-memory:\n"
        "      mcp_allowlist: [memory-server]\n",
        encoding="utf-8",
    )
    plugin = tmp_path / "plugins" / "native-memory"
    plugin.mkdir(parents=True)
    plugin.joinpath("__init__.py").write_text(
        "from types import SimpleNamespace\n"
        "def register(ctx):\n"
        "    ctx.register_memory_provider(SimpleNamespace(call=ctx.call_mcp))\n",
        encoding="utf-8",
    )
    calls = []

    def make_handler(server, tool, timeout):
        def handler(arguments):
            calls.append((server, tool, timeout, arguments))
            return '{"result": "stored"}'
        return handler

    monkeypatch.setattr(mcp_tool_handlers, "_make_tool_handler", make_handler)
    provider = load_memory_provider("native-memory")
    assert provider is not None
    assert provider.call("memory-server", "write_note", {"title": "Remember"}, timeout=42) == {
        "ok": True, "result": "stored",
    }
    assert calls == [("memory-server", "write_note", 42.0, {"title": "Remember"})]
    with pytest.raises(PermissionError, match="native-memory.mcp_allowlist"):
        provider.call("ungranted-server", "write_note")
    assert len(calls) == 1

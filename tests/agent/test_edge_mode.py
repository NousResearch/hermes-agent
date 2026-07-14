"""Edge-mode runtime tool-output truncation (string path only)."""

import json
from types import SimpleNamespace

from agent.tool_executor import _edge_mode_truncate_string_tool_result


def test_edge_mode_truncates_long_string() -> None:
    agent = SimpleNamespace(edge_mode=True)
    s = "a" * 5000
    out = _edge_mode_truncate_string_tool_result(agent, s)
    assert len(out) < len(s)
    assert "TRUNCATED BY EDGE-MODE TO PROTECT LOCAL CPU KV-CACHE" in out
    assert out.startswith("a" * 2000)
    assert out.endswith("a" * 1500)


def test_edge_mode_off_passthrough() -> None:
    agent = SimpleNamespace(edge_mode=False)
    s = "a" * 5000
    assert _edge_mode_truncate_string_tool_result(agent, s) is s


def test_edge_mode_short_string_unchanged() -> None:
    agent = SimpleNamespace(edge_mode=True)
    s = "x" * 100
    assert _edge_mode_truncate_string_tool_result(agent, s) is s


def test_edge_mode_truncates_terminal_json() -> None:
    agent = SimpleNamespace(edge_mode=True)
    payload = {
        "output": "o" * 5000,
        "exit_code": 0,
        "error": None,
    }
    raw = json.dumps(payload)
    out = _edge_mode_truncate_string_tool_result(agent, raw, tool_name="terminal")
    data = json.loads(out)
    assert data["exit_code"] == 0
    assert data["error"] is None
    assert data["truncated"] is True
    assert len(data["output"]) < len(payload["output"])


def test_edge_mode_terminal_json_invalid_fallback() -> None:
    agent = SimpleNamespace(edge_mode=True)
    raw = '{"output": "' + ("x" * 5000) + '"'
    out = _edge_mode_truncate_string_tool_result(agent, raw, tool_name="terminal")
    assert len(out) < len(raw)
    assert "TRUNCATED BY EDGE-MODE" in out

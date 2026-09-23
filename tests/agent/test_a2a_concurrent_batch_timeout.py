"""Behavior contract for A2A-aware concurrent batch deadlines (#109089)."""

import json
import os
from pathlib import Path
from types import SimpleNamespace


def _parsed_calls(*calls):
    from agent.tool_executor import _parse_tool_call

    agent = SimpleNamespace()
    return [
        _parse_tool_call(
            agent,
            SimpleNamespace(
                id=f"call-{index}",
                function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
            ),
        )
        for index, (name, arguments) in enumerate(calls)
    ]


def _write_config(text: str) -> None:
    config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")


def test_named_a2a_peers_extend_default_batch_deadline_to_required_max(monkeypatch):
    from agent import tool_executor

    monkeypatch.delenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", raising=False)
    _write_config(
        """
a2a_agents:
  researcher:
    url: http://researcher.test
    timeout: 900
  reviewer:
    url: http://reviewer.test
    timeout: "1200"
  broken:
    url: http://broken.test
    timeout: not-a-number
"""
    )

    assert tool_executor._resolve_concurrent_batch_timeout(
        _parsed_calls(("a2a_call", {"agent": "researcher", "message": "investigate"}))
    ) == 900.0
    assert tool_executor._resolve_concurrent_batch_timeout(
        _parsed_calls(
            ("terminal", {"command": "pwd"}),
            ("a2a_call", {"agent": "researcher", "message": "investigate"}),
            ("a2a_call", {"agent": "reviewer", "message": "review"}),
            ("a2a_call", {"agent": "broken", "message": "ignore invalid timeout"}),
        )
    ) == 1200.0

    captured = {}

    class _BatchProbe:
        results = []

        def __init__(self, _agent, _messages, _task_id, _parsed_calls, timeout_s):
            captured["timeout_s"] = timeout_s

        def run(self):
            return None

    monkeypatch.setattr(tool_executor, "_ConcurrentBatch", _BatchProbe)
    monkeypatch.setattr(tool_executor, "_append_batch_results", lambda *_args: True)
    agent = SimpleNamespace(
        _interrupt_requested=False,
        quiet_mode=True,
        _should_emit_quiet_tool_messages=lambda: False,
        _touch_activity=lambda _message: None,
    )
    assistant_message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id="call-dispatch",
                function=SimpleNamespace(
                    name="a2a_call",
                    arguments=json.dumps({"agent": "researcher", "message": "investigate"}),
                ),
            )
        ]
    )
    tool_executor.execute_tool_calls_concurrent(
        agent, assistant_message, [], "task", finalize=False
    )
    assert captured["timeout_s"] == 900.0


def test_a2a_extension_fails_open_and_never_overrides_global_timeout_sources(monkeypatch):
    from agent.tool_executor import (
        _resolve_concurrent_batch_timeout,
        _resolve_concurrent_tool_timeout,
    )

    calls = _parsed_calls(("a2a_call", {"agent": "long", "message": "work"}))
    monkeypatch.delenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", raising=False)
    _write_config(
        """
a2a_agents:
  long:
    url: http://long.test
    timeout: 900
  invalid:
    url: http://invalid.test
    timeout: -5
  no_url:
    timeout: 900
"""
    )

    assert _resolve_concurrent_tool_timeout() == 420.0
    assert _resolve_concurrent_batch_timeout(
        _parsed_calls(("a2a_call", {"agent": "https://direct.test", "message": "work"}))
    ) == 420.0
    assert _resolve_concurrent_batch_timeout(
        _parsed_calls(("a2a_call", {"agent": "unknown", "message": "work"}))
    ) == 420.0
    assert _resolve_concurrent_batch_timeout(
        _parsed_calls(("a2a_call", {"agent": "invalid", "message": "work"}))
    ) == 420.0
    assert _resolve_concurrent_batch_timeout(
        _parsed_calls(("a2a_call", {"agent_name": "long", "message": "work"}))
    ) == 420.0
    assert _resolve_concurrent_batch_timeout(
        _parsed_calls(("a2a_call", {"agent": "long"}))
    ) == 420.0
    assert _resolve_concurrent_batch_timeout(
        _parsed_calls(("a2a_call", {"agent": "no_url", "message": "work"}))
    ) == 420.0
    assert _resolve_concurrent_batch_timeout(
        _parsed_calls(("a2a_call", {"agent": "long", "name": "other", "message": "work"}))
    ) == 420.0

    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "300")
    assert _resolve_concurrent_batch_timeout(calls) == 300.0
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "0")
    assert _resolve_concurrent_batch_timeout(calls) is None

    _write_config(
        """
timeouts:
  tools:
    concurrent_batch: 240
a2a_agents:
  long:
    url: http://long.test
    timeout: 900
"""
    )
    monkeypatch.setenv("HERMES_CONCURRENT_TOOL_TIMEOUT_S", "100")
    assert _resolve_concurrent_batch_timeout(calls) == 240.0

    _write_config(
        """
timeouts:
  tools:
    concurrent_batch: 0
a2a_agents:
  long:
    url: http://long.test
    timeout: 900
"""
    )
    assert _resolve_concurrent_batch_timeout(calls) is None

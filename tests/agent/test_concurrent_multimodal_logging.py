"""The concurrent completion log reports the serialized size of multimodal results.

The native vision fast path returns an envelope dict; the concurrent worker logged
``len(result)`` directly, so a ~100 KB image payload showed up as ``completed
(0.14s, 4 chars)`` — the dict key count — while the sequential path logged the real
serialized size (#112095).
"""

import logging
from unittest.mock import MagicMock

from tests.agent.test_start_order_gate import (  # noqa: F401 — autouse fixture rides along
    _FakeAssistantMsg,
    _FakeToolCall,
    _isolate_hermes,
    _make_agent,
)


def test_concurrent_completion_log_reports_serialized_multimodal_size(monkeypatch, caplog):
    agent = _make_agent(monkeypatch)
    import agent.tool_executor as te

    monkeypatch.setattr(te, "_resolve_concurrent_tool_timeout", lambda: 6.0)
    envelope = {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": "Image loaded into your context — " + "x" * 800},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "A" * 4000}},
        ],
        "text_summary": "Image attached natively for the main model.",
        "meta": {"image_url": "photo.jpg", "size_bytes": 204800, "native_vision": True},
    }
    agent._tool_guardrails = MagicMock()
    agent._tool_guardrails.before_call = lambda *a, **kw: MagicMock(allows_execution=True)
    agent._invoke_tool = MagicMock(return_value=envelope)

    msg = _FakeAssistantMsg([_FakeToolCall("vision_analyze", "tc_1")])
    with caplog.at_level(logging.INFO, logger="agent.tool_executor"):
        agent._execute_tool_calls_concurrent(msg, [], "task")

    completed = [r.getMessage() for r in caplog.records if "vision_analyze completed (" in r.getMessage()]
    assert completed, "no completion log line for the concurrent vision_analyze call"
    # Same measurement as the sequential path's success_log_chars.
    assert f", {len(str(envelope))} chars)" in completed[0], completed[0]

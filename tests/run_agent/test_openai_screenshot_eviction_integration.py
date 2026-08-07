"""Integration: request-build-time screenshot eviction on chat_completions.

Drives ``run_conversation`` with a screenshot-heavy history and captures the
outbound API payload. Asserts the wiring in ``agent/conversation_loop.py``:

  * the payload sent to the provider keeps only the most recent 3
    image-bearing tool results (older ones placeholdered), and
  * the stored conversation history keeps every image — eviction operates
    on the per-call copy only (prompt-caching invariant: never mutate past
    context).

Unit-level semantics of ``_evict_old_screenshots_openai`` are covered in
tests/agent/test_openai_screenshot_eviction.py; the Anthropic-side
equivalent lives in tests/tools/test_computer_use.py.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import run_agent


FAKE_PNG = "iVBORw0KGgo="


def _patch_bootstrap(monkeypatch):
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [{
        "type": "function",
        "function": {"name": "t", "description": "t", "parameters": {"type": "object", "properties": {}}},
    }])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _final_text_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=SimpleNamespace(
            role="assistant", content="ok", tool_calls=None, reasoning_content=None,
        ), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        model="test-model",
    )


def _make_agent(monkeypatch, captured_kwargs: List[Dict[str, Any]]):
    _patch_bootstrap(monkeypatch)

    class _A(run_agent.AIAgent):
        def __init__(self, *a, **kw):
            kw.update(skip_context_files=True, skip_memory=True, max_iterations=4)
            super().__init__(*a, **kw)
            self._cleanup_task_resources = self._persist_session = lambda *a, **k: None
            self._save_trajectory = lambda *a, **k: None

        def _model_supports_vision(self):
            # Keep the non-vision image-strip fallback out of the way so the
            # captured payload reflects the eviction transform alone.
            return True

        def run_conversation(self, msg, conversation_history=None, task_id=None):
            def _capture(kw):
                captured_kwargs.append(kw)
                return _final_text_response()

            self._interruptible_api_call = _capture
            self._disable_streaming = True
            return super().run_conversation(
                msg, conversation_history=conversation_history, task_id=task_id,
            )

    return _A(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:1234/v1",
        provider="openrouter",
        api_mode="chat_completions",
    )


def _image_part(tag: str) -> Dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{FAKE_PNG}{tag}"},
    }


def _screenshot_history(n: int) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = [{"role": "user", "content": "start"}]
    for i in range(n):
        history.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "computer_use", "arguments": "{}"},
            }],
        })
        history.append({
            "role": "tool",
            "name": "computer_use",
            "tool_call_id": f"call_{i}",
            "content": [
                {"type": "text", "text": f"cap {i}"},
                _image_part(str(i)),
            ],
        })
    history.append({"role": "assistant", "content": "done"})
    return history


def _tool_messages(payload_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [m for m in payload_messages if m.get("role") == "tool"]


def _has_image(msg: Dict[str, Any]) -> bool:
    content = msg.get("content")
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in content
    )


def _has_placeholder(msg: Dict[str, Any]) -> bool:
    content = msg.get("content")
    return isinstance(content, list) and any(
        isinstance(p, dict)
        and p.get("type") == "text"
        and "screenshot removed" in p.get("text", "")
        for p in content
    )


def test_outbound_payload_evicts_old_screenshots_history_keeps_them(monkeypatch):
    captured: List[Dict[str, Any]] = []
    agent = _make_agent(monkeypatch, captured)

    history = _screenshot_history(5)
    agent.run_conversation("continue", conversation_history=history)

    assert captured, "expected at least one API call"
    sent = _tool_messages(captured[0]["messages"])
    assert len(sent) == 5

    with_images = [m for m in sent if _has_image(m)]
    placeholdered = [m for m in sent if _has_placeholder(m)]
    assert len(with_images) == 3
    assert len(placeholdered) == 2
    # The oldest two were the ones evicted.
    assert {m["tool_call_id"] for m in placeholdered} == {"call_0", "call_1"}
    # Captions survive next to the placeholder.
    for m in placeholdered:
        assert any(
            p.get("type") == "text" and p.get("text", "").startswith("cap ")
            for p in m["content"]
        )

    # The stored history the caller handed in still carries every image.
    stored_tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert len(stored_tool_msgs) == 5
    assert all(_has_image(m) for m in stored_tool_msgs)
    assert not any(_has_placeholder(m) for m in stored_tool_msgs)

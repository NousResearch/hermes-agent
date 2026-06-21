from __future__ import annotations

import json

from gateway.context_dump import (
    agent_history_for_context_dump,
    context_dump_slug,
    write_context_dump_payload,
    write_context_dump_text,
)
from gateway.context_layers import (
    ContextLayer,
    layers_to_payload,
    total_estimated_tokens,
)


def test_context_dump_slug_is_bounded_and_filesafe() -> None:
    raw = "discord:thread/with spaces?" + "x" * 300

    slug = context_dump_slug(raw)

    assert "/" not in slug
    assert " " not in slug
    assert len(slug) == 180


def test_write_context_dump_text_includes_layers_and_token_total(tmp_path) -> None:
    payload = {
        "captured_at": "2026-06-21T00:00:00Z",
        "capture_mode": "zero_inference_current_context",
        "session_key": "discord:1",
        "session_id": "s1",
        "model": "openai/gpt-oss-20b",
        "provider": "openrouter",
        "estimated_total_tokens": 123,
        "context_layers": [
            ContextLayer.from_text("soul", source="SOUL.md", text="hello").to_payload()
        ],
        "api_messages": [{"role": "system", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "chat_startup_context"}}],
    }

    path = write_context_dump_text(tmp_path, "discord:1", payload)

    text = path.read_text(encoding="utf-8")
    assert "Estimated total tokens: 123" in text
    assert "## Context Layers" in text
    assert '"id": "soul"' in text
    assert "## Raw API Messages" in text


def test_write_context_dump_payload_is_private_json(tmp_path) -> None:
    path = write_context_dump_payload(tmp_path, "session", {"schema": "test"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"schema": "test"}
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_agent_history_for_context_dump_filters_meta_and_preserves_tools() -> None:
    history = [
        {"role": "system", "content": "skip"},
        {"role": "session_meta", "content": "skip"},
        {"role": "user", "content": "hi", "timestamp": "ignored"},
        {"role": "assistant", "content": "there", "reasoning": "kept"},
        {
            "role": "tool",
            "content": "result",
            "tool_call_id": "t1",
            "timestamp": "ignored",
        },
        {
            "role": "user",
            "content": "mirrored",
            "mirror": True,
            "mirror_source": "other",
        },
    ]

    result = agent_history_for_context_dump(history)

    assert result == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "there", "reasoning": "kept"},
        {"role": "tool", "content": "result", "tool_call_id": "t1"},
        {"role": "user", "content": "[Delivered from other] mirrored"},
    ]


def test_context_layers_payload_and_total() -> None:
    layers = [
        ContextLayer.from_text("soul", source="SOUL.md", text="abcd"),
        ContextLayer.from_text("optional", source="skill", text="abcd", enabled=False),
    ]

    assert layers_to_payload(layers)[0]["id"] == "soul"
    assert total_estimated_tokens(layers) == 1

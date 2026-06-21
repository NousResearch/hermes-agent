from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gateway.context_dump import (
    agent_history_for_context_dump,
    context_dump_path,
    context_dump_slug,
    context_dump_text_path,
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


@pytest.mark.asyncio
async def test_context_dump_command_refreshes_text_from_actual_v2_payload(
    monkeypatch, tmp_path
) -> None:
    from gateway.config import Platform
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner

    monkeypatch.setenv("HERMES_CONTEXT_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("DISCORD_SLASH_OWNER_IDS", "owner")
    session_key = "discord:test"
    text_path = context_dump_text_path(tmp_path, session_key)
    text_path.write_text("stale text", encoding="utf-8")
    payload = {
        "schema": "hermes.context_dump.v2",
        "capture_mode": "actual_agent_run",
        "phase": "after_run",
        "session_key": session_key,
        "session_id": "s1",
        "estimated_total_tokens": 42,
        "context_layers": [
            ContextLayer.from_text("soul", source="SOUL.md", text="SOUL").to_payload()
        ],
        "api_messages": [{"role": "system", "content": "SOUL"}],
        "tools": [],
    }
    context_dump_path(tmp_path, session_key).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    sent = {}

    class Adapter:
        async def send_document(self, **kwargs):
            sent["path"] = kwargs["file_path"]
            return SendResult(success=True, message_id="dump-message")

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: Adapter()}
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_key=session_key)
    )
    source = SimpleNamespace(
        platform=Platform.DISCORD,
        user_id="owner",
        chat_id="chat",
        thread_id=None,
    )
    event = SimpleNamespace(source=source, message_id="request-message")

    result = await runner._handle_context_dump_command(event)

    assert result == "Sent the current zero-inference context dump."
    assert sent["path"] == str(text_path)
    assert "Estimated total tokens: 42" in text_path.read_text(encoding="utf-8")

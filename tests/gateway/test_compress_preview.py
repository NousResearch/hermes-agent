"""Tests for gateway /compress --preview/--dry-run/--aggressive flags
(PR #3243 salvage).

The preview path must return a report WITHOUT building an agent or
touching the transcript; --aggressive must return an explanatory
message rather than being mis-parsed as a focus topic.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_history(n_pairs: int = 3) -> list[dict[str, str]]:
    h: list[dict[str, str]] = []
    for i in range(n_pairs):
        h.append({"role": "user", "content": f"u{i}"})
        h.append({"role": "assistant", "content": f"a{i}"})
    return h


def _make_runner(history: list[dict[str, str]]):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store._save = MagicMock()
    runner._session_db = None
    return runner


@pytest.mark.asyncio
async def test_preview_with_here_boundary():
    runner = _make_runner(_make_history(4))
    result = await runner._handle_compress_command(
        _make_event("/compress --preview here 2")
    )
    assert "last 2 exchange" in result
    assert "4 of 8" in result
    runner.session_store.rewrite_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_aggressive_dry_run_shows_preview_plus_note():
    runner = _make_runner(_make_history(3))
    result = await runner._handle_compress_command(
        _make_event("/compress --aggressive --dry-run")
    )
    assert "no changes made" in result.lower()
    assert "--aggressive is not supported" in result
    runner.session_store.rewrite_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_preview_feeds_tool_messages_system_prompt_and_tools_to_estimator():
    """The preview must feed the shared estimator the same payload buckets
    as the real compression path and the CLI's preview branch: tool-result
    messages, the session's system prompt, and the enabled tool schemas
    (#98360). Before the fix, the gateway preview filtered the transcript
    to user/assistant messages only and passed neither system_prompt nor
    tools, so it could report ~100x fewer tokens than the CLI reports for
    the identical session state.
    """
    history = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t1", "type": "function",
                            "function": {"name": "x", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "BIG RESULT " * 500, "tool_call_id": "t1"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "np"},
    ]
    runner = _make_runner(history)
    runner._session_db = MagicMock()
    runner._session_db.get_session = AsyncMock(
        return_value={"system_prompt": "SYSTEM PROMPT " * 100}
    )
    fake_tools = [{
        "type": "function",
        "function": {
            "name": "t0", "description": "d",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    runner._resolve_enabled_toolsets_for_source = MagicMock(return_value=["memory"])

    captured = {}

    def _capture(messages, **kwargs):
        captured["messages"] = messages
        captured["system_prompt"] = kwargs.get("system_prompt")
        captured["tools"] = kwargs.get("tools")
        return 42

    with (
        patch("gateway.run._load_gateway_config", return_value={}),
        patch("model_tools.get_tool_definitions", return_value=fake_tools),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_capture),
    ):
        result = await runner._handle_compress_command(
            _make_event("/compress --preview")
        )

    assert "42" in result
    roles = [m.get("role") for m in captured["messages"]]
    assert "tool" in roles, f"tool-result messages dropped from the preview estimate: {roles}"
    assert captured["system_prompt"], "system prompt dropped from the preview estimate"
    assert captured["tools"] == fake_tools, "tool schemas dropped from the preview estimate"


@pytest.mark.asyncio
async def test_preview_labels_lower_bound_when_tool_schema_lookup_fails():
    """If tool-schema resolution raises, the preview must say the number
    is a lower bound instead of quietly reporting a partial estimate as
    if it were the full context (reviewer feedback on #98368).
    """
    runner = _make_runner(_make_history(3))
    runner._session_db = MagicMock()
    runner._session_db.get_session = AsyncMock(
        return_value={"system_prompt": "SYSTEM PROMPT"}
    )
    runner._resolve_enabled_toolsets_for_source = MagicMock(
        side_effect=RuntimeError("boom")
    )

    with patch("gateway.run._load_gateway_config", return_value={}):
        result = await runner._handle_compress_command(
            _make_event("/compress --preview")
        )

    assert "lower bound" in result.lower(), result
    assert "tool schemas" in result



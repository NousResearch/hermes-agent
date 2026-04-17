from __future__ import annotations

from contextlib import contextmanager

from gateway.agent_execution_service import (
    collect_history_media_paths,
    gateway_approval_context,
    normalize_conversation_history,
)


def test_normalize_conversation_history_preserves_tool_sequences_and_reasoning():
    history = [
        {"role": "session_meta", "content": "skip"},
        {"role": "system", "content": "skip"},
        {"role": "user", "content": "hello", "timestamp": "t1"},
        {
            "role": "assistant",
            "content": "thinking",
            "timestamp": "t2",
            "reasoning": "chain",
            "reasoning_details": [{"type": "summary", "text": "r"}],
        },
        {
            "role": "assistant",
            "tool_calls": [{"id": "call-1", "type": "function"}],
            "content": "",
            "timestamp": "t3",
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": 'ok MEDIA:/tmp/a.png MEDIA:/tmp/b.mp3"}',
            "timestamp": "t4",
        },
        {
            "role": "user",
            "content": "copied",
            "mirror": True,
            "mirror_source": "weixin",
        },
        {"role": "assistant", "content": ""},
    ]

    normalized = normalize_conversation_history(history)

    assert normalized == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "thinking",
            "reasoning": "chain",
            "reasoning_details": [{"type": "summary", "text": "r"}],
        },
        {
            "role": "assistant",
            "tool_calls": [{"id": "call-1", "type": "function"}],
            "content": "",
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": 'ok MEDIA:/tmp/a.png MEDIA:/tmp/b.mp3"}',
        },
        {"role": "user", "content": "[Delivered from weixin] copied"},
    ]


def test_collect_history_media_paths_extracts_unique_paths_from_tool_messages():
    agent_history = [
        {"role": "user", "content": "ignore"},
        {"role": "tool", "content": 'foo MEDIA:/tmp/a.png bar MEDIA:/tmp/b.mp3"}'},
        {"role": "function", "content": "MEDIA:/tmp/a.png"},
    ]

    assert collect_history_media_paths(agent_history) == {"/tmp/a.png", "/tmp/b.mp3"}


def test_gateway_approval_context_sets_and_resets_context(monkeypatch):
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        "tools.approval.set_current_session_key",
        lambda value: calls.append(("set_session", value)) or "session-token",
    )
    monkeypatch.setattr(
        "tools.approval.reset_current_session_key",
        lambda token: calls.append(("reset_session", token)),
    )
    monkeypatch.setattr(
        "tools.approval.set_current_admin_policy",
        lambda ids, flag: calls.append(("set_admin", (list(ids), flag))) or "admin-token",
    )
    monkeypatch.setattr(
        "tools.approval.reset_current_admin_policy",
        lambda token: calls.append(("reset_admin", token)),
    )
    monkeypatch.setattr(
        "tools.approval.set_external_approval_backend",
        lambda backend: calls.append(("set_backend", backend)) or "backend-token",
    )
    monkeypatch.setattr(
        "tools.approval.reset_external_approval_backend",
        lambda token: calls.append(("reset_backend", token)),
    )

    backend = object()
    with gateway_approval_context(
        session_key="agent:main:test",
        admin_user_ids=["179033731"],
        is_admin_user=True,
        external_backend=backend,
    ):
        calls.append(("inside", None))

    assert calls == [
        ("set_session", "agent:main:test"),
        ("set_admin", (["179033731"], True)),
        ("set_backend", backend),
        ("inside", None),
        ("reset_backend", "backend-token"),
        ("reset_admin", "admin-token"),
        ("reset_session", "session-token"),
    ]

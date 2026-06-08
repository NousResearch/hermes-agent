"""Regression coverage for QQ runtime reconciliation patches.

These tests lock the live-runtime fixes that this user's QQ gateway depends on:
Chinese QQ-facing status text, indexed multi-bot credentials, and scoped
approval lookup for mode-routed sessions.
"""

from __future__ import annotations

from gateway.config import Platform, PlatformConfig


def test_qqbot_status_messages_are_localized_to_chinese():
    from gateway.run import _prepare_gateway_status_message

    samples = [
        (
            "⚡ Interrupting current task (7 min elapsed, iteration 16/90, running: execute_code). "
            "I'll respond to your message shortly.",
            "正在中断当前任务",
            "Interrupting current task",
        ),
        (
            "📦 Preflight compression: ~252,573 tokens >= 240,000 threshold. "
            "This may take a moment.",
            "预先压缩 context",
            "This may take a moment",
        ),
        (
            "⏳ Working — 9 min — iteration 6/90, receiving stream response",
            "正在处理",
            "Working",
        ),
    ]

    for raw, expected_zh, forbidden_en in samples:
        rendered = _prepare_gateway_status_message(Platform.QQBOT, "status", raw)
        assert expected_zh in rendered
        assert forbidden_en not in rendered


def test_qqbot_multi_adapter_collects_indexed_credentials(monkeypatch):
    monkeypatch.delenv("QQ_APP_ID", raising=False)
    monkeypatch.delenv("QQ_CLIENT_SECRET", raising=False)
    for idx in range(10):
        monkeypatch.delenv(f"QQ_APP_ID_{idx}", raising=False)
        monkeypatch.delenv(f"QQ_CLIENT_SECRET_{idx}", raising=False)
    monkeypatch.setenv("QQ_APP_ID_0", "bot-a")
    monkeypatch.setenv("QQ_CLIENT_SECRET_0", "secret-a")
    monkeypatch.setenv("QQ_APP_ID_1", "bot-b")
    monkeypatch.setenv("QQ_CLIENT_SECRET_1", "secret-b")

    from gateway.platforms.qqbot import (
        QQMultiAdapter,
        collect_qq_credentials,
        has_any_qq_credentials,
    )

    config = PlatformConfig(enabled=True, extra={"markdown_support": True})
    assert has_any_qq_credentials(config) is True
    assert collect_qq_credentials(config) == [
        ("env:0", "bot-a", "secret-a"),
        ("env:1", "bot-b", "secret-b"),
    ]

    adapter = QQMultiAdapter(config)
    assert adapter.name == "QQBotMulti"
    assert len(adapter._children) == 2


def test_approval_lookup_falls_back_to_scoped_mode_session():
    from tools import approval
    from tools.approval import find_blocking_approval_session_key

    base_key = "agent:main:qqbot:dm:chat-1"
    scoped_key = f"{base_key}:mode:ops"
    with approval._lock:
        approval._gateway_queues.clear()
        approval._gateway_queues[scoped_key] = [approval._ApprovalEntry({"command": "rm -rf /tmp/x"})]

    try:
        assert find_blocking_approval_session_key(base_key) == scoped_key
        assert find_blocking_approval_session_key(scoped_key) == scoped_key
        assert find_blocking_approval_session_key("agent:main:qqbot:dm:other") is None
    finally:
        with approval._lock:
            approval._gateway_queues.clear()

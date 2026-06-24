"""Tests for Discord natural-language kanban intake."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
import gateway.run as gateway_run
from hermes_cli import kanban_db as kb


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-1",
        chat_type="channel",
        chat_name="ops",
        user_id="user-1",
        user_name="Uchida",
        thread_id="thread-1",
        message_id="source-message-1",
    )


def _runner(tmp_path, monkeypatch, *, enabled=True):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = {
        "kanban": {
            "discord_natural_intake": {
                "enabled": enabled,
                "board": "intake-test",
                "default_assignee": "operations-orchestrator",
                "priority": 25,
            }
        }
    }
    runner._kanban_notifier_profile = None
    runner._active_profile_name = lambda: "gateway-profile"
    return runner


def test_discord_natural_intake_ignores_questions():
    decision = gateway_run._classify_discord_natural_kanban_request(
        "Hermesのカンバンってどういう仕組みですか？"
    )

    assert decision.should_register is False


def test_discord_natural_intake_ignores_polite_explanation_requests():
    for text in [
        "教えてください、お願いします",
        "レビュー観点を教えてくださいお願いします",
    ]:
        decision = gateway_run._classify_discord_natural_kanban_request(text)

        assert decision.should_register is False


@pytest.mark.asyncio
async def test_maybe_register_discord_natural_task_skips_original_slash_command_after_rewrite(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    event = MessageEvent(
        text="レビュー観点を教えてくださいお願いします",
        source=_source(),
        message_id="slash-message-1",
    )
    event._hermes_original_slash_command = True

    assert await runner._maybe_register_discord_natural_task(event) is None


def test_discord_natural_intake_detects_work_request():
    decision = gateway_run._classify_discord_natural_kanban_request(
        "Discord通常依頼をカンバンに入れる実装を調査して対応しておいて"
    )

    assert decision.should_register is True
    assert decision.initial_status == "triage"
    assert decision.risk_level == "R1"


def test_discord_natural_intake_blocks_risky_request():
    decision = gateway_run._classify_discord_natural_kanban_request(
        "GitHubにpushしてVPS本番へ反映してからDiscordに送信して"
    )

    assert decision.should_register is True
    assert decision.initial_status == "blocked"
    assert decision.risk_level == "R3"


def test_discord_natural_intake_redacts_secret_like_text():
    redacted = gateway_run._redact_discord_natural_kanban_text(
        "APIキー sk-testsecret1234567890abcdef と token=abcdef1234567890abcdef1234567890 を使って"
    )

    assert "sk-testsecret" not in redacted
    assert "abcdef1234567890abcdef1234567890" not in redacted
    assert "[REDACTED]" in redacted


@pytest.mark.asyncio
async def test_maybe_register_discord_natural_task_creates_triage_card(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    event = MessageEvent(
        text="ログを確認して原因を調査してカンバンに入れて",
        source=_source(),
        message_id="source-message-1",
    )

    ack = await runner._maybe_register_discord_natural_task(event)

    assert ack is not None
    assert "カンバンに登録しました" in ack
    task_id = ack.split("`", 2)[1]
    conn = kb.connect(board="intake-test")
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.assignee == "operations-orchestrator"
        assert task.status == "triage"
        assert task.priority == 25
        assert task.created_by == "discord-natural-intake"
        assert "Discord由来" in (task.body or "")
        assert "source-message-1" in (task.body or "")
        subs = conn.execute(
            "SELECT platform, chat_id, thread_id, user_id FROM notify_subscriptions WHERE task_id = ?",
            (task_id,),
        ).fetchall()
        assert [(r["platform"], r["chat_id"], r["thread_id"], r["user_id"]) for r in subs] == [
            ("discord", "channel-1", "thread-1", "user-1")
        ]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_maybe_register_discord_natural_task_is_idempotent_by_message_id(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    event = MessageEvent(
        text="リノ口調の安定化を調査して対応しておいて",
        source=_source(),
        message_id="source-message-1",
    )

    first = await runner._maybe_register_discord_natural_task(event)
    second = await runner._maybe_register_discord_natural_task(event)

    first_id = first.split("`", 2)[1]
    second_id = second.split("`", 2)[1]
    assert second_id == first_id
    conn = kb.connect(board="intake-test")
    try:
        rows = conn.execute("SELECT id FROM tasks WHERE idempotency_key LIKE ?", ("discord-natural:%",)).fetchall()
        assert [r["id"] for r in rows] == [first_id]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_maybe_register_discord_natural_task_skips_when_disabled(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, enabled=False)
    event = MessageEvent(text="調査して対応しておいて", source=_source())

    assert await runner._maybe_register_discord_natural_task(event) is None

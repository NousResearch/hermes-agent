"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

from types import SimpleNamespace
import inspect
from unittest.mock import AsyncMock

from gateway.kanban_watchers import (
    GatewayKanbanWatchersMixin,
    _format_blocked_notification,
    _format_completed_notification,
)

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_gateway_runner_inherits_mixin():
    # Import here so a heavy gateway import only happens if the first test passed.
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one holder of the dispatcher lock at a time — the backstop that
    stops concurrent dispatchers double reclaiming and corrupting shared
    kanban SQLite index pages under wal_autocheckpoint=0."""
    import os

    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    h1, st1 = _acquire_singleton_lock(lock)
    assert st1 == "held" and h1 is not None

    # A second acquire while the first is held must be refused, not granted.
    h2, st2 = _acquire_singleton_lock(lock)
    assert st2 == "contended" and h2 is None

    # Releasing the first lets a fresh acquire succeed (lock is reusable).
    _release_singleton_lock(h1)
    h3, st3 = _acquire_singleton_lock(lock)
    assert st3 == "held" and h3 is not None
    _release_singleton_lock(h3)


def test_blocked_notification_uses_platform_budget_instead_of_160_char_cutoff():
    reason = (
        "ClawOps capability unavailable: current OpenClaw bridge rejected "
        "Facebook/external browser execution as dry-run only "
        "(`external_capability_unavailable`); Hermes is contractually "
        "prohibited from doing this UI work directly."
    )

    msg = _format_blocked_notification(
        task_id="t_e91d380a",
        tag="@default ",
        reason=reason,
        platform_limit=4096,
    )

    assert msg.endswith("directly.")
    assert "`external_capability_unavailable`" in msg


def test_completed_notification_surfaces_multiline_publish_result_in_chat():
    result = """本次已發佈清單

1
• 社團名稱: 餐飲美食交流.經營管理.二手中古.新舊餐飲設備買賣.頂讓.
• 成員數: 58.4K
• 類型: Public
• 狀態: 已送出 / 已分享

Facebook 畫面驗證

顯示：
Your listing has been shared to 餐飲美食交流.經營管理.二手中古.新舊餐飲設備買賣.頂讓.

這次沒有做的事

- 沒有建立新的 Marketplace 商品
- 沒有 Join group
- 沒有 Promote / 付款
- 沒有私訊、留言
"""

    msg = _format_completed_notification(
        task_id="t_publish",
        title="ClawOps: Facebook 發佈",
        tag="@clawops-browser ",
        payload_summary="",
        task_result=result,
        platform_limit=4096,
    )

    assert "本次已發佈清單" in msg
    assert "餐飲美食交流.經營管理" in msg
    assert "Facebook 畫面驗證" in msg
    assert "Your listing has been shared" in msg
    assert "沒有建立新的 Marketplace 商品" in msg


async def _deliver_artifacts(adapter, tmp_path, artifacts):
    runner = SimpleNamespace()
    await GatewayKanbanWatchersMixin._deliver_kanban_artifacts(
        runner,
        adapter=adapter,
        chat_id="chat-1",
        metadata={},
        event_payload={"artifacts": artifacts},
        task=None,
    )


def test_kanban_artifact_delivery_warns_when_explicit_artifact_missing(tmp_path):
    from gateway.platforms.base import BasePlatformAdapter

    adapter = SimpleNamespace(
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(),
        send_document=AsyncMock(),
        send_multiple_images=AsyncMock(),
    )
    missing = str(tmp_path / "missing_report.md")

    import asyncio

    asyncio.run(_deliver_artifacts(adapter, tmp_path, [missing]))

    adapter.send.assert_awaited_once()
    warning = adapter.send.await_args.args[1]
    assert "artifact missing" in warning
    assert missing in warning
    adapter.send_document.assert_not_awaited()


def test_kanban_artifact_delivery_uploads_existing_markdown(tmp_path):
    from gateway.platforms.base import BasePlatformAdapter

    report = tmp_path / "report.md"
    report.write_text("# report\n", encoding="utf-8")
    adapter = SimpleNamespace(
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(),
        send_document=AsyncMock(),
        send_multiple_images=AsyncMock(),
    )

    import asyncio

    asyncio.run(_deliver_artifacts(adapter, tmp_path, [str(report)]))

    adapter.send.assert_not_awaited()
    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(report),
        metadata={},
    )

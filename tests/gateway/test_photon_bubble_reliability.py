"""AirHermes Phase 6 Reliability Regression Tests (Stage 2 & Stage 3).

Covers:
1. DEFECT-01: BubbleStreamConsumer cancellation lifecycle & worker leak prevention
   - Cancellation while waiting for deltas
   - Cancellation with empty queue
   - Cancellation with queued bubbles
   - Cancellation while native send is active
   - Cancellation during queue draining
   - Repeated cancellation across 20 consecutive turns
   - Cancellation followed by immediate creation of a new turn
   - Multi-conversation isolation (cancelling conversation A leaves B running)
2. DEFECT-02: enable_maps feature flag enforcement
   - <map> and <location> structured tags
   - Self-closing tags
   - Coordinate-based locations
   - Ordinary prose URLs preservation
   - Streaming vs non-streaming format_message fallback
   - Isolation from other semantic features (calendar, contact, poll)
   - enable_maps=True preserves VULN-05 URL sanitization
3. DEFECT-03: Windows Sidecar port conflict diagnostics
   - Detection of occupied port
   - Safe diagnostic reporting without leaking secrets
   - Refusal of unproven termination
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import unittest
from typing import Any

from plugins.platforms.photon.adapter import PhotonAdapter, SendResult

from gateway.airhermes_config import AirHermesConfig
from gateway.bubble_actions import EffectAction, FileAttachmentAction, PollAction, TextBubble
from gateway.bubble_parser import BubbleParser
from gateway.bubble_stream_consumer import BubbleStreamConsumer


class MockReliabilityAdapter:
    """Mock adapter for reliability tests."""

    def __init__(self, send_delay: float = 0.0) -> None:
        self.dispatched: list[tuple[str, Any]] = []
        self.send_delay = send_delay
        self.in_send_event = asyncio.Event()
        self.block_send_event = asyncio.Event()

    async def send_raw_text(self, chat_id: str, content: str, **kwargs: Any) -> SendResult:
        self.in_send_event.set()
        if self.block_send_event.is_set():
            await self.block_send_event.wait()
        if self.send_delay > 0:
            await asyncio.sleep(self.send_delay)  # health: allow HX013 -- mock adapter delay
        self.dispatched.append((chat_id, content))
        return SendResult(success=True, message_id=f"msg_{len(self.dispatched)}")

    async def send(self, chat_id: str, content: str, **kwargs: Any) -> SendResult:
        return await self.send_raw_text(chat_id, content, **kwargs)

    async def send_document(self, chat_id: str, file_path: str, **kwargs: Any) -> SendResult:
        self.dispatched.append((chat_id, file_path))
        return SendResult(success=True, message_id=f"doc_{len(self.dispatched)}")

    async def _sidecar_send_poll(self, chat_id: str, title: str, options: list[str]) -> SendResult:
        self.dispatched.append((chat_id, f"POLL:{title}"))
        return SendResult(success=True, message_id=f"poll_{len(self.dispatched)}")

    async def _sidecar_send_effect(self, chat_id: str, text: str, effect: str) -> SendResult:
        self.dispatched.append((chat_id, f"EFFECT:{effect}:{text}"))
        return SendResult(success=True, message_id=f"eff_{len(self.dispatched)}")


class TestDefect01ConsumerCancellation(unittest.IsolatedAsyncioTestCase):
    """DEFECT-01: BubbleStreamConsumer cancellation must terminate owned worker task."""

    async def test_cancellation_waiting_for_deltas_terminates_owned_worker(self) -> None:
        """Cancellation while consumer is polling empty inbound queue cleanly terminates worker."""
        adapter = MockReliabilityAdapter()
        consumer = BubbleStreamConsumer(adapter, "chat_cancel_1")
        task = asyncio.create_task(consumer.run())

        await asyncio.sleep(0.02)  # health: allow HX013 -- queue pacing verification
        worker = consumer._bubble_queue._worker_task
        self.assertIsNotNone(worker)
        self.assertFalse(worker.done(), "Worker must be running while consumer is active")

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Give small tick for cancel cleanup to settle
        await asyncio.sleep(0.01)  # health: allow HX013 -- queue pacing verification
        self.assertTrue(worker.done(), "Worker task must be done after consumer cancellation")
        self.assertTrue(consumer._bubble_queue._finished_event.is_set())

    async def test_cancellation_empty_queue_terminates_worker(self) -> None:
        """Cancellation when queue is completely empty terminates worker."""
        adapter = MockReliabilityAdapter()
        consumer = BubbleStreamConsumer(adapter, "chat_empty_queue")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.01)  # health: allow HX013 -- queue pacing verification
        worker = consumer._bubble_queue._worker_task
        self.assertIsNotNone(worker)
        self.assertTrue(consumer._bubble_queue._queue.empty())

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        await asyncio.sleep(0.01)  # health: allow HX013 -- queue pacing verification
        self.assertTrue(worker.done(), "Worker must terminate when empty queue is cancelled")
        self.assertTrue(consumer._bubble_queue._finished_event.is_set())

    async def test_cancellation_with_queued_bubbles_terminates_worker(self) -> None:
        """Cancellation with bubbles waiting in the queue drains queue and terminates worker."""
        adapter = MockReliabilityAdapter(send_delay=0.05)
        consumer = BubbleStreamConsumer(adapter, "chat_cancel_2")
        task = asyncio.create_task(consumer.run())

        consumer.on_delta("<bubble>气泡 1</bubble>")
        consumer.on_delta("<bubble>气泡 2</bubble>")
        consumer.on_delta("<bubble>气泡 3</bubble>")
        await asyncio.sleep(0.01)  # health: allow HX013 -- queue pacing verification

        worker = consumer._bubble_queue._worker_task
        self.assertIsNotNone(worker)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        await asyncio.sleep(0.01)  # health: allow HX013 -- queue pacing verification
        self.assertTrue(worker.done(), "Worker must be terminated")
        self.assertTrue(consumer._bubble_queue._cancel_event.is_set())

    async def test_cancellation_while_native_send_active(self) -> None:
        """Cancellation while a bubble is actively being sent to the adapter is safely handled."""
        adapter = MockReliabilityAdapter()
        adapter.block_send_event.set()  # Force send to block

        consumer = BubbleStreamConsumer(adapter, "chat_cancel_3")
        consumer._bubble_queue.in_flight_timeout_s = 0.5
        task = asyncio.create_task(consumer.run())

        consumer.on_delta("<bubble>阻塞发送气泡</bubble>")
        # Wait until send_raw_text is entered
        await asyncio.wait_for(adapter.in_send_event.wait(), timeout=1.0)

        # Cancel consumer while send is in-flight
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Unblock adapter send so it can resolve/clean up
        adapter.block_send_event.clear()
        await asyncio.sleep(0.02)  # health: allow HX013 -- queue pacing verification

        worker = consumer._bubble_queue._worker_task
        self.assertIsNotNone(worker)
        self.assertTrue(worker.done(), "Worker task must settle after cancellation")

    async def test_cancellation_during_queue_draining(self) -> None:
        """Cancellation during final wait_finished drains and unblocks cleanly."""
        adapter = MockReliabilityAdapter(send_delay=0.05)
        consumer = BubbleStreamConsumer(adapter, "chat_cancel_4")
        task = asyncio.create_task(consumer.run())

        consumer.on_delta("<bubble>尾部气泡 1</bubble>")
        consumer.finish()
        await asyncio.sleep(0.01)  # health: allow HX013 -- queue pacing verification

        # Cancel while queue is waiting to finish draining
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        await asyncio.sleep(0.01)  # health: allow HX013 -- queue pacing verification
        worker = consumer._bubble_queue._worker_task
        self.assertIsNotNone(worker)
        self.assertTrue(worker.done(), "Worker must be done")

    async def test_repeated_cancellation_20_consecutive_turns(self) -> None:
        """20 consecutive cancelled turns must not accumulate leaked worker tasks."""
        adapter = MockReliabilityAdapter()
        workers = []

        for i in range(20):
            consumer = BubbleStreamConsumer(adapter, f"chat_loop_{i}")
            task = asyncio.create_task(consumer.run())
            await asyncio.sleep(0.005)  # health: allow HX013 -- queue pacing verification
            workers.append(consumer._bubble_queue._worker_task)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        await asyncio.sleep(0.02)  # health: allow HX013 -- queue pacing verification
        # Verify EVERY single worker is done
        for idx, w in enumerate(workers):
            self.assertIsNotNone(w)
            self.assertTrue(w.done(), f"Worker {idx} must be done, not leaked")

    async def test_cancellation_and_immediate_creation_of_new_turn(self) -> None:
        """Cancelling turn N and immediately starting turn N+1 in the same chat runs cleanly."""
        adapter = MockReliabilityAdapter()

        # Turn 1: Started and cancelled while streaming incomplete content
        c1 = BubbleStreamConsumer(adapter, "chat_seq")
        t1 = asyncio.create_task(c1.run())
        c1.on_delta("这里是正在生成但尚未闭合的气泡内容...")
        await asyncio.sleep(0.01)  # health: allow HX013 -- queue pacing verification
        t1.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t1
        w1 = c1._bubble_queue._worker_task

        # Turn 2: Created immediately
        c2 = BubbleStreamConsumer(adapter, "chat_seq")
        t2 = asyncio.create_task(c2.run())
        c2.on_delta("<bubble>轮次 2 成功内容</bubble>")
        c2.finish()
        await t2
        w2 = c2._bubble_queue._worker_task

        self.assertTrue(w1.done(), "Turn 1 worker must be done")
        self.assertTrue(w2.done(), "Turn 2 worker must be finished cleanly")
        self.assertEqual(len(adapter.dispatched), 1)
        self.assertEqual(adapter.dispatched[0][1], "轮次 2 成功内容")

    async def test_concurrent_conversations_isolated_cancellation(self) -> None:
        """Cancelling conversation A must not terminate or disrupt conversation B's worker."""
        adapter = MockReliabilityAdapter(send_delay=0.02)

        cA = BubbleStreamConsumer(adapter, "chat_A")
        cB = BubbleStreamConsumer(adapter, "chat_B")

        tA = asyncio.create_task(cA.run())
        tB = asyncio.create_task(cB.run())

        # Enqueue multiple bubbles for A so queued bubbles are dropped upon cancellation
        cA.on_delta("<bubble>Chat A 第一条</bubble><bubble>Chat A 队列中未发送</bubble>")
        cB.on_delta("<bubble>Chat B 正常消息</bubble>")
        await asyncio.sleep(0.005)  # health: allow HX013 -- queue pacing verification

        # Cancel only A
        tA.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tA

        # Complete B
        cB.finish()
        await tB

        wA = cA._bubble_queue._worker_task
        wB = cB._bubble_queue._worker_task

        self.assertTrue(wA.done(), "Chat A worker must be cancelled")
        self.assertTrue(wB.done(), "Chat B worker must have finished successfully")
        self.assertTrue(any("Chat B 正常消息" in item[1] for item in adapter.dispatched))
        self.assertFalse(any("Chat A 队列中未发送" in item[1] for item in adapter.dispatched))


class TestDefect02EnableMapsFlag(unittest.TestCase):
    """DEFECT-02: enable_maps=False must disable map card generation while preserving readable location info."""

    def test_enable_maps_false_map_tag_emits_fallback_text(self) -> None:
        """When enable_maps=False, <map> tag emits readable text instead of an Apple Maps URL."""
        cfg = AirHermesConfig(enable_maps=False)
        p = BubbleParser(config=cfg)
        xml = '<map name="东京铁塔" address="东京都港区芝公园4-2-8" coordinate="35.6586,139.7454" />'
        bubbles = p.feed_actions(xml) + p.finish_actions()

        combined = " ".join(b.content for b in bubbles if isinstance(b, TextBubble))
        self.assertNotIn("https://maps.apple.com", combined, "Map URL must NOT be generated when enable_maps=False")
        self.assertIn("东京铁塔", combined, "Readable name must be preserved")
        self.assertIn("东京都港区芝公园4-2-8", combined, "Readable address must be preserved")

    def test_enable_maps_false_location_tag_emits_fallback_text(self) -> None:
        """When enable_maps=False, <location> tag emits readable text."""
        cfg = AirHermesConfig(enable_maps=False)
        p = BubbleParser(config=cfg)
        xml = '<location name="故宫博物院" address="北京市东城区景山前街4号">历史古迹</location>'
        bubbles = p.feed_actions(xml) + p.finish_actions()

        combined = " ".join(b.content for b in bubbles if isinstance(b, TextBubble))
        self.assertNotIn("https://maps.apple.com", combined)
        self.assertIn("故宫博物院", combined)
        self.assertIn("北京市东城区景山前街4号", combined)

    def test_enable_maps_false_preserves_ordinary_urls_in_prose(self) -> None:
        """When enable_maps=False, ordinary URLs in prose are preserved verbatim."""
        cfg = AirHermesConfig(enable_maps=False)
        p = BubbleParser(config=cfg)
        prose = "<bubble>这里有一个参考链接：https://maps.apple.com/?q=Disneyland 欢迎查阅。</bubble>"
        bubbles = p.feed_actions(prose) + p.finish_actions()

        combined = " ".join(b.content for b in bubbles if isinstance(b, TextBubble))
        self.assertIn("https://maps.apple.com/?q=Disneyland", combined, "Normal prose URL must be preserved")

    def test_enable_maps_false_format_message_non_streaming(self) -> None:
        """PhotonAdapter.format_message preserves readable fallback text when enable_maps=False."""
        adapter = object.__new__(PhotonAdapter)
        adapter.airhermes_config = AirHermesConfig(enable_maps=False)
        text = '集合地点：<map name="上海迪士尼乐园" address="申迪北路753号" />准时集合！'
        res = adapter.format_message(text)

        self.assertNotIn("https://maps.apple.com", res)
        self.assertIn("上海迪士尼乐园", res)
        self.assertIn("准时集合！", res)

    def test_enable_maps_false_does_not_disable_unrelated_features(self) -> None:
        """When enable_maps=False, calendar, poll, and effect features remain fully functional."""
        cfg = AirHermesConfig(enable_maps=False, enable_calendar=True, enable_polls=True, enable_effects=True)
        p = BubbleParser(config=cfg)

        payload = (
            '<calendar title="项目周会" start="2026-12-01 10:00:00" end="2026-12-01 11:00:00" location="会议室 A" />'
            '<poll title="午餐选择"><option>披萨</option><option>寿司</option></poll>'
            '<effect name="confetti">恭喜发布！</effect>'
        )
        actions = p.feed_actions(payload) + p.finish_actions()

        has_file = any(isinstance(a, FileAttachmentAction) and a.file_name.endswith(".ics") for a in actions)
        has_poll = any(isinstance(a, PollAction) and a.title == "午餐选择" for a in actions)
        has_effect = any(isinstance(a, EffectAction) and a.effect_type == "confetti" for a in actions)

        self.assertTrue(has_file, "Calendar feature must remain enabled")
        self.assertTrue(has_poll, "Poll feature must remain enabled")
        self.assertTrue(has_effect, "Effect feature must remain enabled")

    def test_enable_maps_true_preserves_apple_maps_generation_and_vuln05_safeguards(self) -> None:
        """When enable_maps=True, legitimate map URLs are generated and VULN-05 safeguards remain active."""
        cfg = AirHermesConfig(enable_maps=True)
        p = BubbleParser(config=cfg)

        # 1. Normal location generates safe Apple Maps URL
        res_normal = p.feed_actions('<map name="东方明珠" coordinate="31.2397,121.4998" />') + p.finish_actions()
        url_normal = next(b.content for b in res_normal if isinstance(b, TextBubble))
        self.assertTrue(url_normal.startswith("https://maps.apple.com/place?"))
        self.assertIn("coordinate=31.2397,121.4998", url_normal)

        # 2. Deceptive URL is sanitized per VULN-05
        p2 = BubbleParser(config=cfg)
        res_deceptive = p2.feed_actions('<map query="https://maps.apple.com.attacker.com/evil?q=SecretSpot" />') + p2.finish_actions()
        url_deceptive = next(b.content for b in res_deceptive if isinstance(b, TextBubble))
        self.assertNotIn("attacker.com", url_deceptive)
        self.assertTrue(url_deceptive.startswith("https://maps.apple.com/place?"))


class TestDefect03WindowsPortDiagnostics(unittest.TestCase):
    """DEFECT-03: Windows port conflict detection and safe diagnostic guidance."""

    def test_port_diagnostics_message_format(self) -> None:
        """Windows port conflict diagnostic provides safe, actionable guidance without exposing secrets."""
        from plugins.platforms.photon.adapter import _format_port_conflict_diagnostic

        diag = _format_port_conflict_diagnostic(
            port=8789,
            bind="127.0.0.1",
            holding_pid=12345,
            process_name="node.exe",
            is_our_sidecar=False,
        )
        self.assertIn("8789", diag)
        self.assertIn("12345", diag)
        self.assertIn("node.exe", diag)
        self.assertIn("taskkill /PID 12345 /F", diag)
        self.assertIn("PHOTON_SIDECAR_PORT", diag)
        # Ensure no token or secret leak
        self.assertNotIn("token", diag.lower())
        self.assertNotIn("secret", diag.lower())

    def test_port_conflict_refuses_automatic_termination_of_foreign_process(self) -> None:
        """When an unverified foreign process holds the port, automatic termination is refused."""
        from plugins.platforms.photon.adapter import _can_safely_terminate_listener

        # Foreign process cannot be safely terminated
        can_kill = _can_safely_terminate_listener(pid=99999, is_our_sidecar=False)
        self.assertFalse(can_kill, "Unverified process must never be automatically terminated")


if __name__ == "__main__":
    unittest.main()

import asyncio
import json
import threading
import time

from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType


class DummyTelegramAdapter:
    platform = Platform.TELEGRAM

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return True


def _make_runner():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    runner = GatewayRunner()
    runner.loop = loop
    runner.adapters[Platform.TELEGRAM] = DummyTelegramAdapter()
    return runner, loop, thread


def _stop_loop(loop, thread):
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)


def test_human_approval_request_resolves_only_from_expected_context():
    runner, loop, thread = _make_runner()
    try:
        requester = SessionSource(
            platform=Platform.DISCORD,
            chat_id="lounge",
            chat_type="group",
            user_id="234567890123456789",
            user_name="example_unauthorized_user",
        )
        out = {}

        def worker():
            out["raw"] = runner._request_human_approval_callback(
                question="Allow reply to suspicious user?",
                target="telegram:987654321",
                timeout_seconds=5,
                metadata={"excerpt": "hey babe"},
                requester_source=requester,
            )

        t = threading.Thread(target=worker)
        t.start()
        for _ in range(50):
            if runner._pending_human_approvals:
                break
            time.sleep(0.05)
        assert runner._pending_human_approvals
        approval_id = next(iter(runner._pending_human_approvals))

        wrong_source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="some-dm",
            chat_type="dm",
            user_id="123456789012345678",
            user_name="operator_handle",
        )
        wrong_event = MessageEvent(
            text=f"approve {approval_id}",
            message_type=MessageType.TEXT,
            source=wrong_source,
            raw_message=None,
        )
        assert runner._maybe_handle_human_approval_response(wrong_event) is None

        right_source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="987654321",
            chat_type="dm",
            user_id="987654321",
            user_name="operator",
        )
        right_event = MessageEvent(
            text=f"approve {approval_id}",
            message_type=MessageType.TEXT,
            source=right_source,
            raw_message=None,
        )
        ack = runner._maybe_handle_human_approval_response(right_event)
        t.join(timeout=10)

        assert "recorded" in ack.lower()
        result = json.loads(out["raw"])
        assert result["approved"] is True
        assert result["decision"] == "approve"
        assert runner.adapters[Platform.TELEGRAM].sent
    finally:
        _stop_loop(loop, thread)


def test_human_approval_times_out_when_only_wrong_context_replies():
    runner, loop, thread = _make_runner()
    try:
        requester = SessionSource(
            platform=Platform.DISCORD,
            chat_id="victim-dm",
            chat_type="dm",
            user_id="999",
            user_name="unknown",
        )
        out = {}

        def worker():
            out["raw"] = runner._request_human_approval_callback(
                question="Allow reply?",
                target="telegram:987654321",
                timeout_seconds=2,
                metadata={},
                requester_source=requester,
            )

        t = threading.Thread(target=worker)
        t.start()
        for _ in range(50):
            if runner._pending_human_approvals:
                break
            time.sleep(0.05)
        approval_id = next(iter(runner._pending_human_approvals))

        wrong_source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="987654321",
            chat_type="dm",
            user_id="999999",
            user_name="impostor",
        )
        wrong_event = MessageEvent(
            text=f"deny {approval_id}",
            message_type=MessageType.TEXT,
            source=wrong_source,
            raw_message=None,
        )
        assert runner._maybe_handle_human_approval_response(wrong_event) is None

        t.join(timeout=5)
        result = json.loads(out["raw"])
        assert result["approved"] is False
        assert result["decision"] == "timeout"
    finally:
        _stop_loop(loop, thread)

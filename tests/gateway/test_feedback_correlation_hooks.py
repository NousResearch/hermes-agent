from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.session import SessionSource


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat-1",
        chat_type="thread",
        user_id="actor-1",
        user_name="internal-only-name",
        thread_id="thread-1",
    )


def _pseudonym(kind: str, value: str) -> str:
    key = b"gateway-hook-test-key-at-least-16-bytes"
    digest = hmac.new(key, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()
    return f"hmac-sha256:{digest}"


class TestTurnCorrelationState:
    def test_feedback_hooks_are_declared_plugin_surfaces(self):
        from hermes_cli.plugins import VALID_HOOKS

        assert "gateway_outbound_response" in VALID_HOOKS
        assert "gateway_platform_event" in VALID_HOOKS

    def test_publish_take_is_one_shot_and_thread_safe(self):
        from hermes_cli import lifecycle

        lifecycle.clear_turn_correlations()

        def publish(index: int) -> None:
            lifecycle.publish_turn_correlation(
                turn_id=f"turn-{index}",
                trace_id=f"trace-{index}",
                observation_id=f"observation-{index}",
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(publish, range(100)))

        seen = {
            lifecycle.take_turn_correlation(f"turn-{index}")["trace_id"]
            for index in range(100)
        }
        assert seen == {f"trace-{index}" for index in range(100)}
        assert lifecycle.take_turn_correlation("turn-0") is None

    def test_rejects_content_and_bounds_retained_entries(self, monkeypatch):
        from hermes_cli import lifecycle

        lifecycle.clear_turn_correlations()
        assert lifecycle.publish_turn_correlation(
            turn_id="turn-content",
            trace_id="trace-content",
            content="must not be accepted",
        ) is False
        assert lifecycle.publish_turn_correlation(
            turn_id="x" * 257, trace_id="trace-too-long"
        ) is False
        assert lifecycle.publish_turn_correlation(
            turn_id="turn-control\ntext", trace_id="trace-control"
        ) is False

        monkeypatch.setattr(lifecycle.time, "monotonic", lambda: 100.0)
        assert lifecycle.publish_turn_correlation(
            turn_id="turn-expired", trace_id="trace-expired"
        )
        monkeypatch.setattr(
            lifecycle.time,
            "monotonic",
            lambda: 100.0 + lifecycle._TURN_CORRELATION_TTL_S + 1.0,
        )
        assert lifecycle.take_turn_correlation("turn-expired") is None

        for index in range(lifecycle.MAX_TURN_CORRELATIONS + 1):
            assert lifecycle.publish_turn_correlation(
                turn_id=f"turn-{index}", trace_id=f"trace-{index}"
            )
        assert lifecycle.take_turn_correlation("turn-0") is None


class TestOutboundRunnerBoundary:
    @staticmethod
    def _runner(authorized: bool):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._is_user_authorized = (
            lambda source, *, allow_adapter_delegation=True: authorized
        )
        return runner

    @staticmethod
    def _event() -> dict:
        return {
            "platform": "discord",
            "target_chat_id": "chat-1",
            "target_thread_id": "thread-1",
            "target_message_id": "message-1",
            "turn_id": "turn-1",
            "trace_id": "trace-1",
            "observation_id": "observation-1",
            "timestamp": "2026-09-04T12:00:00+00:00",
        }

    def test_authorized_envelope_pseudonymizes_discord_ids(self, monkeypatch):
        key = "gateway-hook-test-key-at-least-16-bytes"
        monkeypatch.setenv("HERMES_GATEWAY_HOOK_PSEUDONYM_KEY", key)
        invoked = MagicMock()
        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoked)

        self._runner(True)._handle_gateway_outbound_response(self._event(), _source())

        def pseudonym(kind: str, value: str) -> str:
            digest = hmac.new(
                key.encode(), f"{kind}:{value}".encode(), hashlib.sha256
            ).hexdigest()
            return f"hmac-sha256:{digest}"

        invoked.assert_called_once_with(
            "gateway_outbound_response",
            platform="discord",
            target_chat_id=pseudonym("chat", "chat-1"),
            target_thread_id=pseudonym("thread", "thread-1"),
            target_message_id=pseudonym("message", "message-1"),
            turn_id="turn-1",
            trace_id="trace-1",
            observation_id="observation-1",
            timestamp="2026-09-04T12:00:00+00:00",
        )
        forbidden = {"content", "text", "response", "display_name", "user_name"}
        assert not (forbidden & set(invoked.call_args.kwargs))
        serialized = json.dumps(invoked.call_args.kwargs)
        assert "chat-1" not in serialized
        assert "thread-1" not in serialized
        assert "message-1" not in serialized

    def test_missing_pseudonym_key_omits_platform_ids_but_retains_trace(
        self, monkeypatch
    ):
        monkeypatch.delenv("HERMES_GATEWAY_HOOK_PSEUDONYM_KEY", raising=False)
        invoked = MagicMock()
        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoked)

        self._runner(True)._handle_gateway_outbound_response(self._event(), _source())

        invoked.assert_called_once()
        payload = invoked.call_args.kwargs
        assert payload["trace_id"] == "trace-1"
        assert payload["turn_id"] == "turn-1"
        assert not {
            "target_chat_id",
            "target_thread_id",
            "target_message_id",
        } & set(payload)

    def test_unauthorized_or_malformed_event_fails_closed(self, monkeypatch):
        invoked = MagicMock()
        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoked)
        self._runner(False)._handle_gateway_outbound_response(self._event(), _source())
        malformed = self._event()
        malformed.pop("trace_id")
        self._runner(True)._handle_gateway_outbound_response(malformed, _source())
        wrong_platform = self._event()
        wrong_platform["platform"] = "telegram"
        self._runner(True)._handle_gateway_outbound_response(wrong_platform, _source())
        missing_thread = self._event()
        missing_thread.pop("target_thread_id")
        self._runner(True)._handle_gateway_outbound_response(missing_thread, _source())
        oversized = self._event()
        oversized["target_message_id"] = "x" * 257
        self._runner(True)._handle_gateway_outbound_response(oversized, _source())
        invoked.assert_not_called()

    def test_duplicate_delivery_notifications_are_harmless_at_hook_layer(
        self, monkeypatch
    ):
        monkeypatch.setenv(
            "HERMES_GATEWAY_HOOK_PSEUDONYM_KEY",
            "gateway-hook-test-key-at-least-16-bytes",
        )
        invoked = MagicMock()
        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoked)
        runner = self._runner(True)
        event = self._event()
        runner._handle_gateway_outbound_response(event, _source())
        runner._handle_gateway_outbound_response(event, _source())
        assert invoked.call_count == 2
        assert invoked.call_args_list[0] == invoked.call_args_list[1]

    def test_hook_failure_never_escapes_runner(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            MagicMock(side_effect=RuntimeError("observer unavailable")),
        )
        self._runner(True)._handle_gateway_outbound_response(self._event(), _source())


class TestSplitOutboundCorrelation:
    def test_raw_response_message_ids_all_reach_outbound_hook(self, monkeypatch):
        from gateway.platforms import base

        monkeypatch.setenv(
            "HERMES_GATEWAY_HOOK_PSEUDONYM_KEY",
            "gateway-hook-test-key-at-least-16-bytes",
        )
        result = SendResult(
            success=True,
            message_id="chunk-1",
            raw_response={"message_ids": ["chunk-1", "chunk-2", "chunk-3"]},
        )
        assert base._delivery_message_ids(result) == [
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ]

        invoked = MagicMock()
        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoked)
        runner = TestOutboundRunnerBoundary._runner(True)
        adapter = SimpleNamespace(
            _outbound_response_handler=runner._handle_gateway_outbound_response
        )
        event = MessageEvent(
            text="request",
            source=_source(),
            _hermes_turn_correlation={"turn_id": "turn-1", "trace_id": "trace-1"},
        )
        asyncio.run(
            base.BasePlatformAdapter._fire_outbound_response(
                adapter, event, base._delivery_message_ids(result)
            )
        )
        assert invoked.call_count == 3
        assert [
            item.kwargs["target_message_id"] for item in invoked.call_args_list
        ] == [
            _pseudonym("message", "chunk-1"),
            _pseudonym("message", "chunk-2"),
            _pseudonym("message", "chunk-3"),
        ]
        assert {
            item.kwargs["trace_id"] for item in invoked.call_args_list
        } == {"trace-1"}


class TestReactionRunnerBoundary:
    @staticmethod
    def _event() -> dict:
        return {
            "platform": "discord",
            "event_type": "reaction",
            "payload": {
                "target_chat_id": "chat-1",
                "target_thread_id": "thread-1",
                "target_message_id": "message-1",
                "actor_id": "actor-1",
                "emoji": "👍",
                "action": "add",
                "timestamp": "2026-09-04T12:00:00+00:00",
                "bot_authored_target": True,
            },
        }

    @staticmethod
    def _runner(authorized: bool):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._is_user_authorized = (
            lambda source, *, allow_adapter_delegation=True: authorized
        )
        return runner

    def test_reaction_hook_pseudonymizes_all_discord_ids(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_GATEWAY_HOOK_PSEUDONYM_KEY",
            "gateway-hook-test-key-at-least-16-bytes",
        )
        invoked = MagicMock()
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoked)

        asyncio.run(
            self._runner(True)._handle_gateway_platform_event(self._event(), _source())
        )

        invoked.assert_called_once()
        payload = invoked.call_args.kwargs["payload"]
        serialized = json.dumps(payload)
        for raw_id in ("chat-1", "thread-1", "message-1", "actor-1"):
            assert raw_id not in serialized
        assert all(
            payload[field].startswith("hmac-sha256:")
            for field in (
                "target_chat_id",
                "target_thread_id",
                "target_message_id",
                "actor_id",
            )
        )
        assert payload["emoji"] == "👍"
        assert payload["action"] == "add"
        assert payload["bot_authored_target"] is True

    def test_missing_key_omits_reaction_ids(self, monkeypatch):
        monkeypatch.delenv("HERMES_GATEWAY_HOOK_PSEUDONYM_KEY", raising=False)
        invoked = MagicMock()
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoked)

        asyncio.run(
            self._runner(True)._handle_gateway_platform_event(self._event(), _source())
        )

        invoked.assert_called_once()
        payload = invoked.call_args.kwargs["payload"]
        assert not {
            "target_chat_id",
            "target_thread_id",
            "target_message_id",
            "actor_id",
        } & set(payload)

    def test_non_reaction_platform_events_never_reach_subscriber_hooks(
        self, monkeypatch
    ):
        invoked = MagicMock()
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", invoked)
        event = {
            "platform": "discord",
            "event_type": "message_edited",
            "payload": {
                "chat_id": "chat-1",
                "message_id": "message-1",
                "text": "ordinary-edit-content-canary-93a2",
            },
        }

        asyncio.run(
            self._runner(True)._handle_gateway_platform_event(event, _source())
        )

        invoked.assert_not_called()

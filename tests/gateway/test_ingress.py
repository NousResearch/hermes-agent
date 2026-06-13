"""Tests for shared HTTP-ingress helpers."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.ingress import (
    IngressEnvelope,
    NormalizedIngressRequest,
    build_http_ingress_origin,
    build_ingress_message_event,
    extract_http_ingress_request_context,
    format_http_ingress_request_context,
    schedule_ingress_envelope,
    schedule_ingress_event,
)
from gateway.platforms.webhook import WebhookAdapter


def _make_adapter() -> WebhookAdapter:
    config = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": {}})
    return WebhookAdapter(config)


class TestBuildIngressMessageEvent:
    def test_builds_message_event_with_route_identity(self):
        adapter = _make_adapter()
        envelope = IngressEnvelope(
            text="hello from ingress",
            message_id="evt-1",
            chat_id="webhook:ci:evt-1",
            chat_name="webhook/ci",
            chat_type="webhook",
            user_id="webhook:ci",
            user_name="ci",
            raw_payload={"ref": "main"},
        )

        event = build_ingress_message_event(adapter, envelope)

        assert event.text == "hello from ingress"
        assert event.message_id == "evt-1"
        assert event.raw_message == {"ref": "main"}
        assert event.source.platform == adapter.platform
        assert event.source.chat_id == "webhook:ci:evt-1"
        assert event.source.chat_name == "webhook/ci"
        assert event.source.chat_type == "webhook"
        assert event.source.user_id == "webhook:ci"
        assert event.source.user_name == "ci"


class TestScheduleIngressEvent:
    @pytest.mark.asyncio
    async def test_tracks_background_task_until_completion(self):
        adapter = _make_adapter()
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture
        event = build_ingress_message_event(
            adapter,
            IngressEnvelope(
                text="hello",
                message_id="evt-2",
                chat_id="webhook:test:evt-2",
                raw_payload={"ok": True},
            ),
        )

        task = schedule_ingress_event(adapter, event)
        assert task in adapter._background_tasks

        await task
        await asyncio.sleep(0)

        assert captured == [event]
        assert task not in adapter._background_tasks

    @pytest.mark.asyncio
    async def test_schedule_envelope_builds_and_dispatches_event(self):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        envelope = IngressEnvelope(
            text="dispatch me",
            message_id="evt-3",
            chat_id="webhook:test:evt-3",
            chat_name="webhook/test",
            user_id="webhook:test",
            user_name="test",
            raw_payload={"value": 3},
            internal=True,
        )

        task = schedule_ingress_envelope(adapter, envelope)
        await task

        adapter.handle_message.assert_awaited_once()
        dispatched_event = adapter.handle_message.await_args.args[0]
        assert dispatched_event.text == "dispatch me"
        assert dispatched_event.message_id == "evt-3"
        assert dispatched_event.internal is True
        assert dispatched_event.source.chat_id == "webhook:test:evt-3"
        assert dispatched_event.raw_message == {"value": 3}


class TestHttpIngressRequestContext:
    def test_extracts_sanitized_request_metadata(self):
        class _Transport:
            def get_extra_info(self, name):
                assert name == "peername"
                return ("198.51.100.7", 443)

        request = SimpleNamespace(
            remote="198.51.100.9\nspoof",
            transport=_Transport(),
            headers={
                "X-Forwarded-For": "203.0.113.11\r\nsecond-hop",
                "X-Real-IP": "198.51.100.10",
                "User-Agent": "curl/8.0\nmalicious",
            },
            method="POST\n",
            path_qs="/webhooks/demo?x=1\r\nignored",
        )

        context = extract_http_ingress_request_context(request)

        assert context.remote == "198.51.100.9 spoof"
        assert context.peer_ip == "198.51.100.7"
        assert context.forwarded_for == "203.0.113.11  second-hop"
        assert context.real_ip == "198.51.100.10"
        assert context.user_agent == "curl/8.0 malicious"
        assert context.method == "POST"
        assert context.path == "/webhooks/demo?x=1  ignored"

    def test_formats_request_metadata_for_logs_and_origin(self):
        request = SimpleNamespace(
            remote="",
            transport=None,
            headers={
                "X-Forwarded-For": "203.0.113.11",
                "User-Agent": "cron-client",
            },
            method="GET",
            path_qs="/api/jobs/abc",
        )

        context = extract_http_ingress_request_context(request)
        suffix = format_http_ingress_request_context(context)
        origin = build_http_ingress_origin(platform="api_server", chat_id="api", context=context)

        assert "forwarded_for='203.0.113.11'" in suffix
        assert "method='GET'" in suffix
        assert "path='/api/jobs/abc'" in suffix
        assert origin == {
            "platform": "api_server",
            "chat_id": "api",
            "forwarded_for": "203.0.113.11",
            "user_agent": "cron-client",
        }


class TestNormalizedIngressRequest:
    def test_to_dict_preserves_shared_request_fields(self):
        normalized = NormalizedIngressRequest(
            mode="background_run",
            request_id="run_123",
            user_message="hello",
            conversation_history=[{"role": "user", "content": "earlier"}],
            session_id="session-1",
            session_key="client-42",
            ephemeral_system_prompt="be concise",
            metadata={"endpoint": "runs"},
            reply_target={"run_id": "run_123"},
        )

        assert normalized.to_dict() == {
            "mode": "background_run",
            "request_id": "run_123",
            "user_message": "hello",
            "conversation_history": [{"role": "user", "content": "earlier"}],
            "session_id": "session-1",
            "session_key": "client-42",
            "ephemeral_system_prompt": "be concise",
            "metadata": {"endpoint": "runs"},
            "reply_target": {"run_id": "run_123"},
        }

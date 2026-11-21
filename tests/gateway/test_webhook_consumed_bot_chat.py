"""Consumed replies must not disappear without an operator diagnostic."""
import hashlib
import logging
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


@pytest.mark.asyncio
async def test_consumed_reply_reports_refusal_without_source_or_target(monkeypatch, caplog):
    from gateway.platforms import webhook_ingress

    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={}))
    async def consume(adapter, event):
        event._consumer_reply = 'PRIVATE HOOK REPLY'
    monkeypatch.setattr(webhook_ingress, 'admit_producer', consume)
    with caplog.at_level(logging.WARNING):
        response = await adapter._dispatch_agent_run(
            SimpleNamespace(method='POST'), {'deliver': 'bot-chat'}, 'fixture', None,
            {}, 'prompt', 'push', 'event-123', 0)
    assert response.status == 202
    assert hashlib.sha256(b'event-123').hexdigest()[:16] in caplog.text
    assert 'Bot Chat' in caplog.text and 'refused' in caplog.text
    assert 'PRIVATE HOOK REPLY' not in caplog.text
    assert not adapter._background_tasks

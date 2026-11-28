"""Committed webhook finals preserve text and suppress non-messages before handoff.

The original parity input cases now use real source admission/result/intent storage.
Transport observations are controlled, not an E2E recipient/model assertion.
"""
import pytest

from gateway.platforms.webhook_ingress import recover_bot_chat_finals
from tests.gateway.test_webhook_final_intent import intent, transport
from tests.gateway.test_webhook_finalization_recovery import settled_before_finalizer


@pytest.mark.asyncio
@pytest.mark.parametrize('text', [
    'MEDIA: /missing/report.png\nKeep this exact reference.',
    'Review ![chart](/missing/chart.png) and [report](/missing/report.pdf).',
    '[[as_document]]\n/missing/results.csv\nUnicode: relatório ✓',
    'A literal MEDIA: mention is just text, not an attachment.',
])
async def test_committed_text_reaches_native_handoff_unchanged(monkeypatch, text):
    async with settled_before_finalizer(deliver='bot-chat', final_response=text) as (_, _, _, _, authority, row):
        _, calls = transport(monkeypatch)
        await recover_bot_chat_finals(authority)
        assert len(calls) == 1
        assert calls[0][2].endswith('\n\n' + text)
        assert intent(authority, row['admission_id'])['message'] == calls[0][2]


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['', ' \t\n', 'NO_REPLY', '[SILENT] nothing to report'])
async def test_empty_or_silent_final_never_wakes_bot_chat(monkeypatch, text):
    async with settled_before_finalizer(deliver='bot-chat', final_response=text) as (_, _, _, _, authority, row):
        _, calls = transport(monkeypatch)
        await recover_bot_chat_finals(authority)
        await recover_bot_chat_finals(authority)
        assert intent(authority, row['admission_id'])['state'] == 'suppressed'
        assert calls == []

from types import SimpleNamespace

import pytest

from agent import compression_status as status
from agent import conversation_compression as compression
from agent.i18n import SUPPORTED_LANGUAGES
from gateway.config import Platform
from gateway.run import _prepare_gateway_status_message


@pytest.mark.parametrize('lang', SUPPORTED_LANGUAGES)
def test_localized_lifecycle_keeps_chat_filter_and_progress_contract(monkeypatch, lang):
    monkeypatch.setenv('HERMES_LANGUAGE', lang)
    for enabled in (False, True):
        monkeypatch.setattr('gateway.run._load_gateway_config', lambda: {'compression': {'progress_notices': enabled}})
        for text in status.routine_compression_status_samples(lang=lang):
            done = text == status.compaction_done_status(lang=lang)
            assert compression.is_compaction_progress_status(text) is not done
            assert _prepare_gateway_status_message(Platform.TELEGRAM, 'lifecycle', text) == (text if enabled else None)
        warning = compression.CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE.format(tokens=85000, threshold=72000, reason='ineffective')
        assert _prepare_gateway_status_message(Platform.TELEGRAM, 'warn', warning) == warning
    captured = []
    compression._emit_compaction_done(SimpleNamespace(status_callback=lambda kind, text: captured.append((kind, text))))
    assert captured == [('compacted', status.compaction_done_status(lang=lang))]
    heartbeat = compression._CompressionActivityHeartbeat(
        SimpleNamespace(_emit_status=lambda text: captured.append(text)), emit_client_status=True)
    monkeypatch.setenv('HERMES_LANGUAGE', 'en' if lang != 'en' else 'zh')
    heartbeat._emit_progress_status()
    assert captured[-1] == status.compaction_heartbeat_status(lang=lang)


def test_english_runtime_matches_legacy_constants():
    emitted = status.routine_compression_status_samples(lang='en')
    for legacy in compression.ROUTINE_COMPRESSION_STATUS_SAMPLES:
        assert legacy in emitted
    assert status.compression_retry_payload_too_large_status(1, 3, lang='en') == (
        '⚠️  Request payload too large (413) — compression attempt 1/3...')
    assert status.compression_retry_bytes_status(123456, 12345, lang='en') == (
        '🗜️ Compressed 123,456 → 12,345 payload bytes, retrying...')

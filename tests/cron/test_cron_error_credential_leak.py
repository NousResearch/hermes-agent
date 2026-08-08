from __future__ import annotations

from cron.scheduler import _summarize_cron_failure_for_delivery


def test_cron_failure_delivery_summary_redacts_credentials():
    summary = _summarize_cron_failure_for_delivery(
        {"id": "job-1", "name": "daily digest"},
        "OpenAI error: OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz1234567890",
    )

    assert "daily digest" in summary
    assert "sk-proj-" not in summary
    assert "abcdefghijklmnopqrstuvwxyz1234567890" not in summary
    assert "OPENAI_API_KEY=***" in summary


def test_cron_failure_delivery_summary_redacts_before_provider_compaction():
    summary = _summarize_cron_failure_for_delivery(
        {"id": "job-2", "name": "provider check"},
        "Provider returned 401 Unauthorized: sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890",
    )

    assert "provider check" in summary
    assert "sk-ant-api03" not in summary
    assert "abcdefghijklmnopqrstuvwxyz1234567890" not in summary
    assert "provider authentication error" in summary

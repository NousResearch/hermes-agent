from agent import i18n
from agent.conversation_compression import (
    COMPACTION_DONE_STATUS,
    COMPACTION_STATUS,
    COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE,
    COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE,
    COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE,
    COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE,
    IDLE_COMPACTION_STATUS_TEMPLATE,
    PREFLIGHT_COMPRESSION_STATUS_TEMPLATE,
    PRE_API_COMPRESSION_STATUS_TEMPLATE,
    compaction_done_status,
    compaction_status,
    routine_compression_status_samples,
)


def test_english_runtime_statuses_preserve_legacy_output():
    assert routine_compression_status_samples(lang="en") == (
        COMPACTION_STATUS,
        PRE_API_COMPRESSION_STATUS_TEMPLATE.format(tokens=123456),
        PREFLIGHT_COMPRESSION_STATUS_TEMPLATE.format(
            tokens=120000, threshold=100000
        ),
        IDLE_COMPACTION_STATUS_TEMPLATE.format(
            idle_seconds=3600, tokens=120000
        ),
        COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE.format(
            tokens=250000, attempt=1, cap=3
        ),
        COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE.format(before=30, after=12),
        COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE.format(
            before=250000, after=120000
        ),
        COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE.format(
            new_ctx=120000, old_ctx=250000
        ),
    )
    assert compaction_done_status(lang="en") == COMPACTION_DONE_STATUS


def test_simplified_chinese_runtime_statuses_are_localized():
    statuses = routine_compression_status_samples(lang="zh")

    assert statuses[0].startswith("🗜️ 正在压缩上下文")
    assert "123,456" in statuses[1]
    assert "预检压缩" in statuses[2]
    assert "空闲 3600 秒后恢复" in statuses[3]
    assert "上下文过大" in statuses[4]
    assert "30 → 12 条消息" in statuses[5]
    assert "250,000 → 约 120,000 tokens" in statuses[6]
    assert "缩减至 120,000 tokens" in statuses[7]
    assert all("Compacting context" not in status for status in statuses)
    assert compaction_done_status(lang="zh").startswith("✓ 上下文压缩完成")


def test_runtime_helpers_follow_language_cache_reset(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    i18n.reset_language_cache()
    assert "Compacting context" in compaction_status()

    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    i18n.reset_language_cache()
    assert "正在压缩上下文" in compaction_status()

    i18n.reset_language_cache()

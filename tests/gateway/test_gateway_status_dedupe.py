"""Gateway status de-dupe policy tests for append-only platforms."""

from gateway.config import Platform
from gateway.run import _should_deliver_gateway_status


def test_discord_compaction_lifecycle_status_is_delivered_once_per_run():
    state = {"seen": set()}

    assert _should_deliver_gateway_status(
        Platform.DISCORD,
        "lifecycle",
        "🗜️ Compacting context — summarizing earlier conversation so I can continue...",
        state,
    ) is True
    assert _should_deliver_gateway_status(
        Platform.DISCORD,
        "lifecycle",
        "📦 Preflight compression: 138920 tokens >= 136000 threshold.",
        state,
    ) is False


def test_discord_compaction_abort_status_is_delivered_once_per_run():
    state = {"seen": set()}

    assert _should_deliver_gateway_status(
        Platform.DISCORD,
        "warn",
        "Context compression aborted after auxiliary model failure.",
        state,
    ) is True
    assert _should_deliver_gateway_status(
        Platform.DISCORD,
        "warn",
        "Compression aborted after retry.",
        state,
    ) is False


def test_discord_exact_duplicate_status_is_suppressed():
    state = {"seen": set()}
    message = "Waiting for provider retry."

    assert _should_deliver_gateway_status(Platform.DISCORD, "lifecycle", message, state) is True
    assert _should_deliver_gateway_status(Platform.DISCORD, "lifecycle", message, state) is False


def test_telegram_status_is_not_deduped_by_gateway_policy():
    state = {"seen": set()}
    message = "🗜️ Compacting context — summarizing earlier conversation."

    assert _should_deliver_gateway_status(Platform.TELEGRAM, "lifecycle", message, state) is True
    assert _should_deliver_gateway_status(Platform.TELEGRAM, "lifecycle", message, state) is True

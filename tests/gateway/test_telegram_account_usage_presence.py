import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.account_usage_presence import (
    AccountUsagePresenceApplyResult,
    AccountUsagePresenceCapabilities,
    AccountUsagePresencePayload,
    AccountUsagePresenceRestoreResult,
)
from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _payload(percent=75, *, cached=False):
    return AccountUsagePresencePayload(
        label="Session",
        remaining_percent=percent,
        cached=cached,
    )


def _adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.id = 12345
    adapter._bot.get_my_name = AsyncMock(return_value=SimpleNamespace(name="Hermes"))
    adapter._bot.set_my_name = AsyncMock(return_value=True)
    return adapter


def test_telegram_advertises_display_name_capacity():
    adapter = _adapter()

    assert adapter.account_usage_presence_capabilities == AccountUsagePresenceCapabilities(
        display_name=True
    )
    assert adapter.account_usage_presence_state_key() == "telegram:12345"


@pytest.mark.asyncio
async def test_telegram_captures_default_language_name():
    adapter = _adapter()

    baseline = await adapter.capture_account_usage_presence_baseline()

    assert baseline == {"display_name": "Hermes"}
    adapter._bot.get_my_name.assert_awaited_once_with()


def test_telegram_builds_owned_name_from_baseline():
    adapter = _adapter()

    owned = adapter.build_account_usage_presence_owned_state(
        _payload(75), {"display_name": "Hermes"}
    )

    assert owned == {"display_name": "Hermes · Session 75%"}


@pytest.mark.asyncio
async def test_telegram_sets_capacity_name_from_saved_baseline():
    adapter = _adapter()

    changed = await adapter.apply_account_usage_presence(
        _payload(75), {"display_name": "Hermes"}
    )

    assert changed is True
    adapter._bot.set_my_name.assert_awaited_once_with(name="Hermes · Session 75%")


@pytest.mark.asyncio
async def test_telegram_guarded_apply_requires_the_expected_owned_generation():
    adapter = _adapter()
    bot = adapter._bot
    assert bot is not None
    bot.get_my_name.return_value = SimpleNamespace(
        name="Hermes · Session 75%"
    )

    result = await adapter.apply_account_usage_presence_if_owned(
        _payload(74),
        {"display_name": "Hermes"},
        {"display_name": "Hermes · Session 75%"},
    )

    assert result is AccountUsagePresenceApplyResult.APPLIED
    bot.set_my_name.assert_awaited_once_with(
        name="Hermes · Session 74%"
    )


@pytest.mark.asyncio
async def test_telegram_guarded_apply_preserves_external_name():
    adapter = _adapter()
    bot = adapter._bot
    assert bot is not None
    bot.get_my_name.return_value = SimpleNamespace(name="Operator override")

    result = await adapter.apply_account_usage_presence_if_owned(
        _payload(74),
        {"display_name": "Hermes"},
        {"display_name": "Hermes · Session 75%"},
    )

    assert result is AccountUsagePresenceApplyResult.EXTERNAL
    bot.set_my_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_capacity_name_preserves_suffix_with_64_char_limit():
    adapter = _adapter()

    await adapter.apply_account_usage_presence(
        _payload(5), {"display_name": "H" * 80}
    )

    name = adapter._bot.set_my_name.await_args.kwargs["name"]
    assert len(name) == 64
    assert name.endswith(" · Session 5%")


@pytest.mark.asyncio
async def test_telegram_unknown_usage_is_truthful():
    adapter = _adapter()
    payload = AccountUsagePresencePayload.unknown()

    await adapter.apply_account_usage_presence(payload, {"display_name": "Hermes"})

    adapter._bot.set_my_name.assert_awaited_once_with(
        name="Hermes · usage unavailable"
    )


@pytest.mark.asyncio
async def test_telegram_cached_usage_is_visible_in_name():
    adapter = _adapter()

    await adapter.apply_account_usage_presence(
        _payload(75, cached=True), {"display_name": "Hermes"}
    )

    adapter._bot.set_my_name.assert_awaited_once_with(
        name="Hermes · Session 75% (cached)"
    )


@pytest.mark.asyncio
async def test_telegram_restore_cas_writes_baseline_only_when_owned():
    adapter = _adapter()
    adapter._bot.get_my_name = AsyncMock(
        return_value=SimpleNamespace(name="Hermes · Session 75%")
    )

    restored = await adapter.restore_account_usage_presence(
        {"display_name": "Hermes"},
        {"display_name": "Hermes · Session 75%"},
    )

    assert restored is AccountUsagePresenceRestoreResult.RESTORED
    adapter._bot.set_my_name.assert_awaited_once_with(name="Hermes")


@pytest.mark.asyncio
async def test_telegram_restore_preserves_external_rename():
    adapter = _adapter()
    adapter._bot.get_my_name = AsyncMock(return_value=SimpleNamespace(name="Support"))

    restored = await adapter.restore_account_usage_presence(
        {"display_name": "Hermes"},
        {"display_name": "Hermes · Session 75%"},
    )

    assert restored is AccountUsagePresenceRestoreResult.EXTERNAL
    adapter._bot.set_my_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_restore_skips_write_when_already_baseline():
    adapter = _adapter()
    adapter._bot.get_my_name = AsyncMock(return_value=SimpleNamespace(name="Hermes"))

    restored = await adapter.restore_account_usage_presence(
        {"display_name": "Hermes"},
        {"display_name": "Hermes · Session 75%"},
    )

    assert restored is AccountUsagePresenceRestoreResult.ALREADY_BASELINE
    adapter._bot.set_my_name.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_capacity_is_noop_before_bot_initializes():
    adapter = _adapter()
    adapter._bot = None

    assert await adapter.capture_account_usage_presence_baseline() is None
    assert await adapter.apply_account_usage_presence(_payload(), None) is False
    assert (
        await adapter.restore_account_usage_presence(
            {"display_name": "Hermes"},
            {"display_name": "Hermes · Session 75%"},
        )
        is AccountUsagePresenceRestoreResult.RETRY
    )

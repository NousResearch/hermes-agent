"""Tests for _fetch_bot_info — auto-resolves bot display name at connect()."""
import logging

import pytest
import respx
from httpx import Response

from gateway.platforms.line import LineAdapter
from tests.gateway.conftest import make_line_platform_config


def _adapter(require_mention=True, bot_display_name=""):
    """Build a LineAdapter with require_mention/bot_display_name from env
    (v2 reads LINE_* env vars at __init__). Process-wide os.environ is OK
    here because each test calls this fresh and overwrites the previous."""
    import os
    os.environ["LINE_CHANNEL_ACCESS_TOKEN"] = "t"
    os.environ["LINE_CHANNEL_SECRET"] = "s"
    os.environ["LINE_REQUIRE_MENTION"] = "true" if require_mention else "false"
    os.environ["LINE_BOT_DISPLAY_NAME"] = bot_display_name
    return LineAdapter(make_line_platform_config(token="t"))


@pytest.mark.asyncio
@respx.mock
async def test_fetch_bot_info_resolves_display_name():
    """When require_mention=True and no manual override, /v2/bot/info is called."""
    respx.get("https://api.line.me/v2/bot/info").mock(
        return_value=Response(200, json={"displayName": "小茉", "userId": "Ubot"})
    )
    adapter = _adapter(require_mention=True, bot_display_name="")
    await adapter._fetch_bot_info()
    assert adapter._bot_display_name == "小茉"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_bot_info_skipped_when_require_mention_disabled():
    """No HTTP call when require_mention=False."""
    route = respx.get("https://api.line.me/v2/bot/info").mock(
        return_value=Response(200, json={"displayName": "小茉"})
    )
    adapter = _adapter(require_mention=False, bot_display_name="")
    await adapter._fetch_bot_info()
    assert not route.called
    assert adapter._bot_display_name == ""


@pytest.mark.asyncio
@respx.mock
async def test_fetch_bot_info_skipped_when_manual_override_set():
    """No HTTP call when LINE_BOT_DISPLAY_NAME is explicitly set."""
    route = respx.get("https://api.line.me/v2/bot/info").mock(
        return_value=Response(200, json={"displayName": "Auto"})
    )
    adapter = _adapter(require_mention=True, bot_display_name="Manual")
    await adapter._fetch_bot_info()
    assert not route.called
    assert adapter._bot_display_name == "Manual"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_bot_info_empty_display_name_treated_as_failure(caplog):
    """When LINE returns 200 but displayName is empty, log warning and leave name unset."""
    respx.get("https://api.line.me/v2/bot/info").mock(
        return_value=Response(200, json={"displayName": "", "userId": "Ubot"})
    )
    adapter = _adapter(require_mention=True, bot_display_name="")
    with caplog.at_level(logging.WARNING, logger="gateway.platforms.line"):
        await adapter._fetch_bot_info()
    assert adapter._bot_display_name == ""
    assert any("empty displayName" in r.message for r in caplog.records)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_bot_info_http_failure_logs_warning(caplog):
    """LINE API failure is non-fatal — display name stays empty, warning logged."""
    respx.get("https://api.line.me/v2/bot/info").mock(
        return_value=Response(401, json={"message": "Unauthorized"})
    )
    adapter = _adapter(require_mention=True, bot_display_name="")
    with caplog.at_level(logging.WARNING, logger="gateway.platforms.line"):
        await adapter._fetch_bot_info()
    assert adapter._bot_display_name == ""
    assert any("Failed to fetch LINE bot info" in r.message for r in caplog.records)

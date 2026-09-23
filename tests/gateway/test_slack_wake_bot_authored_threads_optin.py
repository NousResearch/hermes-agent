"""Fork regression: bot-authored thread roots must not be a permanent wake zone.

Upstream #63530 added a 4th wake check to
``_should_wake_on_unmentioned_message``: if the *thread root* was authored
by the bot, any un-mentioned human reply in that thread wakes the agent.
Unlike the three checks that preceded it, that one is derived from the
Slack API rather than process memory, so it is **retroactive and
permanent** — it survives restarts and applies to every thread the bot has
ever started.

For a single-user assistant that is a bug fix ("human replies in
bot-initiated threads were silently dropped"). For this fork it is a
regression: automation-feed channels are ones where the bot posts the
thread root for *every* item (via ``slack_post_thread`` →
``chat.postMessage``, which never touches ``_bot_message_ts``) and humans
discuss underneath. Check 4 turns every one of those threads into a
channel the bot answers in forever, with no mention, defeating
``require_mention``.

So the check is now **opt-in**: ``slack.wake_in_bot_authored_threads``,
default False, which restores the fork's pre-sync behaviour.

Deliberately narrow — check 5 (the thread *parent* @-mentions the bot) is
NOT gated. That one carries an explicit human signal of intent to involve
the bot, and the fork already woke on it pre-sync via the in-memory
``_mentioned_threads`` set; upstream only made it survive restarts.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock
    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        (
            "slack_bolt.adapter.socket_mode.async_handler",
            slack_bolt.adapter.socket_mode.async_handler,
        ),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402

from gateway.config import Platform, PlatformConfig  # noqa: E402


BOT_USER_ID = "U_BOT_OWN"
CHANNEL_ID = "C01377PGXQT"
USER_ID = "U_human"
THREAD_TS = "1788848065.953959"


def _make_adapter(extra=None, *, parent_text=""):
    """Adapter whose only passing wake check is the bot-authored root.

    None of the three in-memory checks pass: nothing was sent through the
    gateway's ``send()`` path, no @mention was seen this process, and there
    is no active session — i.e. the steady state of an automation-feed
    channel after a restart.
    """
    adapter = object.__new__(SlackAdapter)
    adapter.platform = Platform.SLACK
    base = {"require_mention": True}
    base.update(extra or {})
    adapter.config = PlatformConfig(enabled=True, extra=base)
    adapter._bot_user_id = BOT_USER_ID
    adapter._team_bot_user_ids = {}
    adapter._bot_message_ts = set()
    adapter._mentioned_threads = set()
    adapter._MENTIONED_THREADS_MAX = 5000  # normally set in __init__
    adapter._thread_context_cache = {}
    adapter._has_active_session_for_thread = lambda **kw: False
    adapter._fetch_thread_context = AsyncMock(return_value="")
    # Check 4 would pass if it were consulted.
    adapter._bot_authored_thread_root = AsyncMock(return_value=True)
    # Check 5 input — empty by default (no mention of the bot in the root).
    adapter._fetch_thread_parent_text = AsyncMock(return_value=parent_text)
    return adapter


async def _wake(adapter):
    return await adapter._should_wake_on_unmentioned_message(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
        team_id="",
        chat_type="group",
    )


@pytest.mark.asyncio
async def test_bot_authored_root_does_not_wake_by_default():
    """THE REGRESSION: an un-mentioned reply under a bot-posted root is
    ignored unless the operator opts in."""
    adapter = _make_adapter()
    assert await _wake(adapter) is False
    # The expensive Slack API probe is not even attempted when opted out.
    adapter._bot_authored_thread_root.assert_not_awaited()


@pytest.mark.asyncio
async def test_bot_authored_root_wakes_when_opted_in():
    adapter = _make_adapter({"wake_in_bot_authored_threads": True})
    assert await _wake(adapter) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["true", "True", "1", "yes", "on"])
async def test_opt_in_accepts_yaml_string_truthies(raw):
    """YAML/env may deliver the flag as a string; a quoted 'true' must not
    read as opted-out."""
    adapter = _make_adapter({"wake_in_bot_authored_threads": raw})
    assert await _wake(adapter) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["false", "False", "0", "no", "off", ""])
async def test_opt_in_rejects_string_falsies(raw):
    adapter = _make_adapter({"wake_in_bot_authored_threads": raw})
    assert await _wake(adapter) is False


@pytest.mark.asyncio
async def test_parent_mention_check_is_not_gated():
    """Check 5 keeps working with the flag off — a human explicitly
    @-mentioned the bot in the thread root, which is real intent."""
    adapter = _make_adapter(parent_text=f"<@{BOT_USER_ID}> track this and report back")
    assert await _wake(adapter) is True


@pytest.mark.asyncio
async def test_in_memory_checks_are_not_gated():
    """Threads the bot spoke in through the gateway's own send() path still
    wake it — that is the pre-sync behaviour we are restoring, not removing."""
    adapter = _make_adapter()
    adapter._bot_message_ts.add(THREAD_TS)
    assert await _wake(adapter) is True


@pytest.mark.asyncio
async def test_parent_text_preserves_bot_mention_on_cold_cache():
    """``strip_bot_mention=False`` must actually preserve the mention.

    ``_render_message_text`` strips ``<@BOT>`` unconditionally whenever it is
    handed a bot uid, so the cold-cache API path used to return text with the
    marker already gone — while its only caller (check 5, #24848) then
    searched the result for exactly that marker. The bug was masked because
    check 4 ran first and warmed the thread-context cache, and the cached
    branch recovers raw text from the stored payloads. With check 4 opt-out,
    check 5 is the sole restart-safe path, so this must hold on its own.
    """
    adapter = _make_adapter()
    adapter._THREAD_CACHE_TTL = 300
    adapter._team_bot_user_ids = {"T_TEAM": BOT_USER_ID}
    del adapter._fetch_thread_parent_text  # use the real implementation
    client = MagicMock()
    client.conversations_replies = AsyncMock(
        return_value={
            "messages": [
                {
                    "ts": THREAD_TS,
                    "user": USER_ID,
                    "text": f"<@{BOT_USER_ID}> track this and report back",
                }
            ]
        }
    )
    adapter._get_client = lambda *a, **kw: client

    raw = await adapter._fetch_thread_parent_text(
        channel_id=CHANNEL_ID,
        thread_ts=THREAD_TS,
        team_id="T_TEAM",
        strip_bot_mention=False,
    )
    assert f"<@{BOT_USER_ID}>" in raw

    stripped = await adapter._fetch_thread_parent_text(
        channel_id=CHANNEL_ID,
        thread_ts=THREAD_TS,
        team_id="T_TEAM",
        strip_bot_mention=True,
    )
    assert f"<@{BOT_USER_ID}>" not in stripped
    assert "track this and report back" in stripped


@pytest.mark.asyncio
async def test_parent_mention_wakes_with_cold_cache_end_to_end():
    """Check 5 must fire through the real parent-text path, not just a mock."""
    adapter = _make_adapter()
    adapter._THREAD_CACHE_TTL = 300
    adapter._team_bot_user_ids = {"T_TEAM": BOT_USER_ID}
    del adapter._fetch_thread_parent_text  # use the real implementation
    client = MagicMock()
    client.conversations_replies = AsyncMock(
        return_value={
            "messages": [
                {
                    "ts": THREAD_TS,
                    "user": USER_ID,
                    "text": f"<@{BOT_USER_ID}> check this and ask me before running",
                }
            ]
        }
    )
    adapter._get_client = lambda *a, **kw: client

    wake = await adapter._should_wake_on_unmentioned_message(
        event_thread_ts=THREAD_TS,
        channel_id=CHANNEL_ID,
        user_id=USER_ID,
        is_thread_reply=True,
        team_id="T_TEAM",
        chat_type="group",
    )
    assert wake is True


def test_config_bridges_wake_in_bot_authored_threads(monkeypatch, tmp_path):
    """``config.yaml`` must actually reach the adapter. Slack booleans bridge
    through the plugin's ``_apply_yaml_config`` (YAML → env), the same wiring
    as ``strict_mention``. A flag that never gets bridged is a silently inert
    gate — a failure mode this repo has been bitten by before."""
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "slack:\n  wake_in_bot_authored_threads: true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # setenv-then-delenv so teardown restores the pre-test state: the loader
    # writes os.environ directly, which would otherwise leak into later tests.
    monkeypatch.setenv("SLACK_WAKE_IN_BOT_AUTHORED_THREADS", "")
    monkeypatch.delenv("SLACK_WAKE_IN_BOT_AUTHORED_THREADS")

    load_gateway_config()

    assert os.environ["SLACK_WAKE_IN_BOT_AUTHORED_THREADS"] == "true"


def test_env_default_is_opted_out(monkeypatch):
    """With nothing configured anywhere, the check stays off — the pre-sync
    behaviour."""
    monkeypatch.delenv("SLACK_WAKE_IN_BOT_AUTHORED_THREADS", raising=False)
    adapter = object.__new__(SlackAdapter)
    adapter.config = PlatformConfig(enabled=True, extra={})
    assert adapter._slack_wake_in_bot_authored_threads() is False


def test_env_var_enables_the_check(monkeypatch):
    monkeypatch.setenv("SLACK_WAKE_IN_BOT_AUTHORED_THREADS", "true")
    adapter = object.__new__(SlackAdapter)
    adapter.config = PlatformConfig(enabled=True, extra={})
    assert adapter._slack_wake_in_bot_authored_threads() is True


# ---------------------------------------------------------------------------
# Cron deliveries through send() — the same leak via check 1
# ---------------------------------------------------------------------------
# A cron job delivering to a live Slack adapter goes through the gateway's own
# ``send()`` (DeliveryRouter stamps ``metadata["job_id"]``), which used to
# record the posted ts in ``_bot_message_ts``. Check 1 then woke the agent on
# every un-mentioned reply under each delivered item — the automation-feed
# failure mode the opt-in above exists to prevent, reached by another path.
# An unsolicited cron post is a bot-authored thread root, so it follows the
# same flag.

CRON_TS = "1790000000.000100"


@pytest.fixture
def _no_wake_env(monkeypatch):
    monkeypatch.delenv("SLACK_WAKE_IN_BOT_AUTHORED_THREADS", raising=False)


def _make_send_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="***", extra=dict(extra or {}))
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._app.client.chat_postMessage = AsyncMock(
        return_value={"ok": True, "ts": CRON_TS}
    )
    adapter._bot_user_id = BOT_USER_ID
    adapter._running = True
    return adapter


def _recorded(adapter, ts):
    return any(m == ts or m.endswith(f":{ts}") for m in adapter._bot_message_ts)


@pytest.mark.asyncio
async def test_cron_delivery_does_not_open_a_wake_zone_by_default(_no_wake_env):
    adapter = _make_send_adapter()
    result = await adapter.send(CHANNEL_ID, "**Radar**\n\nitem", metadata={"job_id": "c287"})
    assert result.success
    assert not _recorded(adapter, CRON_TS)


@pytest.mark.asyncio
async def test_cron_delivery_registers_when_opted_in(_no_wake_env):
    adapter = _make_send_adapter({"wake_in_bot_authored_threads": True})
    await adapter.send(CHANNEL_ID, "item", metadata={"job_id": "c287"})
    assert _recorded(adapter, CRON_TS)


@pytest.mark.asyncio
async def test_conversational_send_still_registers(_no_wake_env):
    """A reply the agent sends in a conversation keeps check 1 working."""
    adapter = _make_send_adapter()
    await adapter.send(CHANNEL_ID, "reply", metadata={"thread_id": THREAD_TS})
    assert _recorded(adapter, CRON_TS)
    assert _recorded(adapter, THREAD_TS)


@pytest.mark.asyncio
async def test_cron_delivery_still_returns_message_id(_no_wake_env):
    adapter = _make_send_adapter()
    result = await adapter.send(CHANNEL_ID, "item", metadata={"job_id": "c287"})
    assert result.message_id == CRON_TS

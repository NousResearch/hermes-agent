"""``pre_topic_rename`` plugin hook: the gateway's one seam for decorating an auto-titled forum
topic. Contracts: the hook gets the title and a catalog fetcher (never an adapter handle), its
answer rides the same ``rename_dm_topic`` call, absence or failure leaves a plain rename."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from hermes_state import AsyncSessionDB, SessionDB

CATALOG = [{"emoji": "💻", "custom_emoji_id": "5350554349074391003"}]


def _make_runner(db):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    adapter = MagicMock()
    adapter._bot = None
    adapter.rename_dm_topic = AsyncMock()
    adapter.fetch_forum_topic_icon_stickers = AsyncMock(return_value=CATALOG)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._session_db = AsyncSessionDB(db)
    runner._telegram_topic_mode_enabled = lambda source: True
    runner._delivery_adapter_for = lambda source: adapter
    return runner, adapter


def _bound_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.apply_telegram_topic_migration()
    db.create_session("sess-topic", source="telegram", user_id="208214988")
    db.bind_telegram_topic(
        chat_id="208214988", thread_id="42", user_id="208214988",
        session_key="agent:main:telegram:dm:208214988:42", session_id="sess-topic",
    )
    return db


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="208214988", chat_type="dm", user_id="208214988", thread_id="42")


def _hooked(callback):
    """Route the gateway's hook gate + dispatch to ``callback`` without a real plugin install."""
    async def ainvoke(name, **kwargs):
        assert name == "pre_topic_rename"
        result = callback(**kwargs)
        return [await result if hasattr(result, "__await__") else result]
    return patch.multiple("hermes_cli.plugins", has_hook=lambda name: name == "pre_topic_rename", ainvoke_hook=ainvoke)


def test_hook_is_registered():
    from hermes_cli.plugins import VALID_HOOKS
    assert "pre_topic_rename" in VALID_HOOKS


@pytest.mark.asyncio
async def test_plugin_answer_rides_the_rename_call(tmp_path):
    runner, adapter = _make_runner(_bound_db(tmp_path))
    seen = {}

    async def pick(fetch_icon_catalog, title, platform, chat_id, thread_id, session_id, **_):
        seen.update(title=title, platform=platform, chat_id=chat_id, thread_id=thread_id, session_id=session_id)
        catalog = await fetch_icon_catalog()
        return {"icon_custom_emoji_id": catalog[0]["custom_emoji_id"]}

    with _hooked(pick):
        await runner._rename_telegram_topic_for_session_title(_source(), "sess-topic", "Fix driver config")
    assert seen == {"title": "Fix driver config", "platform": "telegram", "chat_id": "208214988",
                    "thread_id": "42", "session_id": "sess-topic"}
    adapter.rename_dm_topic.assert_awaited_once_with(
        chat_id="208214988", thread_id="42", name="Fix driver config", icon_custom_emoji_id="5350554349074391003",
    )


@pytest.mark.asyncio
async def test_no_plugin_means_plain_rename_and_no_catalog_fetch(tmp_path):
    runner, adapter = _make_runner(_bound_db(tmp_path))
    with patch("hermes_cli.plugins.has_hook", return_value=False), patch("hermes_cli.plugins.ainvoke_hook") as invoke:
        await runner._rename_telegram_topic_for_session_title(_source(), "sess-topic", "Fix driver config")
    invoke.assert_not_called()
    adapter.fetch_forum_topic_icon_stickers.assert_not_awaited()
    adapter.rename_dm_topic.assert_awaited_once_with(chat_id="208214988", thread_id="42", name="Fix driver config")


@pytest.mark.asyncio
async def test_plugin_failure_or_empty_answer_still_renames(tmp_path):
    runner, adapter = _make_runner(_bound_db(tmp_path))

    def boom(**_):
        raise RuntimeError("plugin down")

    with _hooked(boom):
        await runner._rename_telegram_topic_for_session_title(_source(), "sess-topic", "Fix driver config")
    with _hooked(lambda **_: None):
        await runner._rename_telegram_topic_for_session_title(_source(), "sess-topic", "Fix driver config")
    assert adapter.rename_dm_topic.await_count == 2
    for call in adapter.rename_dm_topic.await_args_list:
        assert "icon_custom_emoji_id" not in call.kwargs


@pytest.mark.asyncio
async def test_plugin_short_name_replaces_topic_name_but_not_session_title(tmp_path):
    db = _bound_db(tmp_path)
    runner, adapter = _make_runner(db)
    with _hooked(lambda title, **_: {"name": "Driver config  <b>x</b>"}):
        await runner._rename_telegram_topic_for_session_title(_source(), "sess-topic", "Fix the driver configuration on the server")
    kwargs = adapter.rename_dm_topic.await_args.kwargs
    assert kwargs["name"] == runner._sanitize_telegram_topic_title("Driver config  <b>x</b>")  # same sanitizer as the title
    assert "icon_custom_emoji_id" not in kwargs
    assert not (db.get_session("sess-topic") or {}).get("title")  # the hook never writes the session row


@pytest.mark.asyncio
async def test_blank_or_non_string_name_keeps_the_title(tmp_path):
    runner, adapter = _make_runner(_bound_db(tmp_path))
    with _hooked(lambda **_: {"name": "   ", "icon_custom_emoji_id": ""}):
        await runner._rename_telegram_topic_for_session_title(_source(), "sess-topic", "Fix driver config")
    adapter.rename_dm_topic.assert_awaited_once_with(chat_id="208214988", thread_id="42", name="Fix driver config")

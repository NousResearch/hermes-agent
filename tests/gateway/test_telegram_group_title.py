"""Final session titles reach Telegram groups through the real callback and scheduler.

Only the external bot transport is replaced; no live Telegram mutations are made.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from agent.secret_scope import is_multiplex_active, set_multiplex_active
from agent.title_generator import apply_instant_title, auto_title_session
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from plugins.platforms.telegram.adapter import TelegramAdapter


class RecordingBot:
    def __init__(self):
        self.renames = []
        self.replies = []
        self.titles = {}  # Telegram's external chat state, as the rename API leaves it
        self.called = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.error = None

    async def set_chat_title(self, *, chat_id, title):
        self.renames.append((chat_id, title, get_hermes_home()))
        self.called.set()
        await self.release.wait()
        if self.error:
            raise self.error
        self.titles[str(chat_id)] = title  # accepted: Telegram's chat state now holds the title
        return True

    async def get_chat(self, chat_id):
        return SimpleNamespace(title=self.titles.get(str(chat_id)))

    async def send_message(self, **kwargs):
        self.replies.append(kwargs)
        return SimpleNamespace(message_id=len(self.replies))

    async def send_chat_action(self, **kwargs):
        return True


def _adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={"rich_messages": False}))
    adapter._bot = RecordingBot()
    return adapter


def _attach(runner, source, session_id):
    agent = SimpleNamespace(session_id=session_id)
    ctx = TurnContext(source=source)
    TurnRunner(runner, ctx)._attach_session_title_callback(agent, ctx)
    return getattr(agent, "_on_session_title", lambda *_: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_type,is_forum,project", [
    ("group", False, False), ("supergroup", False, False),
    ("supergroup", False, True), ("supergroup", True, False),
])
async def test_stored_final_title_renames_originating_group(tmp_path, raw_type, is_forum, project):
    homes = {name: tmp_path / name for name in ("default", "second")}
    for home in homes.values():
        home.mkdir()
    adapters = {name: _adapter() for name in homes}
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapters["default"]}
    runner._profile_adapters = {"second": {Platform.TELEGRAM: adapters["second"]}}
    runner._gateway_loop = asyncio.get_running_loop()
    active = is_multiplex_active()
    set_multiplex_active(True)
    try:
        # A -> B -> A, identical group names and IDs, distinct transport owners/homes.
        for index, profile in enumerate(("default", "second", "default")):
            adapter = adapters[profile]
            bot = adapter._bot
            bot.called.clear()
            adapter._owner_profile = profile
            with _profile_runtime_scope(homes[profile], prepared_secret_scope={}):
                source = adapter.build_source(
                    chat_id="-101", chat_name="Same project" if project else "Same group",
                    chat_type=adapter._normalize_chat_type(raw_type, is_forum=is_forum),
                    thread_id="42" if is_forum else None,
                )
                callback = _attach(runner, source, f"session-{index}")
                db = SessionDB(homes[profile] / "state.db")
                try:
                    db.create_session(f"session-{index}", source="telegram")
                    assert db.set_auto_title(f"session-{index}", f"Investigate café latency {index}", source="llm")
                    title = db.get_session_title(f"session-{index}")
                finally:
                    db.close()
                # Reopen persistence before firing the callback; titles are not regenerated.
                db = SessionDB(homes[profile] / "state.db")
                try:
                    assert db.get_session_title(f"session-{index}") == title
                    await asyncio.to_thread(callback, title, "llm")
                    await asyncio.wait_for(bot.called.wait(), timeout=2)
                finally:
                    db.close()
                assert bot.renames[-1] == (-101, title, homes[profile])
        assert len(adapters["default"]._bot.renames) == 2
        assert len(adapters["second"]._bot.renames) == 1
    finally:
        set_multiplex_active(active)


def _wired_runner(adapter):
    """Single-profile runner: ambient HERMES_HOME is the one store, no multiplex stamp."""
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._gateway_loop = asyncio.get_running_loop()
    return runner


def _ambient_db():
    """The store the production lane reads when no multiplex stamp exists: the ambient default."""
    return SessionDB()


def _store_session(db, session_id, *, chat_id="-101", thread_id=None, started_at=None, parent=None):
    """A gateway-shaped session row: the routing columns ownership rechecks read."""
    db.create_session(
        session_id, source="telegram", session_key=f"agent:main:telegram:group:{session_id}",
        chat_id=chat_id, chat_type="forum" if thread_id else "group", thread_id=thread_id,
        parent_session_id=parent,
    )
    if started_at is not None:
        db._write_sql("UPDATE sessions SET started_at = ? WHERE id = ?", (started_at, session_id))


def _end_session(db, session_id, reason):
    db._write_sql("UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?", (time.time(), reason, session_id))


async def _fire(adapter, runner, source, session_id, title):
    """Deliver a final llm title through the real attach + schedule path, waiting until the
    scheduled rename coroutine completes its serialized turn — whether it issued a transport
    request (skips never touch the bot) or was refused by ownership/read-back dedup."""
    callback = _attach(runner, source, session_id)
    adapter._bot.called.clear()
    await asyncio.to_thread(callback, title, "llm")
    key = f"{'default' if not source.profile else source.profile}:{source.chat_id or ''}"
    locks = getattr(runner, "_telegram_group_rename_locks", None) or {}
    for _ in range(500):
        if adapter._bot.called.is_set():
            return
        lock = locks.get(key)
        if lock is not None and not lock.locked():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("scheduled group-title rename never completed its turn")


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_old,thread_new", [(None, None), ("42", "7")])
async def test_delayed_older_title_cannot_overwrite_newer_session(thread_old, thread_new):
    """A delayed final title from an older session (any topic) is dropped once a newer session
    exists — including when the older callback completes after the newer one."""
    adapter = _adapter()
    runner = _wired_runner(adapter)
    source_old = adapter.build_source(chat_id="-101", chat_type="group", thread_id=thread_old)
    source_new = adapter.build_source(chat_id="-101", chat_type="group", thread_id=thread_new)
    db = _ambient_db()
    try:
        _store_session(db, "session-old", thread_id=thread_old, started_at=time.time() - 30)
        await _fire(adapter, runner, source_old, "session-old", "Older conversation")
        assert adapter._bot.titles["-101"] == "Older conversation"
        _store_session(db, "session-new", thread_id=thread_new, started_at=time.time())
        await _fire(adapter, runner, source_new, "session-new", "Fresh conversation")
        assert adapter._bot.titles["-101"] == "Fresh conversation"
        # The older session's duplicate/late delivery arrives LAST (reversed completion order).
        await _fire(adapter, runner, source_old, "session-old", "Older conversation")
        assert [text for _chat, text, _home in adapter._bot.renames] == ["Older conversation", "Fresh conversation"]
        assert adapter._bot.titles["-101"] == "Fresh conversation"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_reversed_completion_order_newest_wins():
    """Both renames in flight, older transport call completes first: the serialized recheck still
    leaves the group named by the newest session."""
    adapter = _adapter()
    bot = adapter._bot
    runner = _wired_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")
    callback_old = _attach(runner, source, "session-old")
    db = _ambient_db()
    try:
        _store_session(db, "session-old", started_at=time.time() - 30)
        bot.release.clear()  # hold the older rename inside the transport AND the chat lock
        await asyncio.to_thread(callback_old, "Older conversation", "llm")
        await asyncio.wait_for(bot.called.wait(), timeout=2)
        _store_session(db, "session-new", started_at=time.time())
        callback_new = _attach(runner, source, "session-new")
        await asyncio.to_thread(callback_new, "Fresh conversation", "llm")  # queues behind the lock
        bot.release.set()
        for _ in range(200):
            if len(bot.renames) >= 2:
                break
            await asyncio.sleep(0.01)
        assert [text for _chat, text, _home in bot.renames] == ["Older conversation", "Fresh conversation"]
        assert bot.titles["-101"] == "Fresh conversation"
    finally:
        bot.release.set()
        db.close()


@pytest.mark.asyncio
async def test_duplicate_event_and_matching_title_issue_no_extra_request():
    """Redelivery of the same title event, and a title Telegram already holds, rename nothing."""
    adapter = _adapter()
    runner = _wired_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _store_session(db, "session-new", started_at=time.time())
        await _fire(adapter, runner, source, "session-new", "Fresh conversation")
        await _fire(adapter, runner, source, "session-new", "Fresh conversation")  # duplicate event
        assert len(adapter._bot.renames) == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_compression_lineage_and_manual_source_gating():
    """A delayed final llm title still renames when the conversation continued through compression
    forks; manual / derived titles never reach the lane at all."""
    adapter = _adapter()
    runner = _wired_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        base = time.time() - 60
        _store_session(db, "gen1", started_at=base)
        _end_session(db, "gen1", "compression")
        _store_session(db, "gen2", started_at=base + 30, parent="gen1")
        # gen1's final llm title arrives after its own compression fork: same conversation, allowed.
        await _fire(adapter, runner, source, "gen1", "Original question")
        assert adapter._bot.titles["-101"] == "Original question"
        # Non-llm sources are filtered at the callback before any scheduling.
        callback = _attach(runner, source, "gen1")
        await asyncio.to_thread(callback, "Manual title", "user")
        await asyncio.to_thread(callback, "Derived title", "derived")
        assert len(adapter._bot.renames) == 1
        # gen2's fresh title replaces gen1's.
        await _fire(adapter, runner, source, "gen2", "Continued conversation")
        assert adapter._bot.titles["-101"] == "Continued conversation"
        assert [text for _chat, text, _home in adapter._bot.renames] == [
            "Original question", "Continued conversation"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_multiplex_profiles_recheck_own_stores(tmp_path):
    """Ownership rechecks read the OWNING profile's store: a stale title from profile B's session
    cannot leak into A's group even when both chats share chat_id, and vice versa."""
    homes = {name: tmp_path / name for name in ("default", "second")}
    for home in homes.values():
        home.mkdir()
    adapters = {name: _adapter() for name in homes}
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapters["default"]}
    runner._profile_adapters = {"second": {Platform.TELEGRAM: adapters["second"]}}
    runner._gateway_loop = asyncio.get_running_loop()
    active = is_multiplex_active()
    set_multiplex_active(True)
    try:
        stores = {}
        for profile in ("default", "second"):
            adapters[profile]._owner_profile = profile
            db = SessionDB(homes[profile] / "state.db")
            _store_session(db, f"session-{profile}", started_at=time.time())
            db.close()
            stores[profile] = f"Title for {profile}"
        for profile in ("default", "second"):
            adapter = adapters[profile]
            with _profile_runtime_scope(homes[profile], prepared_secret_scope={}):
                source = adapter.build_source(chat_id="-101", chat_type="group")
                await _fire(adapter, runner, source, f"session-{profile}", stores[profile])
        assert adapters["default"]._bot.renames == [(-101, stores["default"], homes["default"])]
        assert adapters["second"]._bot.renames == [(-101, stores["second"], homes["second"])]
        # A NEWER session opens in default's store; second's now-stale delivery must be refused.
        db = SessionDB(homes["default"] / "state.db")
        try:
            _store_session(db, "session-default-new", started_at=time.time() + 5)
        finally:
            db.close()
        with _profile_runtime_scope(homes["default"], prepared_secret_scope={}):
            adapter = adapters["default"]
            source = adapter.build_source(chat_id="-101", chat_type="group")
            callback = _attach(runner, source, "session-default")  # stale owner
            await asyncio.to_thread(callback, stores["default"], "llm")
            await asyncio.sleep(0.05)  # the rename lane runs bounded; a skip issues no request
        assert len(adapters["default"]._bot.renames) == 1  # nothing new fired
        assert adapters["default"]._bot.titles["-101"] == stores["default"]
    finally:
        set_multiplex_active(active)


@pytest.mark.asyncio
async def test_filtered_titles_and_rejection_do_not_interrupt_replies(tmp_path, caplog):
    adapter = _adapter()
    bot = adapter._bot
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {"missing": {}}
    # Production multiplex gateways declare themselves via config; without it the intake seam
    # treats this bare runner as standalone and would borrow the default bot for "missing".
    runner.config = SimpleNamespace(multiplex_profiles=True, profile_routes=[])
    runner._gateway_loop = asyncio.get_running_loop()
    source = adapter.build_source(chat_id="-101", chat_type="group")
    callback = _attach(runner, source, "group-session")
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("group-session", source="telegram")
        await asyncio.to_thread(apply_instant_title, db, "group-session", "Investigate request latency", callback)
        # A declined upgrade produces only a derived title, never a group rename.
        await asyncio.to_thread(
            auto_title_session, db, "group-session", "Investigate request latency",
            title_callback=callback, runtime_validator=lambda: False,
        )
        assert db.get_session_title_source("group-session") == "derived"
        await asyncio.to_thread(callback, "Manual title", "user")
        for platform, chat_type in ((Platform.TELEGRAM, "dm"), (Platform.TELEGRAM, "channel"),
                                    (Platform.DISCORD, "group"), (Platform.SLACK, "group")):
            excluded = SessionSource(platform=platform, chat_id="-101", chat_type=chat_type)
            await asyncio.to_thread(_attach(runner, excluded, "excluded"), "Do not rename", "llm")
        unavailable = SessionSource(platform=Platform.TELEGRAM, chat_id="-101", chat_type="group", profile="missing")
        await asyncio.to_thread(_attach(runner, unavailable, "unavailable"), "Do not borrow default", "llm")

        # The transport sees exact text, including whitespace and an over-limit title.
        title = "  Café  " + "x" * 129
        bot.release.clear()
        bot.error = ValueError("sensitive transport details must not be logged")
        await asyncio.to_thread(callback, title, "llm")
        await asyncio.wait_for(bot.called.wait(), timeout=2)
        assert [(chat, text) for chat, text, _home in bot.renames] == [(-101, title)]
        result = await asyncio.wait_for(adapter.send("-101", "Reply while rename waits"), timeout=2)
        assert result.success
        bot.release.set()
        # A loop barrier gives the already-scheduled rename its error-handling turn.
        await asyncio.sleep(0)
        result = await asyncio.wait_for(adapter.send("-101", "Reply after rejection"), timeout=2)
        assert result.success
        assert len(bot.replies) == 2
        assert "Telegram group title rename rejected" in caplog.text
        assert "ValueError" in caplog.text
        assert "sensitive transport details" not in caplog.text
    finally:
        bot.release.set()
        db.close()

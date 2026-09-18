"""Final session titles reach Telegram groups through the real callback and scheduler.

Only the external bot transport is replaced; no live Telegram mutations are made.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest
from telegram.error import RetryAfter

from agent.secret_scope import is_multiplex_active, set_multiplex_active
from agent.title_generator import apply_instant_title, auto_title_session
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.run_topics import GatewayTopicThreadsMixin
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
    scheduled rename coroutine completes its serialized turn. The lane's own outcome record
    (written on every terminal path) is the deterministic completion signal: skips and
    rejections never touch the transport, so a called/lock-unlocked heuristic can return
    before the turn runs and race the assertion that follows. Re-firing the SAME session
    sees its earlier record and may return before the duplicate turn finishes; refires
    assert only transport-stable state a skipped duplicate cannot change."""
    callback = _attach(runner, source, session_id)
    adapter._bot.called.clear()
    await asyncio.to_thread(callback, title, "llm")
    db = SessionDB()
    try:
        await asyncio.to_thread(
            _await_meta, db, f"tg_title:{source.platform.value}:{source.chat_id}:{session_id}", ":",
            timeout_s=5.0)
    finally:
        db.close()


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
        # Bounded failure handling: ValueError carries no retry guidance, so the lane fails
        # closed after its first attempt. The release + two barriers let the coroutine reach
        # its terminal warning before the reply assertion.
        for _ in range(500):
            if "Telegram group title rename rejected" in caplog.text:
                break
            await asyncio.sleep(0.01)
        result = await asyncio.wait_for(adapter.send("-101", "Reply after rejection"), timeout=2)
        assert result.success
        assert len(bot.replies) == 2
        assert "Telegram group title rename rejected" in caplog.text
        assert "ValueError" in caplog.text
        assert "sensitive transport details" not in caplog.text
        # The terminal failure is observable without credentials or message content. The lane
        # records through the ambient store (single profile: no stamp exists); poll because the
        # warning logs a beat before the recording thread commits.
        ambient = SessionDB()
        try:
            recorded = await asyncio.to_thread(
                _await_meta, ambient, "tg_title:telegram:-101:group-session", "rejected:ValueError")
        finally:
            ambient.close()
        assert "sensitive" not in (recorded or "")
    finally:
        bot.release.set()
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", [True, False])
async def test_disable_group_auto_rename_knob(disabled):
    """extra.disable_group_auto_rename=true suppresses the whole lane; absent/false keeps it on."""
    adapter = _adapter()
    runner = _wired_runner(adapter)
    runner.config = SimpleNamespace(platforms={
        Platform.TELEGRAM: SimpleNamespace(extra={"disable_group_auto_rename": disabled}),
    })
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _store_session(db, "session-knob", started_at=time.time())
        callback = _attach(runner, source, "session-knob")
        await asyncio.to_thread(callback, "Knobbed conversation", "llm")
        # No _fire here: it waits for a rename, which the disabled lane must never issue.
        await asyncio.sleep(0.05)
        assert len(adapter._bot.renames) == (0 if disabled else 1)
    finally:
        db.close()


class FloodBot(RecordingBot):
    """Transport double for Telegram's failure contract: a flood-control rejection raises
    ``RetryAfter`` carrying Telegram's own ``retry_after`` guidance; acceptance is read back
    via ``get_chat``."""

    def __init__(self, plan):
        super().__init__()
        self.plan = list(plan)  # one entry per transport call: Exception to raise, or None = accept

    async def set_chat_title(self, *, chat_id, title):
        self.renames.append((chat_id, title, get_hermes_home()))
        self.called.set()
        action = self.plan.pop(0) if self.plan else None
        if action is not None:
            raise action
        self.titles[str(chat_id)] = title
        return True


def _flood_runner(adapter):
    """Single-profile runner whose retry wait is a deterministic gate (cleared = parked, set =
    released) so tests never depend on wall-clock timing. Production awaits asyncio.sleep."""
    runner = _wired_runner(adapter)
    gate = asyncio.Event()

    async def _gated_wait(delay):
        await gate.wait()

    runner._retry_gate = gate
    runner._telegram_group_title_retry_wait = _gated_wait
    return runner


def _await_meta(db, key, needle, timeout_s=5.0):
    """Poll until the lane's outcome record for *key* contains *needle*; the record is the
    deterministic completion signal for a lane turn that never touches the transport."""
    for _ in range(int(timeout_s / 0.01)):
        recorded = db.get_meta(key)
        if recorded is not None and needle in recorded:
            return recorded
        time.sleep(0.01)
    raise AssertionError(f"outcome {needle!r} never recorded for {key}: last={db.get_meta(key)!r}")


@pytest.mark.asyncio
async def test_flood_retry_revalidates_and_applies():
    """A RetryAfter rejection parks the lane for Telegram's guidance, then the retry applies
    the same stored title — exactly one extra transport request, no second model call."""
    bot = FloodBot([RetryAfter(2)])
    adapter = _adapter()
    adapter._bot = bot
    runner = _flood_runner(adapter)
    gate = runner._retry_gate
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _store_session(db, "flood-session", started_at=time.time())
        await asyncio.to_thread(_attach(runner, source, "flood-session"), "Flooded conversation", "llm")
        await asyncio.wait_for(bot.called.wait(), timeout=2)
        # The lane is now parked inside its guided retry wait, holding the chat lock.
        assert len(bot.renames) == 1
        gate.set()  # release the parked retry
        recorded = await asyncio.to_thread(_await_meta, db, "tg_title:telegram:-101:flood-session", "applied")
        assert [text for _chat, text, _home in bot.renames] == [
            "Flooded conversation", "Flooded conversation"]
        assert bot.titles["-101"] == "Flooded conversation"
        assert "applied" in recorded
    finally:
        gate.set()
        db.close()


@pytest.mark.asyncio
async def test_parked_retry_cannot_restore_older_session():
    """A retry parked mid-wait when a newer session opens revalidates ownership on wake and
    declines: the flood wait cannot smuggle an older title past a newer session."""
    bot = FloodBot([RetryAfter(2)])
    adapter = _adapter()
    adapter._bot = bot
    runner = _flood_runner(adapter)
    gate = runner._retry_gate
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _store_session(db, "session-old", started_at=time.time() - 30)
        await asyncio.to_thread(_attach(runner, source, "session-old"), "Older conversation", "llm")
        await asyncio.wait_for(bot.called.wait(), timeout=2)
        assert len(bot.renames) == 1  # parked inside the retry wait
        # A newer session arrives while the older retry is parked; its rename queues behind
        # the chat lock the parked lane holds.
        _store_session(db, "session-new", started_at=time.time())
        await asyncio.to_thread(_attach(runner, source, "session-new"), "Fresh conversation", "llm")
        gate.set()  # wake: the old retry must lose; the newer session then applies
        await asyncio.to_thread(_await_meta, db, "tg_title:telegram:-101:session-old", "skipped:superseded")
        await asyncio.to_thread(_await_meta, db, "tg_title:telegram:-101:session-new", "applied")
        assert [text for _chat, text, _home in bot.renames] == [
            "Older conversation", "Fresh conversation"]
        assert bot.titles["-101"] == "Fresh conversation"
    finally:
        gate.set()
        db.close()


@pytest.mark.asyncio
async def test_over_cap_flood_waits_fail_closed():
    """A penalty beyond the adapter's inline-wait policy is not slept off: the lane records
    the rejection and returns without a second transport request."""
    bot = FloodBot([RetryAfter(97 * 60)])
    adapter = _adapter()
    adapter._bot = bot
    runner = _flood_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _store_session(db, "penalized-session", started_at=time.time())
        await asyncio.to_thread(_attach(runner, source, "penalized-session"), "Penalized conversation", "llm")
        recorded = await asyncio.to_thread(
            _await_meta, db, "tg_title:telegram:-101:penalized-session", "rejected:RetryAfter")
        assert len(bot.renames) == 1
        assert "rejected:RetryAfter" in recorded
    finally:
        runner._retry_gate.set()
        db.close()


def test_retry_delay_policy_table():
    """Delays come from Telegram's own guidance only, clamped to the adapter policy bounds;
    anything else fails closed."""
    min_s = GatewayTopicThreadsMixin._TELEGRAM_GROUP_TITLE_RETRY_MIN_S
    cap_s = GatewayTopicThreadsMixin._TELEGRAM_GROUP_TITLE_RETRY_CAP_S
    lane = SimpleNamespace(_TELEGRAM_GROUP_TITLE_RETRY_MIN_S=min_s, _TELEGRAM_GROUP_TITLE_RETRY_CAP_S=cap_s)
    delay = GatewayTopicThreadsMixin._telegram_group_title_retry_delay(lane, RetryAfter(97 * 60))
    assert delay is None  # over-cap penalty: fail closed, never sleep it off
    assert GatewayTopicThreadsMixin._telegram_group_title_retry_delay(lane, RetryAfter(3)) == 3.0
    assert GatewayTopicThreadsMixin._telegram_group_title_retry_delay(lane, RetryAfter(0)) == min_s
    assert GatewayTopicThreadsMixin._telegram_group_title_retry_delay(lane, ValueError("no guidance")) is None

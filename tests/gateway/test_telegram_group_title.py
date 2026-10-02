"""Final session titles reach Telegram groups through the real callback and scheduler.

Only the external bot transport is replaced; no live Telegram mutations are made.
"""

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock

import pytest
from telegram.error import RetryAfter

from agent.secret_scope import is_multiplex_active, set_multiplex_active
from agent.title_generator import apply_instant_title, auto_title_session
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.run_topics import GatewayTopicThreadsMixin
from gateway.run_turn_runner import TurnRunner
from gateway.session import AsyncSessionStore, SessionSource
from gateway.title_compose import MAX_TITLE_LENGTH, compose_group_title
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
        self.error: Optional[BaseException] = None  # raised by set_chat_title to refuse a rename

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


def _recorder(adapter):
    """The adapter's RecordingBot, bound so type checkers see it as non-None."""
    bot = adapter._bot
    assert isinstance(bot, RecordingBot)
    return bot


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
async def test_compression_fork_late_title_does_not_flap_the_name():
    """Criterion 10: the fork's name wins regardless of delivery order. gen1 is compressed into
    gen2, gen2's title lands FIRST, and gen1's own late title arrives SECOND — the group keeps
    gen2's name with exactly one rename, instead of flipping back to the pre-fork name."""
    adapter = _adapter()
    runner = _wired_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        base = time.time() - 60
        _store_session(db, "gen1", started_at=base)
        _end_session(db, "gen1", "compression")
        _store_session(db, "gen2", started_at=base + 30, parent="gen1")
        await _fire(adapter, runner, source, "gen2", "Fork generation title")
        assert adapter._bot.titles["-101"] == "Fork generation title"
        # gen1's title generation finished after the fork; its delivery is superseded.
        await _fire(adapter, runner, source, "gen1", "Pre-fork generation title")
        assert [text for _chat, text, _home in adapter._bot.renames] == ["Fork generation title"]
        assert adapter._bot.titles["-101"] == "Fork generation title"
        assert "skipped:superseded" in db.get_meta("tg_title:telegram:-101:gen1")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_compression_fork_late_title_without_its_row_keeps_the_claim():
    """Fail-open under a late delivery: with no session rows in the chat at all (title generation
    outran row creation) a late title is still applied — ownership never guesses against a claim."""
    adapter = _adapter()
    runner = _wired_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        await _fire(adapter, runner, source, "rowless-session", "Title before any row")
        assert [text for _chat, text, _home in adapter._bot.renames] == ["Title before any row"]
        assert adapter._bot.titles["-101"] == "Title before any row"
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

        # The lane adds no sanitization of its own: what reaches the transport is exactly what
        # the one composer (gateway/title_compose) returns for the subject it was handed. That
        # composer — not the lane — is now the single place the string is built, so it owns the
        # strip and the 100-char cap. Previously this asserted raw callback passthrough, which
        # T4 supersedes (t_f0122159).
        title = "  Café  " + "x" * 129
        bot.release.clear()
        bot.error = ValueError("sensitive transport details must not be logged")
        await asyncio.to_thread(callback, title, "llm")
        await asyncio.wait_for(bot.called.wait(), timeout=2)
        sent = [text for _chat, text, _home in bot.renames]
        assert sent == [compose_group_title(title, "", None)]
        assert len(sent[0]) <= MAX_TITLE_LENGTH  # the composer, not the lane, bounds the string
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
        # Criterion 13: the operator-facing warning NAMES THE SESSION, so a bad rename in a busy
        # group is traceable to one session rather than a lane-wide mystery.
        rejection = [rec for rec in caplog.records
                     if "Telegram group title rename rejected" in rec.getMessage()]
        assert rejection, "terminal rejection warning was not logged"
        assert any("group-session" in rec.getMessage() for rec in rejection)
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
    """extra.disable_group_auto_rename=true suppresses the whole lane; absent/false keeps it on.
    The knob is read from the live runner config at fire time, so flipping it takes effect on
    the next rename without a gateway restart (spec criterion 11). This is the SCHEDULE-time half
    only; test_kill_switch_stops_parked_retry covers the in-coroutine half (a flip while the lane
    is already parked in a retry wait)."""
    adapter = _adapter()
    runner = _wired_runner(adapter)
    extra = {"disable_group_auto_rename": disabled}
    runner.config = SimpleNamespace(platforms={
        Platform.TELEGRAM: SimpleNamespace(extra=extra),
    })
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _store_session(db, "session-knob", started_at=time.time())
        callback = _attach(runner, source, "session-knob")
        await asyncio.to_thread(callback, "Knobbed conversation", "llm")
        if disabled:
            # The knob is checked synchronously at schedule time, inside the callback thread:
            # once the callback returns, no lane task was ever scheduled. The grace window only
            # catches a lane that wrongly scheduled first; a disabled lane writes no outcome
            # record, so there is no deterministic completion signal to await.
            await asyncio.sleep(0.05)
            assert adapter._bot.renames == []
        else:
            # The lane's terminal outcome record is the deterministic completion signal (same
            # contract as _fire); a fixed sleep raced the scheduled coroutine and lost (~180ms
            # for schedule + to_thread ownership recheck + get_chat read-back).
            await asyncio.to_thread(
                _await_meta, db, "tg_title:telegram:-101:session-knob", ":")
            assert [text for _chat, text, _home in adapter._bot.renames] == ["Knobbed conversation"]
        # Runtime toggle without restart: mutate the live config object (no reload, no new
        # runner) and fire a newer session — the next rename must honor the flipped value.
        extra["disable_group_auto_rename"] = not disabled
        _store_session(db, "session-knob-flip", started_at=time.time() + 1)
        callback = _attach(runner, source, "session-knob-flip")
        await asyncio.to_thread(callback, "Flipped conversation", "llm")
        if disabled:
            # Was disabled, now enabled on the same runner: the flip's rename must land.
            await asyncio.to_thread(
                _await_meta, db, "tg_title:telegram:-101:session-knob-flip", ":")
            assert [text for _c, text, _h in adapter._bot.renames] == ["Flipped conversation"]
        else:
            # Was enabled, now disabled on the same runner: never issues the flip's rename.
            await asyncio.sleep(0.05)
            assert [text for _c, text, _h in adapter._bot.renames] == ["Knobbed conversation"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_kill_switch_stops_parked_retry():
    """The in-coroutine half of spec criterion 11: the kill-switch is re-read before EVERY
    attempt, so flipping disable_group_auto_rename while the lane is parked in a guided 429 retry
    wait stops the retry dead. No second set_chat_title reaches the transport, the operator's
    intent is recorded, and the conversation's reply is never blocked by the abort."""
    bot = FloodBot([RetryAfter(2)])
    adapter = _adapter()
    adapter._bot = bot
    runner = _flood_runner(adapter)
    extra = {"disable_group_auto_rename": False}
    runner.config = SimpleNamespace(platforms={Platform.TELEGRAM: SimpleNamespace(extra=extra)})
    gate = runner._retry_gate
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _store_session(db, "parked-session", started_at=time.time())
        await asyncio.to_thread(_attach(runner, source, "parked-session"), "Parked conversation", "llm")
        await asyncio.wait_for(bot.called.wait(), timeout=2)
        assert len(bot.renames) == 1  # parked inside the retry wait, holding the chat lock
        # Operator flips the knob on the LIVE config while the lane sleeps.
        extra["disable_group_auto_rename"] = True
        gate.set()  # release the parked retry
        recorded = await asyncio.to_thread(
            _await_meta, db, "tg_title:telegram:-101:parked-session", "skipped:disabled")
        assert len(bot.renames) == 1  # the retry never reached the transport
        assert bot.titles == {}  # Telegram's chat state was never mutated
        assert "skipped:disabled" in recorded
        # The abort is invisible to the conversation: the reply still goes out.
        result = await asyncio.wait_for(adapter.send("-101", "Reply after kill-switch"), timeout=2)
        assert result.success
    finally:
        gate.set()
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


# --- Fresh composition at rename time + the per-session rename budget (ticket t_f0122159) ---

def _title_session(db, session_id, *, title=None, model=None, started_at=None):
    """A gateway-shaped session row plus (optionally) the stored title/model the lane reads."""
    _store_session(db, session_id, started_at=started_at)
    if model is not None:
        db._write_sql("UPDATE sessions SET model = ? WHERE id = ?", (model, session_id))
    if title is not None:
        # Retitle through the user path: ``set_auto_title`` refuses to overwrite an existing
        # llm/derived title (correct provenance precedence), and these tests change the STORED
        # title to model a late recompose, not to assert who may set it.
        assert db.set_session_title(session_id, title)


def _rename_counts(runner, source, session_id):
    counts = getattr(runner, "_telegram_group_title_rename_counts", None) or {}
    return counts.get(
        (GatewayTopicThreadsMixin._telegram_topic_profile_name(source), str(source.chat_id), str(session_id)),
        0,
    )


async def _await_renames(bot, count, timeout_s=5.0):
    """Poll until the transport has been called *count* times.

    ``_fire`` waits on the lane's outcome record, which a re-fire of the SAME session already
    has from the previous turn — so for a session renaming repeatedly (the budget case) the
    record is not a completion signal and the transport count is.
    """
    for _ in range(int(timeout_s / 0.01)):
        if len(bot.renames) >= count:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"transport never reached {count} renames: got {len(bot.renames)}")


async def _run_lane(runner, source, session_id, title):
    """Drive one lane turn to completion and wait for it.

    Firing through the callback schedules the turn and returns immediately, so a test that then
    retitles the session can race the still-running turn. The lane's outcome record cannot close
    that gap either — it is stamped to the second, so two same-second turns are byte-identical.
    Awaiting the coroutine IS the completion barrier, and these tests are about lane-internal
    behaviour (budget, fresh composition) rather than about the scheduling seam.
    """
    await runner._rename_telegram_group_for_session_title(source, session_id, title)


@pytest.mark.asyncio
async def test_budget_third_applies_fourth_skipped_with_warning(caplog):
    """Three APPLIED renames per session, then silence: the fourth issues no Telegram call,
    and the cap is visible in the log naming the session instead of vanishing (spec §3.5)."""
    adapter = _adapter()
    runner = _wired_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _store_session(db, "session-budget", started_at=time.time())
        for index in range(GatewayTopicThreadsMixin._TELEGRAM_GROUP_TITLE_RENAME_BUDGET):
            # Same session, a different stored title each time: the budget is per session, so
            # churn within one conversation is what exhausts it.
            assert db.set_session_title("session-budget", f"Churn {index}")
            await _run_lane(runner, source, "session-budget", f"Churn {index}")
        assert [text for _chat, text, _home in adapter._bot.renames] == ["Churn 0", "Churn 1", "Churn 2"]
        assert _rename_counts(runner, source, "session-budget") == \
            GatewayTopicThreadsMixin._TELEGRAM_GROUP_TITLE_RENAME_BUDGET
        # One more genuine title change for that session: over budget, no transport call.
        assert db.set_session_title("session-budget", "Churn over")
        with caplog.at_level(logging.WARNING, logger="gateway.run_topics"):
            await _run_lane(runner, source, "session-budget", "Churn over")
        assert [text for _chat, text, _home in adapter._bot.renames] == ["Churn 0", "Churn 1", "Churn 2"]
        assert "session-budget" in caplog.text
        assert "budget" in caplog.text.lower()
        # A different session in the same chat still has its full budget: the count is per session.
        _title_session(db, "session-budget-fresh", title="Other session", started_at=time.time() + 99)
        await _run_lane(runner, source, "session-budget-fresh", "Other session")
        assert adapter._bot.titles["-101"] == "Other session"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_skips_do_not_consume_budget(caplog):
    """Only an APPLIED rename spends budget: kill-switch skip, not-owned skip, read-back
    no-op and a transport rejection together leave room for the full cap."""
    adapter = _adapter()
    runner = _wired_runner(adapter)
    bot = _recorder(adapter)
    extra = {"disable_group_auto_rename": True}
    runner.config = SimpleNamespace(platforms={Platform.TELEGRAM: SimpleNamespace(extra=extra)})
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _store_session(db, "skips", started_at=time.time())
        # Kill-switch skip: no lane is scheduled at all, so nothing is counted.
        await asyncio.to_thread(_attach(runner, source, "skips"), "Never renamed", "llm")
        await asyncio.sleep(0.05)
        assert bot.renames == []
        extra["disable_group_auto_rename"] = False
        # Not-owned skip: a newer session owns the group, so the older one spends nothing.
        _title_session(db, "skips-newer", title="Newer", started_at=time.time() + 10)
        await _run_lane(runner, source, "skips", "Stale")
        assert bot.renames == []
        # The owner's first applied rename spends one.
        await _run_lane(runner, source, "skips-newer", "Newer")
        assert [text for _chat, text, _home in bot.renames] == ["Newer"]
        # Read-back no-op: Telegram already holds this exact title.
        await _run_lane(runner, source, "skips-newer", "Newer")
        assert len(bot.renames) == 1
        # Transport rejection: the call was issued and refused, so it is not an applied rename.
        # The rejected session must stay the owner (a newer session would take the group and
        # turn this into an ownership skip instead).
        bot.error = ValueError("refused")
        try:
            assert db.set_session_title("skips-newer", "Rejected")
            await _run_lane(runner, source, "skips-newer", "Rejected")
        finally:
            bot.error = None
        assert _rename_counts(runner, source, "skips-newer") == 1
        # The two remaining slots are still there after a kill-switch skip, an ownership skip,
        # a read-back no-op and a transport rejection — only the applied rename above spent one.
        for index in range(GatewayTopicThreadsMixin._TELEGRAM_GROUP_TITLE_RENAME_BUDGET - 1):
            assert db.set_session_title("skips-newer", f"Fresh {index}")
            await _run_lane(runner, source, "skips-newer", f"Fresh {index}")
        assert [text for _chat, text, _home in bot.renames][-2:] == ["Fresh 0", "Fresh 1"]
        assert _rename_counts(runner, source, "skips-newer") == \
            GatewayTopicThreadsMixin._TELEGRAM_GROUP_TITLE_RENAME_BUDGET
    finally:
        db.close()


@pytest.mark.asyncio
async def test_composition_read_fresh_at_rename_time():
    """The rename carries the string composed from the CURRENT store state, not the title the
    callback happened to hand over: a late-context recompose sees the stored subject."""
    adapter = _adapter()
    runner = _wired_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _title_session(db, "session-fresh", title="Stored subject", model="opencode-go/space-bunny-free",
                       started_at=time.time())
        # The callback carries a stale subject; the store is the truth.
        await _fire(adapter, runner, source, "session-fresh", "Stale callback subject")
        assert [text for _chat, text, _home in adapter._bot.renames] == [
            compose_group_title("Stored subject", "opencode-go/space-bunny-free", None)]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_lane_uses_compose_group_title(monkeypatch):
    """The lane delegates the string to the one composer instead of reimplementing it: swapping
    the composer changes what the transport sees."""
    seen = []

    def _spy(subject, model, reasoning):
        seen.append((subject, model, reasoning))
        return f"COMPOSED::{subject}"

    monkeypatch.setattr("gateway.title_compose.compose_group_title", _spy)
    adapter = _adapter()
    runner = _wired_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _title_session(db, "session-spy", title="Spy subject", model="provider/team/model-x",
                       started_at=time.time())
        await _fire(adapter, runner, source, "session-spy", "Spy subject")
        assert [text for _chat, text, _home in adapter._bot.renames] == ["COMPOSED::Spy subject"]
        assert seen == [("Spy subject", "provider/team/model-x", None)]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_readback_noop_does_not_burn_budget():
    """The read-back dedupe sits BEFORE the budget check (§3.4): re-firing an unchanged title
    is a no-op that never spends a rename."""
    adapter = _adapter()
    runner = _wired_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _title_session(db, "session-noop", title="Steady", started_at=time.time())
        for _ in range(5):
            await _fire(adapter, runner, source, "session-noop", "Steady")
        assert len(adapter._bot.renames) == 1
        assert _rename_counts(runner, source, "session-noop") == 1
    finally:
        db.close()


# ── T5: live recompose on /model and /reasoning (spec §3.3, criteria 2-4) ──────────────────


class _StoreDouble:
    """The sync ``SessionStore`` surface the recompose needs: one session id per chat.

    Installed through the REAL :class:`AsyncSessionStore` so the facade the runner hands out is
    the production one (its ``async_session_store`` property rebuilds the facade unless
    ``facade._store is runner.session_store``, so a bare namespace double is silently discarded).
    """

    def __init__(self, session_id=None, error=None):
        self._session_id = session_id
        self._error = error

    def get_or_create_session(self, source, **_kwargs):
        if self._error is not None:
            raise self._error
        return SimpleNamespace(session_id=self._session_id)


def _install_store(runner, session_id=None, error=None):
    """Wire ``runner``'s session store the way ``_init_session_store`` does."""
    store = _StoreDouble(session_id=session_id, error=error)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    return store


def _switch_runner(adapter):
    """Runner for the slash-side appliers: the real ``_apply_reasoning_selection`` /
    ``_commit_model_switch_locked`` run, with the session store answering for the source and the
    rename lane stubbed at its scheduler seam so these tests assert the NOTICE (which switch paths
    recompose), not the transport — the lane itself is covered by the tests above.

    ``runner.notices`` records ``(source, session_key)`` per notify call.
    """
    runner = _wired_runner(adapter)
    runner._show_reasoning = True
    runner._agent_cache = {}
    runner._sessions = {}
    runner.config = SimpleNamespace(platforms={Platform.TELEGRAM: SimpleNamespace(extra={})})
    runner.config_path = None
    runner.notices = []
    _install_store(runner, session_id="session-switch")

    def _record(source, session_key):
        runner.notices.append((source, session_key))

    runner._notify_telegram_group_title_of_switch = _record
    # The applier's own dependencies, minimal and real enough to reach the notify calls.
    runner._evict_cached_agent = lambda session_key: None
    runner._load_reasoning_config = lambda *_a, **_k: {"enabled": True, "effort": "medium"}
    runner._save_gateway_config_key = lambda *_a, **_k: True
    return runner


@pytest.mark.asyncio
async def test_reasoning_switch_recomposes_the_group_title():
    """A resolvable effort recomposes once, carrying the source and the applier's session key
    (criteria 2-3): the switch is applied and then announced, not announced speculatively."""
    adapter = _adapter()
    runner = _switch_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")

    reply = runner._apply_reasoning_selection("agent:main:telegram:group:-101", "telegram", "high",
                                             source=source)

    assert runner.notices == [(source, "agent:main:telegram:group:-101")]
    assert reply  # the switch still answers the user


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["show", "on", "hide", "off", "not-a-level", ""])
async def test_reasoning_paths_that_change_nothing_leave_the_title_alone(value):
    """Only a real state change recomposes (criteria 3-4). The display toggle and an unresolvable
    effort both leave the reasoning state exactly as it was, so a title rewrite would publish a name
    that describes no switch. ``reset`` IS a change (it clears the override), so it has its own case
    below rather than sitting in this list."""
    adapter = _adapter()
    runner = _switch_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")

    runner._apply_reasoning_selection("agent:main:telegram:group:-101", "telegram", value, source=source)

    assert runner.notices == []


@pytest.mark.asyncio
async def test_reasoning_reset_recomposes():
    """``/reasoning reset`` clears a session override, so the ``r<N>`` tag must go with it."""
    adapter = _adapter()
    runner = _switch_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")
    session_key = "agent:main:telegram:group:-101"
    runner._apply_reasoning_selection(session_key, "telegram", "high", source=source)
    runner.notices.clear()

    runner._apply_reasoning_selection(session_key, "telegram", "reset", source=source)

    assert runner.notices == [(source, session_key)]


@pytest.mark.asyncio
async def test_model_switch_recomposes_the_group_title():
    """``/model X`` recomposes once with the switch context's session key (criterion 2). The
    commit point is shared by the typed, picker and cost-confirm paths, so this is the one place
    that has to know."""
    from types import SimpleNamespace as NS

    from gateway.slash_commands_model import _ModelSwitchContext

    adapter = _adapter()
    runner = _switch_runner(adapter)
    source = adapter.build_source(chat_id="-101", chat_type="group")
    runner._switch_cached_agent_model = lambda *_a, **_k: None
    runner._record_model_switch = AsyncMock(return_value=None)
    runner._record_switch_metrics = lambda *_a, **_k: None
    runner._model_switch_confirmation = AsyncMock(return_value="switched")
    ctx = _ModelSwitchContext(session_key="agent:main:telegram:group:-101", source=source,
                              config_path=None, persist_global=False)

    reply = await runner._commit_model_switch(
        NS(new_model="provider/team/model-x", target_provider="nous"), ctx, source=source)

    assert runner.notices == [(source, "agent:main:telegram:group:-101")]
    assert reply == "switched"


@pytest.mark.asyncio
async def test_switch_recompose_carries_the_new_model_and_effort():
    """End to end through the real notify -> recompose -> schedule -> lane path: the title the
    transport receives is composed from the store AFTER the switch, not from a string cached at the
    first rename. Reads the expected value back out of the store so the assertion tracks the
    relationship, not a literal.

    This is also the proof that T5 added no second rename path — the recompose lands in the same
    lane, so the kill-switch and the budget apply to it unchanged.
    """
    adapter = _adapter()
    runner = _wired_runner(adapter)
    runner.config = SimpleNamespace(platforms={Platform.TELEGRAM: SimpleNamespace(extra={})})
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _title_session(db, "session-switch", title="Fix login", model="openrouter/gpt-x",
                       started_at=time.time())
        # The opening rename publishes the pre-switch name.
        await _fire(adapter, runner, source, "session-switch", "Fix login")
        assert [text for _c, text, _h in adapter._bot.renames] == [
            compose_group_title("Fix login", "openrouter/gpt-x", None)]

        # A /model switch lands in the store; the recompose must notice it without a restart.
        db._write_sql("UPDATE sessions SET model = ? WHERE id = ?",
                      ("openrouter/gpt-y", "session-switch"))
        _install_store(runner, session_id="session-switch")
        runner._notify_telegram_group_title_of_switch(source, "agent:main:telegram:group:-101")
        await _await_renames(adapter._bot, 2)

        assert adapter._bot.renames[-1][1] == compose_group_title(
            "Fix login", db.get_session("session-switch")["model"], None)
        assert "gpt-y" in adapter._bot.renames[-1][1]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_switch_recompose_respects_the_kill_switch():
    """The recompose rides the existing scheduler, so the operator knob stops it with no second
    guard to keep in sync (criterion 11)."""
    adapter = _adapter()
    runner = _wired_runner(adapter)
    runner.config = SimpleNamespace(platforms={
        Platform.TELEGRAM: SimpleNamespace(extra={"disable_group_auto_rename": True})})
    source = adapter.build_source(chat_id="-101", chat_type="group")
    db = _ambient_db()
    try:
        _title_session(db, "session-killed", title="Fix login", model="openrouter/gpt-x",
                       started_at=time.time())
        _install_store(runner, session_id="session-killed")
        runner._notify_telegram_group_title_of_switch(source, "agent:main:telegram:group:-101")
        await asyncio.sleep(0.05)
        assert adapter._bot.renames == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_recompose_swallows_store_failure():
    """A store read that raises must not reach the slash-command caller: the reply is already
    being built, so a recompose can only ever be best-effort."""
    adapter = _adapter()
    runner = _wired_runner(adapter)
    runner.config = SimpleNamespace(platforms={Platform.TELEGRAM: SimpleNamespace(extra={})})
    source = adapter.build_source(chat_id="-101", chat_type="group")
    _install_store(runner, error=RuntimeError("store down"))

    runner._notify_telegram_group_title_of_switch(source, "agent:main:telegram:group:-101")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert adapter._bot.renames == []


def test_notify_without_a_loop_is_silent():
    """A sync caller with no running loop (shutdown, an off-loop test double) must get a no-op,
    not a RuntimeError: ``/reasoning`` still has to return its reply."""
    adapter = _adapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    source = adapter.build_source(chat_id="-101", chat_type="group")

    # No running loop in a plain sync test: the door returns before touching the transport.
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()
    runner._notify_telegram_group_title_of_switch(source, "agent:main:telegram:group:-101")
    assert adapter._bot.renames == []

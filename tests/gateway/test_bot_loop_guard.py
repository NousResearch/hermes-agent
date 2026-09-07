"""Native Telegram ingress invariants; based on the incident in PR #91483."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from gateway.bot_loop_guard import BotLoopGuard
from gateway.config import Platform, load_gateway_config
from gateway.session import SessionStore
from tests.gateway.test_telegram_auth_check import _make_adapter, _make_message


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    from gateway import bot_loop_guard
    from gateway.run import GatewayRunner

    for name in ("TELEGRAM_ALLOW_BOTS", "TELEGRAM_ALLOWED_USERS", "TELEGRAM_ALLOW_ALL_USERS",
                 "TELEGRAM_GROUP_ALLOWED_CHATS", "TELEGRAM_GROUP_ALLOWED_USERS",
                 "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    clock = SimpleNamespace(t=1000.0)
    monkeypatch.setattr(bot_loop_guard.time, "monotonic", lambda: clock.t)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"telegram": {
        "allow_bots": "mentions", "allow_from": ["111"],
        "allowed_chats": ["-100", "-200"], "group_allowed_chats": ["-100", "-200"],
        "bot_loop": {"max_events": 2, "window_seconds": 60, "max_hops": 4},
    }}))
    config = load_gateway_config()
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *a: False)
    adapter = _make_adapter(require_mention=True)
    adapter.config = config.platforms[Platform.TELEGRAM]
    adapter._session_store = runner.session_store
    adapter._owner_profile = "mito"
    runner.adapters = {Platform.TELEGRAM: adapter}
    # Observation stamps the owning profile before dispatch; register the same
    # test transport under both explicitly exercised profile routes.
    runner._profile_adapters = {
        profile: {Platform.TELEGRAM: adapter} for profile in ("mito", "trump")
    }
    adapter._message_handler = runner._handle_message
    # Keep native prefilter, trigger gating, event construction, and runner auth;
    # replace only scheduling/agent execution and the Telegram network transport.
    adapter.handle_message = runner._handle_message
    adapter._ensure_forum_commands = AsyncMock()
    queued = []
    adapter._enqueue_text_event = queued.append
    runner._hm_estop_gate = lambda *a: None
    delivered = []

    async def capture(event, source, key, generation):
        delivered.append(event)
        return "fake-agent-response"

    # Exercise the real cold acceptance gates; replace only agent execution and
    # post-turn persistence, not an earlier control-reply intercept.
    runner._sessions = {}
    runner._draining = runner._external_drain_active = False
    runner._restart_requested = False
    runner._busy_input_mode = "queue"
    runner._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    runner._handle_message_with_agent = capture
    runner._run_post_turn_hooks = AsyncMock()
    runner._persist_active_agents = lambda: None
    runner._clear_durable_active_turn = AsyncMock()

    async def send(mid, *, bot=True, media=False, text="@test_bot hello", chat=-100,
                   thread=None, profile="mito", command=False, private=False, sender=None, anonymous=False):
        adapter._owner_profile = profile
        msg = _make_message(text=None if media else text, from_user_id=sender or (222 if bot else 111),
                            chat_id=chat, chat_type="private" if private else "supergroup")
        msg.message_id = mid
        msg.from_user.is_bot = bot
        if anonymous:
            msg.from_user = None
        msg.message_thread_id = thread
        msg.is_topic_message = thread is not None
        msg.chat.is_forum = thread is not None
        msg.caption = text if media else None
        entities = [SimpleNamespace(type="mention", offset=0, length=9)] if text.startswith("@test_bot") else []
        msg.entities = [] if media else entities
        msg.caption_entities = entities if media else []
        if media:
            file = SimpleNamespace(file_path="test.ogg", download_as_bytearray=AsyncMock(return_value=b"fake voice"))
            msg.voice = SimpleNamespace(file_size=10, get_file=AsyncMock(return_value=file))
        before = len(delivered)
        handler = adapter._handle_command if command else (adapter._handle_media_message if media else adapter._handle_text_message)
        await handler(SimpleNamespace(update_id=mid, message=msg, effective_message=msg), SimpleNamespace())
        while queued:
            await adapter.handle_message(queued.pop(0))
        return len(delivered) - before

    return SimpleNamespace(send=send, runner=runner, adapter=adapter, clock=clock, delivered=delivered)


def test_delayed_human_preserves_newer_peer_replay_evidence_and_hops():
    guard = BotLoopGuard()
    key = ("default", "telegram", "-1001", "")
    settings = {"max_hops": 2, "max_events": 20, "window_seconds": 60}

    assert guard.admit(key, "100", False, settings)
    assert guard.admit(key, "102", True, settings)
    assert guard.admit(key, "104", True, settings)
    assert guard.admit(key, "101", False, settings)  # delayed human
    assert not guard.admit(key, "102", True, settings)
    assert not guard.admit(key, "104", True, settings)
    assert not guard.admit(key, "103", True, settings)  # surviving peers retain both hops

    assert guard.admit(key, "105", False, settings)  # genuinely newer human resets
    assert guard.admit(key, "106", True, settings)


@pytest.mark.asyncio
@pytest.mark.parametrize("media", [False, True])
async def test_native_bot_admission_is_not_control_or_a_chat_gate_bypass(pipeline, media):
    from gateway.platforms.base import BasePlatformAdapter

    p = pipeline
    # Even an open/free-response group only admits explicitly addressed bots in mentions mode.
    p.adapter.config.extra["require_mention"] = False
    assert await p.send(1, media=media, text="passive bot report") == 0
    assert await p.send(2, media=media, chat=-999) == 0
    p.adapter.config.extra["guest_mode"] = True
    assert await p.send(3, media=media, chat=-999) == 0
    p.adapter.config.extra["guest_mode"] = False
    assert await p.send(4, media=media) == 1
    assert p.delivered[-1].allow_gateway_control is False
    assert await p.send(5, media=media) == 1  # passive/rejected traffic used no budget
    assert await p.send(6, media=media) == 0
    assert await p.send(4, media=media, thread=17) == 1
    assert await p.send(4, media=media, chat=-200) == 1
    assert await p.send(4, media=media, profile="trump") == 1
    # Human DMs don't share group state; anonymous/non-triggering human chatter cannot reset it.
    assert await p.send(7, bot=False, private=True) == 1
    p.adapter.config.extra["require_mention"] = True
    assert await p.send(8, bot=False, media=media, text="side chatter") == 0
    assert await p.send(9, media=media) == 0

    # Exercise the real Base active-session command guard: /stop must not cancel
    # an in-flight human task, regardless of native COMMAND vs text/caption path.
    original_handle = p.adapter.handle_message
    p.adapter.handle_message = BasePlatformAdapter.handle_message.__get__(p.adapter)
    p.adapter._event_session_key = lambda event: "human-busy"
    p.adapter._active_sessions["human-busy"] = asyncio.Event()
    p.adapter._heal_stale_session_lock = lambda key: None
    p.adapter._dispatch_active_session_command = AsyncMock()
    for command in (False, True):
        assert await p.send(10, media=media and not command, text="/stop@test_bot", command=command) == 0
    p.adapter._dispatch_active_session_command.assert_not_awaited()
    assert not p.adapter._active_sessions["human-busy"].is_set()
    p.adapter.handle_message = original_handle
    p.adapter.config.extra["require_mention"] = False
    p.adapter.config.extra["allow_bots"] = "none"
    assert await p.send(20, media=media, profile="policy") == 0
    p.adapter.config.extra["allow_bots"] = "all"
    p.adapter.config.extra["group_policy"] = "disabled"
    assert await p.send(20, media=media, profile="policy") == 0
    p.adapter.config.extra["group_policy"] = "open"
    p.adapter.config.extra["bot_loop"] = {"max_hops": 1.5, "max_events": False, "window_seconds": []}
    assert await p.send(20, media=media, profile="policy", text="bot report") == 1
    assert await p.send(21, media=media, profile="policy", text="bot report") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("media", [False, True])
@pytest.mark.parametrize("observe", [False, True])
async def test_one_dispatch_one_budget_and_slow_chain_requires_human(pipeline, media, observe):
    p = pipeline
    p.adapter.config.extra["observe_unmentioned_group_messages"] = observe
    assert await p.send(1, bot=False, media=media) == 1
    assert await p.send(2, media=media) == 1
    # Authorization can be asked any number of times without spending budget.
    for _ in range(5):
        assert p.runner._is_user_authorized(p.delivered[-1].source)
    assert await p.send(2, media=media) == 0  # fresh update object, same native message ID
    assert await p.send(3, media=media, sender=333) == 1
    assert await p.send(4, media=media) == 0  # sliding-window rate limit
    p.clock.t += 61
    assert await p.send(5, media=media) == 1
    p.clock.t += 61
    assert await p.send(6, media=media) == 1
    for mid in (7, 8, 9):
        p.clock.t += 10000
        assert await p.send(mid, media=media) == 0  # no time-based hard-cap reset
    assert await p.send(9, bot=False, media=media, anonymous=True) == 1
    assert await p.send(12, media=media) == 0  # anonymous/channel posts are not human intervention
    assert await p.send(1, bot=False, media=media) == 1
    assert await p.send(9, media=media) == 0  # replay of the initiating human is not a new chain
    assert await p.send(10, bot=False, media=media) == 1
    assert await p.send(11, media=media) == 1
    assert await p.send(10, bot=False, media=media) == 1  # humans remain unaffected
    assert await p.send(11, media=media) == 0  # replayed human must not reset bot dedupe


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["interrupt", "steer", "queue"])
async def test_native_busy_admission_shares_cold_budget_and_consumes_receipt_once(pipeline, mode, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    p = pipeline
    runner = p.runner
    runner._sessions = {}
    runner._draining = False
    runner._restart_requested = False
    runner._busy_input_mode = mode
    runner._busy_text_mode = "queue"
    runner.config.group_sessions_per_user = False
    p.adapter.config.extra["group_sessions_per_user"] = False
    assert await p.send(1, bot=False) == 1  # initiating human predates the first bot
    assert await p.send(2) == 1  # cold ingress spends the same budget
    cold_handle = p.adapter.handle_message

    async def busy(event):
        key = runner._session_key_for_source(event.source)
        await runner._handle_active_session_busy_message(event, key)

    p.adapter.handle_message = busy
    await p.send(3)
    key = runner._session_key_for_source(p.delivered[0].source)
    peer = p.adapter._pending_messages[key]
    assert peer.message_id == "3"
    await busy(peer)  # the same event object must not reuse its queue receipt
    await p.send(3)  # nor may a fresh update replay the native ID
    assert runner._queue_depth(key, adapter=p.adapter) == 1
    await p.send(4)
    assert runner._queue_depth(key, adapter=p.adapter) == 1  # rate limit
    p.clock.t += 61
    await p.send(5)
    p.clock.t += 61
    await p.send(6)
    assert runner._queue_depth(key, adapter=p.adapter) == 3
    p.clock.t += 10000
    await p.send(7)
    assert runner._queue_depth(key, adapter=p.adapter) == 3  # hard hop cap

    # Base's fallback drain re-enters cold admission; the exact queued event
    # gets one pass, not another charge (or a permanent dedupe exemption).
    assert await runner._hm_admit_event(peer) is None  # still queued: redelivery, not handoff
    assert p.adapter._pending_messages.pop(key) is peer
    assert await runner._hm_admit_event(peer) is not None
    assert await runner._hm_admit_event(peer) is None
    await p.send(8, bot=False, anonymous=True)
    await p.send(9, bot=False, sender=999)
    await p.send(1, bot=False)  # old human replay cannot reopen the chain
    depth = runner._queue_depth(key, adapter=p.adapter)
    await p.send(10)
    assert runner._queue_depth(key, adapter=p.adapter) == depth
    await p.send(11, bot=False)
    depth = runner._queue_depth(key, adapter=p.adapter)
    await p.send(12)
    assert runner._queue_depth(key, adapter=p.adapter) == depth + 1

    # A metadata field supplied with an event is not an admission receipt.
    p.adapter.handle_message = cold_handle
    peer.metadata["_hermes_bot_budget_admitted"] = True
    assert await runner._hm_admit_event(peer) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["pause", "external_drain", "session_cap"])
async def test_cold_refusal_preserves_peer_budget_for_retry(pipeline, monkeypatch, tmp_path, refusal):
    from agent import estop
    from gateway.run_inbound import GatewayInboundMixin
    from hermes_cli.active_sessions import try_acquire_active_session

    p = pipeline
    p.adapter.config.extra["bot_loop"] = {"max_hops": 1, "max_events": 1}
    lease = None
    # Restore the real pause gate, confined to this test's Hermes home.
    monkeypatch.setattr(estop, "_canonical_root", lambda: tmp_path)
    p.runner._hm_estop_gate = GatewayInboundMixin._hm_estop_gate.__get__(p.runner)
    if refusal == "pause":
        estop.engage("peer acceptance test")
    elif refusal == "external_drain":
        p.runner._external_drain_active = True
    else:
        p.runner.config.max_concurrent_sessions = 1
        lease, error = try_acquire_active_session(
            session_id="other-active-session", surface="test", config=p.runner.config,
        )
        assert lease is not None and error is None
    try:
        assert await p.send(100) == 0
        assert p.delivered == []
    finally:
        estop.disengage()
        p.runner._external_drain_active = False
        if lease is not None:
            lease.release()

    assert await p.send(100) == 1  # the rejected native ID was not charged/seen
    assert await p.send(100) == 0
    assert await p.send(101) == 0  # neither hop nor rate capacity was reopened


def test_preflight_and_failed_retention_do_not_change_peer_budget(monkeypatch):
    clock = SimpleNamespace(t=1000.0)
    monkeypatch.setattr("gateway.bot_loop_guard.time.monotonic", lambda: clock.t)
    guard = BotLoopGuard()
    key = ("default", "telegram", "chat", "thread")
    settings = {"max_hops": 1, "max_events": 1}
    assert guard.admit(key, "1", True, settings, check_only=True)
    assert guard._conversations == {}
    assert not guard.admit(key, "1", True, settings, accept=lambda: False)

    def failed_retention():
        raise RuntimeError("queue unavailable")

    with pytest.raises(RuntimeError, match="queue unavailable"):
        guard.admit(key, "1", True, settings, accept=failed_retention)
    assert guard._conversations == {}

    def retained():
        clock.t += 61  # rate accounting belongs to acceptance, not preflight
        return True

    assert guard.admit(key, "1", True, settings, accept=retained)
    assert not guard.admit(key, "1", True, settings)
    assert not guard.admit(key, "2", True, settings)
    assert guard.admit(key, "2", False, settings)
    assert not guard.admit(key, "3", True, settings)  # human resets hops, not the rate window

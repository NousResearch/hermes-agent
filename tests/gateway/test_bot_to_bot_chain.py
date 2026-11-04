"""Bot-to-bot exchanges end (#137027): the gateway's chain rules and the Bot Mode hop cap.

Two Hermes gateways DMing each other on Mattermost ran 17 h because every inbound bot message
produced an outbound one (runtime notices included) and nothing counted the exchange.
"""

import json
from types import SimpleNamespace

import pytest

from gateway.bot_loop_guard import BotLoopGuard, BotLoopGuardSettings, configured_bot_user, settings_from_config
from gateway.session import Platform, SessionSource

CHAT = "dm_peer"
PEER = "peer_bot"
HUMAN = "alice"


@pytest.fixture
def clock():
    state = {"t": 1_000.0}
    return SimpleNamespace(now=lambda: state["t"], advance=lambda s: state.__setitem__("t", state["t"] + s))


def _guard(clock, **overrides):
    settings = BotLoopGuardSettings(**{"max_events": 1000, "window_seconds": 60, "cooldown_seconds": 600, **overrides})
    return BotLoopGuard(settings=lambda: settings, clock=clock.now)


# --- messaging gateway chain rules -------------------------------------------------------------


def test_a_bot_ping_pong_stops_at_max_hops_and_a_human_message_starts_a_new_chain(clock):
    guard = _guard(clock, max_hops=3, bot_replies_expect_reply=True)
    verdicts = [guard.chain_admit(CHAT, PEER, f"ping {n}", is_bot=True)[0] for n in range(10)]
    assert verdicts == [True, True, True] + [False] * 7

    assert guard.chain_admit(CHAT, HUMAN, "keep going", is_bot=False) == (True, "human")
    assert guard.chain_admit(CHAT, PEER, "ping 10", is_bot=True) == (True, "ok")


def test_a_bot_answering_this_bots_reply_starts_no_turn_by_default(clock):
    """The default reply rule: our answer to a bot needs no answer, so its answer to that ends it."""
    guard = _guard(clock)
    assert guard.chain_admit(CHAT, PEER, "what is 2+2?", is_bot=True) == (True, "ok")
    assert guard.chain_admit(CHAT, PEER, "thanks!", is_bot=True) == (False, "reply_to_reply")
    # Another bot in the same chat is a new ask, not a reply to our reply.
    assert guard.chain_admit(CHAT, "other_bot", "hello", is_bot=True) == (True, "ok")


def test_the_same_ask_repeated_inside_one_chain_is_one_hop(clock):
    guard = _guard(clock, bot_replies_expect_reply=True)
    assert guard.chain_admit(CHAT, PEER, "Interrupting current task", is_bot=True) == (True, "ok")
    assert guard.chain_admit(CHAT, "bot_b", "x", is_bot=True)[0] is True
    assert guard.chain_admit(CHAT, PEER, "  Interrupting   current task ", is_bot=True) == (False, "duplicate")


def test_chain_rules_follow_enabled_and_config(clock):
    off = BotLoopGuard(settings=lambda: BotLoopGuardSettings(enabled=False, max_hops=1), clock=clock.now)
    assert all(off.chain_admit(CHAT, PEER, "same", is_bot=True)[0] for _ in range(5))
    cfg = settings_from_config({"gateway": {"bot_loop_guard": {"max_hops": 7, "bot_replies_expect_reply": "yes"}}})
    assert (cfg.max_hops, cfg.bot_replies_expect_reply) == (7, True)
    assert settings_from_config({}).max_hops == 4 and settings_from_config({}).bot_replies_expect_reply is False


@pytest.mark.asyncio
async def test_ingress_drops_the_bot_reply_and_still_admits_the_human(monkeypatch, clock):
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._scale_to_zero_note_real_inbound = lambda: None

    async def _passthrough(event, source):
        return event
    runner._hm_pre_gateway_dispatch_hook = _passthrough
    runner._is_user_authorized_for_source = lambda source, **kw: True
    runner._bot_loop_guard = _guard(clock)
    runner._adapter_profile_for_source = lambda source: ""

    def _event(user, text, is_bot):
        source = SessionSource(platform=Platform.MATTERMOST, chat_id=CHAT, chat_type="dm", user_id=user, is_bot=is_bot)
        return MessageEvent(text=text, message_id=text, source=source)

    assert await runner._hm_admit_event(_event(PEER, "question", True)) is not None
    assert await runner._hm_admit_event(_event(PEER, "answer to your answer", True)) is None
    assert await runner._hm_admit_event(_event(HUMAN, "hi", False)) is not None
    assert await runner._hm_admit_event(_event(PEER, "new question", True)) is not None


def test_bot_users_marks_a_peer_on_a_platform_without_a_bot_flag():
    config = SimpleNamespace(extra={"bot_users": ["peer_bot", " other "]})
    assert configured_bot_user(config, "peer_bot") and configured_bot_user(config, "other")
    assert not configured_bot_user(config, HUMAN)
    assert configured_bot_user(SimpleNamespace(extra={"bot_users": "a,peer_bot"}), "peer_bot")
    assert not configured_bot_user(SimpleNamespace(extra={}), "peer_bot")


# --- Bot Mode message_agent hop cap --------------------------------------------------------------


@pytest.fixture
def bot_chain(monkeypatch):
    from agent import bot_chain

    monkeypatch.setattr(bot_chain, "_CHAINS", {})
    monkeypatch.setattr(bot_chain, "_settings", lambda: BotLoopGuardSettings(max_hops=4))
    return bot_chain


def _agent(session_id):
    return SimpleNamespace(session_id=session_id, _turn_author=None)


def _turn(chain, agent, author=None, kind=None):
    from agent.turn_author import parse_turn_author
    agent._turn_author = parse_turn_author(author) if author else None
    chain.on_turn_start(agent, agent._turn_author, kind)


def test_message_agent_ping_pong_is_refused_past_max_hops_on_both_sides(bot_chain, monkeypatch):
    """A -> B -> A -> B with the envelope's hop count: the send that would be hop 5 delivers nothing.
    Replies stay reply-expecting here so only the cap ends the exchange."""
    monkeypatch.setattr(bot_chain, "_settings", lambda: BotLoopGuardSettings(max_hops=4, bot_replies_expect_reply=True))
    alpha, beta = _agent("alpha-chat"), _agent("beta-chat")
    _turn(bot_chain, alpha, kind="internal_notification")  # the user's kick-off
    sent = []
    sender, receiver, envelope = alpha, beta, None
    for n in range(1, 8):
        if envelope is not None:
            _turn(bot_chain, receiver, {"id": f"bot:{sender.session_id[:-5]}", "name": "x", "is_bot": True, **envelope})
            sender, receiver = receiver, sender
        refusal, envelope = bot_chain.admit_send(sender, receiver.session_id[:-5], str(n))
        if refusal:
            assert "max_hops" in refusal
            break
        sent.append(n)
        envelope.setdefault("hops", 1)
    assert sent == [1, 2, 3, 4]


def test_the_hop_count_cannot_be_lowered_by_the_envelope(bot_chain):
    """Each side counts its own bot turns: an envelope claiming hop 1 every time still hits the cap."""
    beta = _agent("beta-chat")
    allowed = []
    for n in range(8):
        _turn(bot_chain, beta, {"id": "bot:alpha", "name": "alpha", "is_bot": True})
        allowed.append(bot_chain.admit_send(beta, "gamma", f"msg {n}")[0] is None)
    assert allowed == [True, True, True, False, False, False, False, False]


def test_reply_to_the_messaging_bot_is_notify_only_and_ends_the_exchange(bot_chain):
    beta = _agent("beta-chat")
    _turn(bot_chain, beta, {"id": "bot:alpha", "name": "alpha", "is_bot": True})
    refusal, envelope = bot_chain.admit_send(beta, "@alpha", "2")
    assert refusal is None and envelope["expects_reply"] is False

    alpha = _agent("alpha-chat")
    _turn(bot_chain, alpha, {"id": "bot:beta", "name": "beta", "is_bot": True, **envelope})
    refusal, _ = bot_chain.admit_send(alpha, "beta", "3")
    assert refusal and "needs no answer" in refusal
    # A third teammate is a new ask, still under the cap.
    assert bot_chain.admit_send(alpha, "gamma", "fyi")[0] is None


def test_identical_ask_is_one_hop_and_a_person_resets_the_chain(bot_chain):
    alpha = _agent("alpha-chat")
    _turn(bot_chain, alpha, {"id": "bot:beta", "name": "beta", "is_bot": True})
    assert bot_chain.admit_send(alpha, "gamma", "status?")[0] is None
    assert "already sent" in bot_chain.admit_send(alpha, "gamma", " status? ")[0]

    _turn(bot_chain, alpha)  # a person's message
    assert bot_chain.admit_send(alpha, "gamma", "status?")[0] is None


def test_message_agent_tool_refuses_past_the_cap_and_delivers_nothing(bot_chain, monkeypatch, tmp_path):
    """Through the real tool entry: the refusal is the tool result and no transport is started."""
    import tools.bot_mode_dm as dm
    import tools.bot_mode_probe as probe

    monkeypatch.setattr(dm, "_session_title", lambda agent: probe.BOT_CHAT_TITLE)
    monkeypatch.setattr(probe, "is_bot_mode_managed", lambda home: True)
    monkeypatch.setattr(dm, "_agent_home", lambda agent: str(tmp_path))
    monkeypatch.setattr(dm, "_start_delivery", lambda *a, **k: pytest.fail("delivered past the cap"))
    monkeypatch.setattr(dm, "_try_relay_delivery", lambda *a, **k: pytest.fail("delivered past the cap"))
    monkeypatch.setattr("hermes_cli.observability.shared_metrics_signals.record_feature_used", lambda *a, **k: None)
    agent = _agent("beta-chat")
    bot_chain._CHAINS["beta-chat"] = {"hops": 4, "sent": 4, "asks": set(), "updated": 1e18}
    agent._turn_author = {"id": "bot:alpha", "name": "alpha", "is_bot": True}
    result = json.loads(dm.message_agent_tool(target="gamma", message="5", agent=agent))
    assert "max_hops" in result["error"]

"""Bot Mode bot-to-bot chains (#137027): the runtime hop cap, repeated asks, the reply rule
(``agent/bot_chain.py``), keyed by the conversation through forks and compression."""

from types import SimpleNamespace

import pytest

from hermes_agent.gateway.bot_loop_guard import BotLoopGuardSettings
from hermes_agent.tools.bot_mode_probe import NO_REPLY_MARKER


@pytest.fixture
def bot_chain(monkeypatch):
    from agent import bot_chain

    monkeypatch.setattr(bot_chain, "_CHAINS", {})
    monkeypatch.setattr(bot_chain, "_settings", lambda: BotLoopGuardSettings(max_hops=4))
    return bot_chain


def _settings(bot_chain, monkeypatch, **overrides):
    monkeypatch.setattr(bot_chain, "_settings", lambda: BotLoopGuardSettings(**{"max_hops": 4, **overrides}))


def _agent(session_id):
    return SimpleNamespace(session_id=session_id, _turn_author=None)


def _turn(chain, agent, author=None, kind=None, message=""):
    from agent.turn_author import parse_turn_author
    agent._turn_author = parse_turn_author(author) if author else None
    chain.on_turn_start(agent, agent._turn_author, kind, message)


def _send(chain, agent, recipient, body):
    """One single-target call: ``(refusal, envelope, notify_only)``."""
    admission = chain.admit_send(agent, [recipient], body)
    refusal = admission.refusal or admission.refused.get(recipient)
    return refusal, admission.envelope(), recipient in admission.notify_only


def test_message_agent_ping_pong_is_refused_past_max_hops_on_both_sides(bot_chain, monkeypatch):
    """A -> B -> A -> B with the envelope's hop count: the send that would be hop 5 delivers nothing.
    Replies stay reply-expecting here so only the cap ends the exchange."""
    _settings(bot_chain, monkeypatch, bot_replies_expect_reply=True)
    bots = {"bot:alpha": _agent("alpha-chat"), "bot:beta": _agent("beta-chat")}
    _turn(bot_chain, bots["bot:alpha"], kind="internal_notification")  # the user's kick-off
    sent, sender, receiver, envelope = [], "bot:alpha", "bot:beta", None
    for n in range(1, 8):
        if envelope is not None:
            _turn(bot_chain, bots[receiver], {"id": sender, "name": "x", "is_bot": True, **envelope})
            sender, receiver = receiver, sender
        refusal, envelope, _ = _send(bot_chain, bots[sender], receiver, str(n))
        if refusal:
            assert "max_hops" in refusal
            break
        sent.append(n)
    assert sent == [1, 2, 3, 4]


def test_the_hop_count_cannot_be_lowered_by_the_envelope(bot_chain):
    """Each side counts its own bot turns: an envelope claiming hop 1 every time still hits the cap."""
    beta = _agent("beta-chat")
    allowed = []
    for n in range(8):
        _turn(bot_chain, beta, {"id": "bot:alpha", "name": "alpha", "is_bot": True})
        allowed.append(_send(bot_chain, beta, "bot:gamma", f"msg {n}")[0] is None)
    assert allowed == [True, True, True, False, False, False, False, False]


def test_every_send_in_one_bot_turn_is_another_hop(bot_chain, monkeypatch):
    """Greptile 4467193907: several distinct sends inside one bot-authored turn each advance the hop."""
    _settings(bot_chain, monkeypatch, max_hops=2)
    beta = _agent("beta-chat")
    _turn(bot_chain, beta, {"id": "bot:alpha", "name": "alpha", "is_bot": True, "hops": 1})
    first = _send(bot_chain, beta, "bot:gamma", "one")
    second = _send(bot_chain, beta, "bot:delta", "two")
    assert first[0] is None and first[1] == {"hops": 2}
    assert second[0] and "max_hops" in second[0]


def test_a_person_may_fan_out_and_send_twice_without_spending_the_chain(bot_chain, monkeypatch):
    _settings(bot_chain, monkeypatch, max_hops=1)
    alpha = _agent("alpha-chat")
    _turn(bot_chain, alpha)  # a person's turn
    assert [_send(bot_chain, alpha, f"bot:{n}", "status?")[0] for n in ("b", "c", "b")] == [None, None, None]


def test_reply_to_the_messaging_bot_is_notify_only_and_ends_the_exchange(bot_chain):
    beta = _agent("beta-chat")
    _turn(bot_chain, beta, {"id": "bot:alpha", "name": "alpha", "is_bot": True})
    refusal, envelope, notify_only = _send(bot_chain, beta, "bot:alpha", "2")
    assert refusal is None and notify_only

    # The notify-only message reaches alpha with #144789's marker in its text.
    alpha = _agent("alpha-chat")
    _turn(bot_chain, alpha, {"id": "bot:beta", "name": "beta", "is_bot": True, **envelope},
          message=f"Message from 🤖 Beta (@beta): {NO_REPLY_MARKER} 2")
    refusal, *_ = _send(bot_chain, alpha, "bot:beta", "3")
    assert refusal and "needs no answer" in refusal
    assert _send(bot_chain, alpha, "bot:gamma", "fyi")[0] is None  # a third bot is a new ask


def test_the_reply_rule_compares_full_identities(bot_chain):
    """Greptile 4467193900: same profile name on another connection is a different bot."""
    beta = _agent("beta-chat")
    _turn(bot_chain, beta, {"id": "bot:conn-a/worker", "name": "worker", "is_bot": True},
          message=f"{NO_REPLY_MARKER} done")
    assert _send(bot_chain, beta, "bot:conn-b/worker", "hi")[0] is None
    assert "needs no answer" in _send(bot_chain, beta, "bot:conn-a/worker", "hi")[0]


def test_identical_ask_is_one_hop_and_a_person_resets_the_chain(bot_chain):
    alpha = _agent("alpha-chat")
    _turn(bot_chain, alpha, {"id": "bot:beta", "name": "beta", "is_bot": True})
    assert _send(bot_chain, alpha, "bot:gamma", "status?")[0] is None
    assert "already sent" in _send(bot_chain, alpha, "bot:gamma", " status? ")[0]

    _turn(bot_chain, alpha)  # a person's message
    assert _send(bot_chain, alpha, "bot:gamma", "status?")[0] is None


def test_a_send_that_never_left_releases_its_ask_and_hop(bot_chain, monkeypatch):
    """Greptile 4467193919: a definite no-send (target offline) leaves no duplicate and no spent hop."""
    _settings(bot_chain, monkeypatch, max_hops=2)
    beta = _agent("beta-chat")
    _turn(bot_chain, beta, {"id": "bot:alpha", "name": "alpha", "is_bot": True})
    failed = bot_chain.admit_send(beta, ["bot:conn/gamma"], "ping")
    failed.release("bot:conn/gamma")
    retry = bot_chain.admit_send(beta, ["bot:conn/gamma"], "ping")
    assert retry.refusal is None and not retry.refused and retry.hop == failed.hop == 2


def test_a_review_fork_neither_resets_nor_advances_the_chain(bot_chain):
    """Greptile 4467194024: the fork shares the session id but runs without a bot author."""
    beta = _agent("beta-chat")
    _turn(bot_chain, beta, {"id": "bot:alpha", "name": "alpha", "is_bot": True, "hops": 4})
    fork = _agent("beta-chat")
    fork._turn_origin = "background_review"
    _turn(bot_chain, fork)
    assert "max_hops" in _send(bot_chain, beta, "bot:gamma", "5")[0]


def test_compression_rotation_and_turn_start_recovery_keep_the_chain(bot_chain):
    """Greptile 4467194037: the chain follows the conversation onto its compression child id."""
    beta = _agent("parent")
    _turn(bot_chain, beta, {"id": "bot:alpha", "name": "alpha", "is_bot": True, "hops": 4})
    beta.session_id = "child"  # recovery / lease reload swapped the id under the agent
    assert "max_hops" in _send(bot_chain, beta, "bot:gamma", "x")[0]

    gamma = _agent("g-parent")
    _turn(bot_chain, gamma, {"id": "bot:alpha", "name": "alpha", "is_bot": True, "hops": 4})
    bot_chain.carry_chain("g-parent", "g-child")  # the rotation's state carry
    assert "max_hops" in _send(bot_chain, _agent("g-child"), "bot:delta", "x")[0]


def test_a_disabled_guard_turns_the_bot_mode_rules_off(bot_chain, monkeypatch):
    """Greptile 4467194052: gateway.bot_loop_guard.enabled false is honoured in Bot Mode too."""
    _settings(bot_chain, monkeypatch, enabled=False, max_hops=1)
    beta = _agent("beta-chat")
    for n in range(4):
        _turn(bot_chain, beta, {"id": "bot:alpha", "name": "alpha", "is_bot": True, "hops": 9},
              message=f"{NO_REPLY_MARKER} hi")
        refusal, envelope, notify_only = _send(bot_chain, beta, "bot:alpha", "same")
        assert (refusal, envelope, notify_only) == (None, {}, False)

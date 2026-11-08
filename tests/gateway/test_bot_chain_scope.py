"""Gateway exchange scope follows live YAML while traffic budgets stay channel-wide."""

import json
from types import SimpleNamespace

import pytest

from gateway.bot_loop_guard import BotLoopGuard, load_settings
from gateway.platforms.base import MessageEvent
from gateway.session import Platform, SessionSource


@pytest.fixture
def setup_gateway(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    home = tmp_path / "transport"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    runner = object.__new__(GatewayRunner)
    runner._bot_loop_guard = BotLoopGuard(clock=lambda: 1000.0)
    runner._adapter_profile_for_source = lambda source: source.profile or ""
    runner._is_user_authorized_for_source = lambda source, **kw: True
    runner._scale_to_zero_note_real_inbound = lambda: None

    async def passthrough(event, source):
        return event

    runner._hm_pre_gateway_dispatch_hook = passthrough

    def configure(**block):
        (home / "config.yaml").write_text(
            json.dumps({"gateway": {"bot_loop_guard": block}}), encoding="utf-8")

    return SimpleNamespace(runner=runner, configure=configure, home=home)


def _event(thread="a", text="question", *, is_bot=True, chat="chat", profile="", platform=Platform.TELEGRAM):
    return MessageEvent(text=text, message_id=text, source=SessionSource(
        platform=platform, chat_id=chat, chat_type="group", thread_id=thread,
        user_id="peer" if is_bot else "human", is_bot=is_bot, profile=profile))


@pytest.mark.asyncio
@pytest.mark.parametrize("scope,other_thread_allowed", [("channel", False), ("thread", True)])
async def test_ingress_duplicate_scope_and_channel_budget(setup_gateway, scope, other_thread_allowed):
    env = setup_gateway
    env.configure(chain_scope=scope, max_events=2)
    runner = env.runner
    assert await runner._hm_admit_event(_event("a")) is not None
    assert (await runner._hm_admit_event(_event("b")) is not None) is other_thread_allowed
    # Even distinct thread chains spend the same channel budget.
    assert await runner._hm_admit_event(_event("c", "different")) is None
    assert not runner._bot_loop_guard.blocked(("", "telegram", "other-chat"))
    assert runner._bot_loop_guard.blocked(("", "telegram", "chat"))
    # Human input still passes during the channel's cooldown.
    assert await runner._hm_admit_event(_event("c", is_bot=False)) is not None


@pytest.mark.parametrize("scope", ["channel", "thread"])
def test_same_thread_duplicate_and_cross_thread_hop_cap(setup_gateway, scope):
    env = setup_gateway
    env.configure(chain_scope=scope, max_hops=1)
    admit = env.runner._admit_bot_chain
    assert admit(_event("a"))
    assert not admit(_event("a", "  question  "))
    assert not admit(_event("a", "new question"))
    assert admit(_event("b", "new question")) is (scope == "thread")


@pytest.mark.parametrize("scope", ["channel", "thread"])
def test_delivered_replies_and_human_resets_use_same_scope(setup_gateway, scope):
    env = setup_gateway
    env.configure(chain_scope=scope)
    runner = env.runner
    first = _event("a")
    assert runner._admit_bot_chain(first)
    runner._note_bot_reply(first.source)
    assert not runner._admit_bot_chain(_event("a", "thanks"))
    assert runner._admit_bot_chain(_event("b", "another question")) is (scope == "thread")
    assert runner._admit_bot_chain(_event("b", "continue", is_bot=False))
    assert runner._admit_bot_chain(_event("a", "question")) is (scope == "channel")
    assert runner._admit_bot_chain(_event("a", "continue", is_bot=False))
    assert runner._admit_bot_chain(_event("a", "question"))


@pytest.mark.parametrize("scope", ["channel", "thread"])
def test_chain_identity_isolates_profile_platform_and_chat(setup_gateway, scope):
    env = setup_gateway
    env.configure(chain_scope=scope)
    admit = env.runner._admit_bot_chain
    assert admit(_event())
    assert admit(_event(profile="other"))
    assert admit(_event(platform=Platform.DISCORD))
    assert admit(_event(chat="other"))
    assert not admit(_event())


def test_thread_keys_preserve_boundaries_and_missing_threads_share_channel(setup_gateway):
    env = setup_gateway
    env.configure(chain_scope="thread")
    admit = env.runner._admit_bot_chain
    assert admit(_event("b:c", chat="a"))
    assert admit(_event("c", chat="a:b"))
    assert admit(_event(None))
    assert not admit(_event(""))
    assert admit(_event("a"))
    assert admit(_event("", is_bot=False))
    assert not admit(_event("a"))
    assert admit(_event(None))


@pytest.mark.parametrize("block", [{}, {"chain_scope": "invalid"}, {"chain_scope": None},
                                  {"chain_scope": []}, {"chain_scope": True}])
def test_absent_or_invalid_scope_keeps_channel_behavior(setup_gateway, block):
    env = setup_gateway
    env.configure(**block)
    assert env.runner._admit_bot_chain(_event("a"))
    assert not env.runner._admit_bot_chain(_event("b"))
    assert load_settings().chain_scope == "channel"


def test_live_scope_reads_and_reply_tracking_use_transport_profile(setup_gateway, tmp_path):
    from gateway.run import _profile_runtime_scope

    env = setup_gateway
    routed = tmp_path / "routed"
    routed.mkdir()
    (routed / "config.yaml").write_text(
        json.dumps({"gateway": {"bot_loop_guard": {"chain_scope": "channel", "enabled": False}}}),
        encoding="utf-8")
    env.runner._adapter_profile_for_source = lambda source: "transport"

    def event(thread, text="question"):
        result = _event(thread, text, profile="routed")
        result.source._authorization_profile_home = env.home
        return result

    with _profile_runtime_scope(routed):
        env.configure(chain_scope="channel")
        assert env.runner._admit_bot_chain(event("a"))
        assert not env.runner._admit_bot_chain(event("b"))
        env.configure(chain_scope="thread")
        assert env.runner._admit_bot_chain(event("a"))
        assert env.runner._admit_bot_chain(event("b"))
        env.runner._note_bot_reply(event("a").source)
        assert not env.runner._admit_bot_chain(event("a", "thanks"))
        assert env.runner._admit_bot_chain(event("b", "follow-up"))

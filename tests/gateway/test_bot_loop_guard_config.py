"""``gateway.bot_loop_guard.max_hops`` / ``bot_replies_expect_reply`` read from a real config.yaml through
the loader both consumers use (``gateway.bot_loop_guard.load_settings``): the messaging gateway's chain
rules and Bot Mode's ``agent/bot_chain.py`` (#137027)."""

import json

import pytest

from hermes_agent.gateway.bot_loop_guard import BotLoopGuard, load_settings


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    def write(block: dict) -> None:
        # JSON is YAML; the CI test env ships no pyyaml for fixtures.
        (home / "config.yaml").write_text(json.dumps({"gateway": {"bot_loop_guard": block}}), encoding="utf-8")
    return write


def test_max_hops_and_reply_rule_from_config_yaml_drive_the_gateway_chain(config_home):
    config_home({"max_hops": 2, "bot_replies_expect_reply": True})
    assert (load_settings().max_hops, load_settings().bot_replies_expect_reply) == (2, True)

    guard = BotLoopGuard()
    verdicts = []
    for n in range(4):
        verdicts.append(guard.chain_admit("chat", "peer", f"ping {n}", is_bot=True)[1])
        guard.note_reply("chat", "peer")  # replies expected: answering an answer is allowed
    assert verdicts == ["ok", "ok", "max_hops", "max_hops"]


def test_max_hops_from_config_yaml_caps_bot_mode_sends(config_home, monkeypatch):
    from types import SimpleNamespace

    from agent import bot_chain

    monkeypatch.setattr(bot_chain, "_CHAINS", {})
    config_home({"max_hops": 2, "bot_replies_expect_reply": False})
    agent = SimpleNamespace(session_id="s", _turn_author={"id": "bot:alpha", "name": "alpha", "is_bot": True})
    bot_chain.on_turn_start(agent, agent._turn_author, None, "hi")

    reply = bot_chain.admit_send(agent, ["bot:alpha"], "answer")
    assert reply.refusal is None and reply.notify_only == {"bot:alpha"}  # bot_replies_expect_reply: false
    assert "limit of 2" in str(bot_chain.admit_send(agent, ["bot:gamma"], "next").refusal)

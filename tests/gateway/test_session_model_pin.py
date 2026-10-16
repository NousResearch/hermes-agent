"""Opt-in ``gateway.pin_session_model``: a live chat keeps the route it last ran on.

Without the flag the gateway re-reads ``model.default`` and ``agent.reasoning_effort`` on every
message, so a config edit moves an open chat to another model mid-conversation and the whole
history is re-sent uncached. These tests drive the real turn-end sync (``_sync_session_model_from_agent``)
into a real ``SessionDB``, edit a real ``config.yaml``, and resolve the next message's route.
"""

from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.run_turn_prepare import GatewayTurnPrepareMixin
from hermes_constants import get_hermes_home
from hermes_state import SessionDB

SKEY = "agent:main:telegram:dm:42"
OLD = ("claude-opus-4-6", "anthropic")
NEW = ("gpt-5.5", "openai-codex")


class _Env:
    def __init__(self, monkeypatch):
        self.home = get_hermes_home()
        monkeypatch.setattr(gateway_run, "_hermes_home", self.home)
        self.db = SessionDB(self.home / "state.db")
        self.session_id = "s1"
        self.provider = None
        from hermes_cli import runtime_provider
        monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", self._resolve)

    def _resolve(self, *, requested=None, target_model=None, **_):
        provider = requested or self.provider
        return {"api_key": f"key-{provider}", "base_url": f"https://{provider}.example/v1", "provider": provider,
                "api_mode": "chat_completions", "command": None, "args": [], "credential_pool": None}

    def config(self, route, *, pin, effort="low"):
        self.provider = route[1]
        lines = [f"model:\n  default: {route[0]}\n  provider: {route[1]}\n", f"agent:\n  reasoning_effort: {effort}\n"]
        if pin is not None:
            lines.append(f"gateway:\n  pin_session_model: {'true' if pin else 'false'}\n")
        (self.home / "config.yaml").write_text("".join(lines), encoding="utf-8")

    def runner(self):
        runner = object.__new__(gateway_run.GatewayRunner)
        runner.config = None
        runner._session_db = SimpleNamespace(_db=self.db)
        runner.session_store = SimpleNamespace(peek_session_id=lambda key: self.session_id,
                                               get_model_override=lambda key: None)
        return runner

    def finish_turn(self, runner, route, *, reasoning=None, fallback=False):
        """The gateway's own post-turn write of the route the agent ran on."""
        self.db.create_session(self.session_id, "telegram", model=route[0])
        agent = SimpleNamespace(model=route[0], provider=route[1], base_url=f"https://{route[1]}.example/v1",
                                api_mode="chat_completions", _fallback_activated=fallback, reasoning_config=reasoning)
        GatewayTurnPrepareMixin._sync_session_model_from_agent(runner, self.session_id, agent)

    def next_route(self, runner):
        model, runtime = runner._resolve_session_agent_runtime(session_key=SKEY)
        return model, runtime["provider"]


@pytest.fixture
def env(monkeypatch):
    return _Env(monkeypatch)


def test_pinned_chat_keeps_its_model_after_a_config_edit(env):
    env.config(OLD, pin=True)
    runner = env.runner()
    env.finish_turn(runner, OLD)
    env.config(NEW, pin=True)

    assert env.next_route(runner) == OLD


def test_pinned_chat_keeps_its_reasoning_effort_after_a_config_edit(env):
    env.config(OLD, pin=True, effort="low")
    runner = env.runner()
    low = runner._resolve_session_reasoning_config(session_key=SKEY, model=OLD[0])
    env.finish_turn(runner, OLD, reasoning=low)
    env.config(OLD, pin=True, effort="high")

    assert runner._resolve_session_reasoning_config(session_key=SKEY, model=OLD[0]) == low


@pytest.mark.parametrize("pin", [False, None])
def test_config_edits_reach_live_chats_by_default(env, pin):
    env.config(OLD, pin=pin, effort="low")
    runner = env.runner()
    env.finish_turn(runner, OLD, reasoning={"enabled": True, "effort": "low"})
    env.config(NEW, pin=pin, effort="high")

    assert env.next_route(runner) == NEW
    assert runner._resolve_session_reasoning_config(session_key=SKEY, model=NEW[0])["effort"] == "high"


def test_a_new_session_follows_the_edited_config(env):
    env.config(OLD, pin=True)
    runner = env.runner()
    env.finish_turn(runner, OLD)
    env.config(NEW, pin=True)
    env.session_id = "s2"  # /new

    assert env.next_route(runner) == NEW


def test_a_turn_that_ended_on_a_provider_fallback_is_not_pinned(env):
    env.config(OLD, pin=True)
    runner = env.runner()
    env.finish_turn(runner, ("fallback-model", "openrouter"), fallback=True)

    assert env.next_route(runner) == OLD


def test_the_pin_survives_a_compression_rotation(env):
    """A rotated child row starts without a recorded route; its compression parent holds the pin."""
    env.config(OLD, pin=True)
    runner = env.runner()
    env.finish_turn(runner, OLD)
    env.db.end_session("s1", "compression")
    env.db.create_session("s1-child", "telegram", model=NEW[0], parent_session_id="s1")
    env.session_id = "s1-child"
    env.config(NEW, pin=True)

    assert env.next_route(runner) == OLD

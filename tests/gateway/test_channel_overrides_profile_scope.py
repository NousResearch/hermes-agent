"""``channel_overrides`` belong to the profile that serves the turn (#136198).

Chat ids repeat across profiles' bots: a Telegram DM's chat id is the user's id, so under
multiplexing the launch profile's override for that DM must not steer another profile's bot in the
same DM (its model, its system prompt, ``/model``'s channel route). Real runner, real loaders.
"""

from __future__ import annotations

import json

from types import SimpleNamespace

import pytest

from agent.secret_scope import set_multiplex_active
from hermes_agent.gateway.config import Platform, load_gateway_config
from hermes_agent.gateway.run import GatewayRunner, _profile_runtime_scope
from hermes_agent.gateway.run_turn_runner import TurnRunner
from hermes_agent.gateway.session import SessionSource
from hermes_agent.gateway.session_identity import RoutingIdentity


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch = tmp_path / ".hermes"
    b2 = launch / "profiles" / "b2"
    b2.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    route = {"provider": "custom", "base_url": "http://127.0.0.1:9/v1"}
    (launch / "config.yaml").write_text(json.dumps({
        "model": {**route, "default": "launch/default"},
        "platforms": {"telegram": {"enabled": True, "channel_overrides": {
            "777": {"model": "launch/override", "system_prompt": "LAUNCH PROMPT"}}}}}), encoding="utf-8")
    (b2 / "config.yaml").write_text(json.dumps({
        "model": {**route, "default": "b2/own"}, "agent": {"system_prompt": "B2 PROMPT"},
        "platforms": {"telegram": {"enabled": True, "channel_overrides": {"888": {"model": "b2/override"}}}},
    }), encoding="utf-8")
    yield launch, b2
    set_multiplex_active(False)


def _turn(runner, profile, home, chat_id):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm", user_id=chat_id)
    source._identity = RoutingIdentity(profile, profile, home, home)
    with _profile_runtime_scope(home, prepared_secret_scope={}):
        model, _runtime = runner._resolve_session_agent_runtime(source=source, session_key=None)
        # The turn's real prompt builder, which owns the source.
        ctx = SimpleNamespace(source=source, context_prompt="", channel_prompt="")
        prompt = TurnRunner(runner, ctx)._combined_ephemeral_prompt()
    channel = runner._channel_override_for(source)
    return model, prompt, getattr(channel, "model", None)


def test_override_applies_only_in_the_profile_that_declares_it(homes):
    launch, b2 = homes
    config = load_gateway_config()
    config.multiplex_profiles = True
    set_multiplex_active(True)
    runner = GatewayRunner(config)

    assert _turn(runner, "default", launch, "777") == ("launch/override", "LAUNCH PROMPT", "launch/override")
    # Same DM, the other profile's bot: its own model and prompt, no foreign channel route.
    assert _turn(runner, "b2", b2, "777") == ("b2/own", "B2 PROMPT", None)
    # The routed profile's own override still applies, and never leaks back to the launch profile.
    assert _turn(runner, "b2", b2, "888") == ("b2/override", "B2 PROMPT", "b2/override")
    assert _turn(runner, "default", launch, "888")[0] == "launch/default"


def test_managed_worker_prompt_reads_the_serving_profiles_override(homes):
    """A managed worker's prompt is resolved owner-side; it must match the in-process turn's."""
    from contextlib import nullcontext

    from hermes_agent.gateway.session_worker_construct import _ephemeral_prompt

    _launch, b2 = homes
    raw = json.loads((b2 / "config.yaml").read_text(encoding="utf-8"))
    raw["platforms"]["local"] = {"channel_overrides": {"desk": {"system_prompt": "B2 LOCAL PROMPT"}}}
    (b2 / "config.yaml").write_text(json.dumps(raw), encoding="utf-8")
    config = load_gateway_config()
    config.multiplex_profiles = True
    set_multiplex_active(True)
    runner = GatewayRunner(config)
    source = SessionSource(platform=Platform.LOCAL, chat_id="desk", chat_type="dm", user_id="desk")
    source._identity = RoutingIdentity("b2", "b2", b2, b2)
    policy = SimpleNamespace(ignore_user_config=False, client_env_scope=nullcontext, skills_prompt=None)

    with _profile_runtime_scope(b2, prepared_secret_scope={}):
        assert _ephemeral_prompt(SimpleNamespace(runner=runner), policy, "desk", source) == "B2 LOCAL PROMPT"

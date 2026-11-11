"""``channel_overrides`` belong to the profile that serves the turn (#136198).

Chat ids repeat across profiles' bots: a Telegram DM's chat id is the user's id, so under
multiplexing the launch profile's override for that DM must not steer another profile's bot in the
same DM (its model, its system prompt, ``/model``'s channel route). Real runner, real loaders.
"""

from __future__ import annotations

import json

import pytest

from agent.secret_scope import set_multiplex_active
from gateway.config import Platform, load_gateway_config
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from gateway.session_identity import RoutingIdentity


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
        prompt = runner._get_system_prompt_for_channel(Platform.TELEGRAM, chat_id)
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

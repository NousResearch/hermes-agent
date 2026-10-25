"""Profile default precedes explicit fallbacks without retrying the session primary."""
from types import SimpleNamespace
import time

import pytest
import json


@pytest.mark.parametrize("enabled", [True, False])
def test_profile_default_chain_reloads_and_respects_cooldown(tmp_path, monkeypatch, enabled):
    from gateway.run import GatewayRunner
    from hermes_cli.fallback_config import get_fallback_chain

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    default = {"provider": "openai-codex", "default": "profile-model"}
    backup = {"provider": "openai-codex", "model": "backup-model"}
    config = {"model": default, "fallback_to_default": enabled,
              "fallback_providers": [backup, {"provider": default["provider"], "model": default["default"]}]}
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))
    runner = SimpleNamespace(_fallback_model=None)
    refresh = GatewayRunner._refresh_fallback_model.__get__(runner)
    chain = refresh()
    expected_default = {"provider": default["provider"], "model": default["default"]}
    assert chain == ([expected_default, backup] if enabled else [backup, expected_default])
    assert chain == get_fallback_chain(config)

    agent = SimpleNamespace(_fallback_chain=[], _fallback_activated=False,
                            _rate_limited_until=0, _fallback_index=0)
    GatewayRunner._apply_fallback_chain_to_agent(agent, chain)
    assert agent._fallback_chain == chain
    default["default"] = "changed-profile-model"
    path.write_text(json.dumps(config))
    updated = refresh()
    if enabled:
        assert updated[0]["model"] == default["default"]
    agent._fallback_activated = True
    agent._rate_limited_until = time.monotonic() + 60
    GatewayRunner._apply_fallback_chain_to_agent(agent, updated)
    assert agent._fallback_chain == chain
    agent._fallback_activated = False
    agent._rate_limited_until = 0
    GatewayRunner._apply_fallback_chain_to_agent(agent, updated)
    assert agent._fallback_chain == updated

    # A single runner serves independent profile homes, then returns to the first.
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import set_secret_scope, reset_secret_scope, set_multiplex_active

    other = tmp_path / "other"
    other.mkdir()
    other_config = {"fallback_to_default": True,
                    "model": {"provider": "custom", "default": "other-model",
                              "base_url": "https://other.example/v1", "api_key": "${PROFILE_KEY}"},
                    "fallback_providers": [backup]}
    (other / "config.yaml").write_text(json.dumps(other_config), encoding="utf-8")
    monkeypatch.setenv("PROFILE_KEY", "wrong-launch-profile-key")
    set_multiplex_active(True)
    try:
        for home, expected in ((tmp_path, updated), (other, None), (tmp_path, updated)):
            home_token = set_hermes_home_override(home)
            secret_token = set_secret_scope({"PROFILE_KEY": "test-scoped-key"})
            try:
                loaded = refresh()
                if expected is None:
                    assert loaded == [{"provider": "custom", "model": "other-model",
                                       "base_url": "https://other.example/v1",
                                       "api_key": "test-scoped-key"}, backup]
                else:
                    assert loaded == expected
            finally:
                reset_secret_scope(secret_token)
                reset_hermes_home_override(home_token)
    finally:
        set_multiplex_active(False)


@pytest.mark.parametrize("candidate,skip", [
    ({"provider": "openai-codex", "model": "session-model"}, True),
    ({"provider": "openai-codex", "model": "default-model"}, True),
    ({"provider": "openai-codex", "model": "backup-model"}, False),
    ({"provider": "openai-codex", "model": "session-model", "base_url": "https://other.example/v1"}, False),
])
def test_fallback_does_not_retry_original_session_route(candidate, skip):
    from agent.chat_completion_helpers import _should_skip_fallback_candidate, _fallback_entry_key

    primary = {"provider": "openai-codex", "model": "session-model", "base_url": "https://original.example/v1"}
    agent = SimpleNamespace(provider="openai-codex", model="default-model",
                            base_url="https://original.example/v1", _primary_runtime=primary,
                            _fallback_activated=True)
    assert _should_skip_fallback_candidate(agent, candidate, _fallback_entry_key(candidate),
                                           candidate["provider"], candidate["model"], set()) is skip
    assert agent._primary_runtime == primary

"""A completed turn's provider identity can launch the next Desktop chat (regression for #143660)."""
from copy import deepcopy
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("requested", ["relay", "custom:relay"])
def test_completed_turn_provider_resolves_the_next_chat(tmp_path, monkeypatch, requested):
    from gateway import session_local
    from gateway.session_local_inventory import publish_settled_info
    from gateway.session_policy import _apply_launch_overrides
    from hermes_cli.config import atomic_config_write
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.runtime_provider import frozen_runtime_config, resolve_runtime_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    atomic_config_write(config_path, {
        "model": {"default": "fleet-model", "provider": "relay", "base_url": "http://127.0.0.1:19200/v1"},
        "providers": {"relay": {"base_url": "http://127.0.0.1:19200/v1", "api_key": "test-key"}},
    })
    config = load_user_config_effective(config_path)
    with frozen_runtime_config(config):
        runtime = resolve_runtime_provider(requested=requested, target_model="fleet-model")
    agent = SimpleNamespace(**{k: runtime[k] for k in ("provider", "requested_provider", "base_url")},
                            model="fleet-model", tools=None, _cached_system_prompt="", context_compressor=None)
    published = []
    turn = SimpleNamespace(_ctx=SimpleNamespace(inbound_message_id="adm"),
                           _approval_owner=(SimpleNamespace(profile_id=str(tmp_path)), "s", 1),
                           _publish_execution=lambda kind, info: published.append((kind, info)))
    monkeypatch.setattr(session_local, "local_session_info", lambda *_: {"model": "fleet-model"})
    with frozen_runtime_config(config):
        publish_settled_info(turn, agent)
    [(kind, info)] = published
    assert kind == "session.info"

    # Desktop carries this pair into session.create without an explicit endpoint or credential.
    following = deepcopy(config)
    _apply_launch_overrides({"provider": info["provider"], "model": info["model"]}, following)
    with frozen_runtime_config(following):
        next_runtime = resolve_runtime_provider(requested=following["model"]["provider"],
                                                target_model=info["model"])
    assert (next_runtime["base_url"], next_runtime["api_key"]) == (runtime["base_url"], runtime["api_key"])
    assert "api_key" not in info and "test-key" not in str(info)

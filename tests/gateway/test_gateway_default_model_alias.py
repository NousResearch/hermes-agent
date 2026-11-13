"""Gateway default-session aliases resolve the whole route before the first platform turn."""

from types import SimpleNamespace

import pytest

from gateway import run
from gateway.run_turn_prepare import GatewayTurnPrepareMixin


def test_platform_default_alias_uses_its_model_endpoint_and_key(monkeypatch):
    """A platform turn must not send the raw alias to model.provider's endpoint."""
    config = {
        "model": {"default": "myalias", "provider": "openrouter"},
        "model_aliases": {
            "myalias": {
                "model": "some-vendor/some-model-id",
                "provider": "custom",
                "base_url": "http://127.0.0.1:9999/v1",
                "key_env": "DEMO_API_KEY",
            },
        },
    }
    monkeypatch.setenv("DEMO_API_KEY", "sk-alias-host")
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr("hermes_cli.runtime_provider._get_model_config", lambda: config["model"])
    requests = []

    def resolve(config_snapshot, **kwargs):
        assert config_snapshot == config
        requests.append(kwargs)
        return {
            "provider": kwargs.get("requested") or "openrouter",
            "base_url": kwargs.get("explicit_base_url") or "https://openrouter.ai/api/v1",
            "api_key": kwargs.get("explicit_api_key") or "sk-default-provider",
            "api_mode": "chat_completions",
        }, None

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_with_fallback", resolve)
    state = SimpleNamespace(conversation=SimpleNamespace(last_resolved_model=""))
    runner = SimpleNamespace(
        _resolve_session_key_or_none=lambda source, key: None,
        _peek_session_state=lambda key: state,
        _session_state=lambda key: state,
    )

    model, runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime(
        runner, user_config=config)
    assert (model, runtime["provider"], runtime["base_url"], runtime["api_key"]) == (
        "some-vendor/some-model-id", "custom", "http://127.0.0.1:9999/v1", "sk-alias-host")
    assert requests == [{
        "requested": "custom",
        "target_model": "some-vendor/some-model-id",
        "explicit_base_url": "http://127.0.0.1:9999/v1",
        "explicit_api_key": "sk-alias-host",
    }]


_ALIAS_CONFIG = {
    "model": {"default": "myalias", "provider": "openrouter"},
    "model_aliases": {"myalias": {"model": "some-vendor/some-model-id", "provider": "custom",
                                  "base_url": "http://127.0.0.1:9999/v1", "key_env": "DEMO_API_KEY"}},
}


def _cron_route():
    from cron.scheduler import _CronJobConfig, _resolve_job_runtime
    jc = _CronJobConfig(cfg=_ALIAS_CONFIG, model="myalias", model_cfg=_ALIAS_CONFIG["model"], cron_default_provider="")
    runtime, model = _resolve_job_runtime({}, "job", jc)
    return model, runtime


def _tui_route():
    from tui_gateway import server
    return server._resolve_agent_model_runtime(None, None)


def _feishu_route():
    from plugins.platforms.feishu.feishu_comment import _resolve_model_and_runtime
    return _resolve_model_and_runtime()


@pytest.mark.parametrize("surface", [_cron_route, _tui_route, _feishu_route], ids=["cron", "tui", "feishu"])
def test_every_config_default_surface_expands_the_alias(monkeypatch, surface):
    """Cron, the TUI agent factory and Feishu comments run the config default too: same whole route."""
    from hermes_cli import runtime_provider as rp
    from tui_gateway import server
    for name in ("HERMES_MODEL", "HERMES_INFERENCE_MODEL", "HERMES_TUI_PROVIDER", "HERMES_INFERENCE_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEMO_API_KEY", "sk-alias-host")
    monkeypatch.setattr(run, "_load_gateway_config", lambda: _ALIAS_CONFIG)
    monkeypatch.setattr(server, "_load_cfg", lambda: _ALIAS_CONFIG)
    monkeypatch.setattr(rp, "load_config", lambda: _ALIAS_CONFIG)
    monkeypatch.setattr(rp, "_get_model_config", lambda: _ALIAS_CONFIG["model"])
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _ALIAS_CONFIG)

    model, runtime = surface()
    assert (model, runtime["base_url"], runtime["api_key"]) == (
        "some-vendor/some-model-id", "http://127.0.0.1:9999/v1", "sk-alias-host")


@pytest.mark.parametrize("fallback", [True, False], ids=["turn", "session-control"])
def test_a_canonical_session_on_a_default_alias_runs_the_alias_route(tmp_path, monkeypatch, fallback):
    """``session.create`` with no model (``hermes --tui``, Desktop) freezes ``model.default`` = the alias into its
    policy; the policy route (in-process turn and managed worker) must expand it like every other surface."""
    from dataclasses import replace
    from gateway.session_policy import build_policy
    from gateway.session_worker_construct import model_route_runtime
    monkeypatch.setenv("DEMO_API_KEY", "sk-alias-host")
    policy = build_policy({"cwd": str(tmp_path), "toolsets": []}, _ALIAS_CONFIG, profile_terminal=False)
    policy = replace(policy, model=run._resolve_gateway_model(policy.config()))  # session_local's no-model step
    frozen, model = policy.effective()
    assert model == "myalias"
    model, runtime, _, aliased = model_route_runtime(policy, frozen, model, None, fallback=fallback)
    assert (model, runtime["base_url"], runtime["api_key"], aliased) == (
        "some-vendor/some-model-id", "http://127.0.0.1:9999/v1", "sk-alias-host", True)


@pytest.mark.parametrize("pick, key", [({"provider": "openrouter"}, None), ({}, "sk-launch")],
                         ids=["explicit-provider", "launch-key"])
def test_an_explicit_launch_route_is_never_replaced_by_the_default_alias(tmp_path, monkeypatch, pick, key):
    """A session created with its own provider or key keeps that route; only an unrouted default expands."""
    from dataclasses import replace
    from gateway.session_policy import build_policy
    from gateway.session_worker_construct import policy_alias_request
    monkeypatch.setenv("DEMO_API_KEY", "sk-alias-host")
    policy = build_policy({"cwd": str(tmp_path), "toolsets": [], **pick}, _ALIAS_CONFIG, profile_terminal=False)
    policy = replace(policy, model="myalias")
    frozen, model = policy.effective()
    assert policy_alias_request(policy, frozen, model, key) is None


def test_a_same_name_alias_still_takes_its_own_worker_route(tmp_path, monkeypatch):
    """An alias whose model id equals its name still routes the managed worker to the alias endpoint and key."""
    from dataclasses import replace
    from gateway.session_policy import build_policy
    from gateway.session_worker_construct import worker_route
    config = {"model": {"default": "gpt-4.1", "provider": "openrouter"},
              "model_aliases": {"gpt-4.1": {"model": "gpt-4.1", "provider": "custom",
                                            "base_url": "http://127.0.0.1:9999/v1", "key_env": "DEMO_API_KEY"}}}
    monkeypatch.setenv("DEMO_API_KEY", "sk-alias-host")
    policy = replace(build_policy({"cwd": str(tmp_path), "toolsets": []}, config, profile_terminal=False), model="gpt-4.1")
    route = worker_route(policy, {"api_key": None})
    assert (route["model"], route["base_url"], route["api_key"]) == ("gpt-4.1", "http://127.0.0.1:9999/v1", "sk-alias-host")


def test_the_fallback_baseline_of_a_frozen_alias_session_is_its_frozen_backing_model(tmp_path, monkeypatch):
    """The post-turn drift check reads the alias from the session's frozen config, not the live profile."""
    from dataclasses import replace
    from gateway.session_policy import build_policy, turn_model
    monkeypatch.setenv("DEMO_API_KEY", "sk-alias-host")
    policy = replace(build_policy({"cwd": str(tmp_path), "toolsets": []}, _ALIAS_CONFIG, profile_terminal=False),
                     model="myalias")
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"model": {"default": "myalias"}})  # alias removed live
    assert turn_model(policy, None) == "some-vendor/some-model-id"

"""Behavioral coverage for static named model-route expansion."""
from __future__ import annotations

import copy
import pytest
import yaml

from hermes_cli.model_presets import ModelPresetError, expand_model_presets


def test_documented_preset_example_expands():
    from pathlib import Path
    doc = Path(__file__).resolve().parents[2] / "website/docs/user-guide/configuring-models.md"
    block = doc.read_text().split("## Reusing a named route", 1)[1].split("```yaml", 1)[1].split("```", 1)[0]
    expanded = expand_model_presets(yaml.safe_load(block))
    assert expanded["delegation"]["fallback_providers"][0]["provider"] == "openai"


def routes():
    return {"model_presets": {
        "primary": {"provider": "provider-a", "model": "model-a", "reasoning_effort": "high", "fallbacks": [{"provider": "provider-b", "model": "model-b", "reasoning_effort": "low"}]},
        "fast": {"provider": "provider-c", "model": "model-c", "reasoning_effort": "minimal"},
    }}


def test_expands_all_runtime_consumers_without_mutating_authored_routes():
    config = {**routes(), "model": {"model_preset": "fast"}, "delegation": {"model_preset": "primary", "max_concurrent_children": 3}, "auxiliary": {"compression": {"model_preset": "primary", "timeout": 20}}, "fallback_providers": [{"model_preset": "fast"}], "moa": {"presets": {"review": {"reference_models": [{"model_preset": "fast"}], "aggregator": {"model_preset": "fast"}}}}}
    authored = copy.deepcopy(config)
    expanded = expand_model_presets(config)
    assert expanded["model"] == {"provider": "provider-c", "default": "model-c"}
    assert expanded["agent"]["reasoning_effort"] == "minimal"
    assert expanded["fallback_providers"] == [{"provider": "provider-c", "model": "model-c", "reasoning_effort": "minimal"}]
    assert expanded["delegation"]["fallback_providers"][0]["provider"] == "provider-b"
    assert expanded["auxiliary"]["compression"]["fallback_chain"][0]["model"] == "model-b"
    assert expanded["moa"]["presets"]["review"]["aggregator"]["reasoning_effort"] == "minimal"
    assert config == authored


@pytest.mark.parametrize("config, fragment", [
    ({**routes(), "auxiliary": {"vision": {"model_preset": "missing"}}}, "auxiliary.vision: unknown preset 'missing'"),
    ({**routes(), "delegation": {"model_preset": "fast", "provider": "x"}}, "delegation: 'model_preset: fast' cannot be combined"),
    ({**routes(), "moa": {"aggregator": {"model_preset": "primary"}}}, "moa.aggregator: a MoA slot cannot reference a preset that declares fallbacks"),
])
def test_invalid_references_are_actionable(config, fragment):
    with pytest.raises(ModelPresetError, match=fragment):
        expand_model_presets(config)


def test_real_config_loader_expands_and_save_keeps_authored_reference(tmp_path, monkeypatch):
    from hermes_cli import config as config_mod
    home = tmp_path / "hermes-home"; home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    raw = {**routes(), "model": {"model_preset": "fast"}, "auxiliary": {"vision": {"model_preset": "fast"}}}
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    config_mod._RAW_CONFIG_CACHE.clear(); config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    loaded = config_mod.load_config()
    assert (loaded["model"]["provider"], loaded["model"]["default"]) == ("provider-c", "model-c")
    assert loaded["auxiliary"]["vision"]["reasoning_effort"] == "minimal"
    loaded["display"]["show_thinking"] = False
    config_mod.save_config(loaded)
    saved = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert saved["model"] == {"model_preset": "fast"}
    assert saved["auxiliary"]["vision"]["model_preset"] == "fast"


def test_save_without_default_stripping_reloads_preset_and_removes_injected_route_defaults(tmp_path, monkeypatch):
    from hermes_cli import config as config_mod
    home = tmp_path / "hermes-home"; home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    raw = {**routes(), "model": {"model_preset": "fast", "context_length": 12345},
           "auxiliary": {"vision": {"model_preset": "fast"}}}
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    config_mod._RAW_CONFIG_CACHE.clear(); config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    loaded = config_mod.load_config()
    # Simulate defaults injected by provider/auxiliary normalization before an all-default save.
    loaded["model"].update({"base_url": "", "api_key": "", "api_mode": ""})
    loaded["auxiliary"]["vision"].update({"base_url": "", "api_key": "", "api_mode": ""})
    config_mod.save_config(loaded, strip_defaults=False)
    saved = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert saved["model"] == {"model_preset": "fast", "context_length": 12345}
    assert saved["auxiliary"]["vision"]["model_preset"] == "fast"
    assert not {"base_url", "api_key", "api_mode"} & set(saved["auxiliary"]["vision"])
    config_mod._RAW_CONFIG_CACHE.clear(); config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    assert config_mod.load_config()["model"]["default"] == "model-c"


def test_main_reference_preserves_context_and_authored_global_route_settings():
    from hermes_cli.model_presets import preserve_model_preset_references
    authored = {**routes(), "model_presets": {**routes()["model_presets"], "plain": {"provider": "plain-p", "model": "plain-m"}},
                "model": {"model_preset": "plain", "context_length": 9000},
                "agent": {"reasoning_effort": "low"},
                "fallback_providers": [{"provider": "global", "model": "global-model"}]}
    # A separately-authored global agent/fallback is not expansion output and must survive save.
    actual = {**copy.deepcopy(authored), "model": {"provider": "plain-p", "default": "plain-m", "context_length": 12000}}
    restored = preserve_model_preset_references(actual, authored)
    assert restored["model"] == {"model_preset": "plain", "context_length": 12000}
    assert restored["agent"]["reasoning_effort"] == "low"
    assert restored["fallback_providers"] == authored["fallback_providers"]


def test_edited_main_route_flattens_instead_of_restoring_a_conflicting_reference():
    from hermes_cli.model_presets import preserve_model_preset_references
    authored = {**routes(), "model": {"model_preset": "fast", "context_length": 9000}}
    actual = {**copy.deepcopy(authored), "model": {"provider": "edited-provider", "default": "edited-model", "context_length": 12000}}
    restored = preserve_model_preset_references(actual, authored)
    assert restored["model"] == actual["model"]


@pytest.mark.parametrize("field", ["base_url", "api_base", "api_key", "api", "key_env", "api_key_env", "api_mode", "transport"])
def test_main_reference_rejects_provider_owned_fields(field):
    with pytest.raises(ModelPresetError, match="cannot be combined"):
        expand_model_presets({**routes(), "model": {"model_preset": "fast", field: "value"}})


def test_gateway_and_delegation_loaders_do_not_swallow_invalid_preset_errors(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gateway.run import GatewayRunner
    from tools import delegate_tool_config

    err = ModelPresetError("invalid preset")
    monkeypatch.setattr("gateway.run._load_gateway_runtime_config", lambda: (_ for _ in ()).throw(err))
    with pytest.raises(ModelPresetError, match="invalid preset"):
        GatewayRunner._load_fallback_model()

    monkeypatch.setattr(delegate_tool_config.os, "environ", {}, raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: (_ for _ in ()).throw(err))
    with pytest.raises(ModelPresetError, match="invalid preset"):
        delegate_tool_config._load_config()

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"model": {"model_preset": "missing"}}), encoding="utf-8")
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner = SimpleNamespace(_fallback_model=None)
    with pytest.raises(ModelPresetError, match="unknown preset"):
        GatewayRunner._refresh_fallback_model.__get__(runner)()


def test_auxiliary_fallback_reasoning_uses_route_or_retains_primary(monkeypatch):
    from agent.auxiliary_client import _fallback_reasoning_config
    primary = {"enabled": True, "effort": "high"}
    monkeypatch.setattr("agent.auxiliary_client._fallback_chain_entry", lambda *_: {"reasoning_effort": "low"})
    assert _fallback_reasoning_config("compression", "fallback_chain[0](x)", primary) == {"enabled": True, "effort": "low"}
    monkeypatch.setattr("agent.auxiliary_client._fallback_chain_entry", lambda *_: {})
    assert _fallback_reasoning_config("compression", "fallback_chain[0](x)", primary) == primary


def test_gateway_and_cron_raw_loaders_expand_temp_home_config(tmp_path, monkeypatch):
    from cron import scheduler
    from gateway import run as gateway_run
    home = tmp_path / "hermes-home"; home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({**routes(), "model": {"model_preset": "fast"}}), encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: yaml.safe_load((home / "config.yaml").read_text()))
    assert gateway_run._resolve_gateway_model(gateway_run._load_gateway_runtime_config()) == "model-c"
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)
    job_cfg = scheduler._load_cron_job_config({"id": "job-1"}, "job-1", "preset job")
    assert (job_cfg.model, job_cfg.model_cfg["provider"]) == ("model-c", "provider-c")


def test_missing_definitions_and_unsupported_route_keys_fail_at_the_shared_boundary():
    with pytest.raises(ModelPresetError, match="model: unknown preset 'missing'"):
        expand_model_presets({"model": {"model_preset": "missing"}})
    with pytest.raises(ModelPresetError, match="unsupported field\\(s\\): base_url"):
        expand_model_presets({"model_presets": {"bad": {"provider": "x", "model": "y", "base_url": "https://x"}}})
    with pytest.raises(ModelPresetError, match=r"fallbacks' may only be \[\]"):
        expand_model_presets({**routes(), "delegation": {"model_preset": "fast", "fallbacks": [{"provider": "x", "model": "y"}]}})


def test_explicit_empty_fallbacks_and_save_preserve_unrelated_delegation_moa_and_fallback_values():
    authored = {
        **routes(),
        "delegation": {"model_preset": "primary", "fallbacks": [], "max_concurrent_children": 2},
        "fallback_providers": [{"model_preset": "fast", "label": "cheap"}],
        "moa": {"aggregator": {"model_preset": "fast", "enabled": False}},
    }
    expanded = expand_model_presets(authored)
    main_opt_out = expand_model_presets({**routes(), "model": {"model_preset": "primary", "fallbacks": []}})
    aux_opt_out = expand_model_presets({**routes(), "auxiliary": {"vision": {"model_preset": "primary", "fallbacks": []}}})
    assert main_opt_out["fallback_providers"] == []
    assert aux_opt_out["auxiliary"]["vision"]["fallback_chain"] == []
    assert expanded["delegation"]["fallback_providers"] == []
    actual = copy.deepcopy(expanded)
    actual["delegation"]["max_concurrent_children"] = 7
    actual["fallback_providers"][0]["label"] = "re-priced"
    actual["moa"]["aggregator"]["enabled"] = True
    from hermes_cli.model_presets import preserve_model_preset_references
    restored = preserve_model_preset_references(actual, authored)
    assert restored["delegation"] == {"model_preset": "primary", "fallbacks": [], "max_concurrent_children": 7}
    assert restored["fallback_providers"] == [{"model_preset": "fast", "label": "re-priced"}]
    assert restored["moa"]["aggregator"] == {"model_preset": "fast", "enabled": True}


def test_delegation_and_fallback_consumers_receive_preset_reasoning_and_chain(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agent.chat_completion_helpers import _reresolve_fallback_reasoning_config
    from tools.delegate_tool_config import _resolve_child_runtime
    cfg = expand_model_presets({**routes(), "delegation": {"model_preset": "primary"}})
    parent = SimpleNamespace(model="parent", provider="parent-provider", base_url="", api_mode="chat_completions", acp_command=None, acp_args=[], reasoning_config={"enabled": False}, _fallback_chain=[], request_overrides=None, capabilities=None)
    child = _resolve_child_runtime(parent, cfg["delegation"], "parent-key", model=None, override_provider=None, override_base_url=None, override_api_key=None, override_api_mode=None, override_acp_command=None, override_acp_args=None)
    assert child["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert child["fallback_model"] == [{"provider": "provider-b", "model": "model-b", "reasoning_effort": "low"}]
    fallback_agent = SimpleNamespace(model="model-b", reasoning_config=None)
    _reresolve_fallback_reasoning_config(fallback_agent, child["fallback_model"][0])
    assert fallback_agent.reasoning_config == {"enabled": True, "effort": "low"}


@pytest.mark.parametrize("strip_defaults", [True, False])
@pytest.mark.parametrize("edit", ["unchanged", "reasoning", "fallbacks"])
def test_main_preset_roundtrip_and_deliberate_projected_edits(tmp_path, monkeypatch, strip_defaults, edit):
    from hermes_cli import config as c
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw = {**routes(), "model": {"model_preset": "primary"}}
    if edit == "fallbacks":
        raw["model"]["fallbacks"] = []
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    c._RAW_CONFIG_CACHE.clear(); c._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    loaded = c.load_config()
    if edit == "reasoning":
        loaded["agent"]["reasoning_effort"] = "low"
    elif edit == "fallbacks":
        loaded["fallback_providers"] = [{"provider": "edited", "model": "edited"}]
    c.save_config(loaded, strip_defaults=strip_defaults)
    saved = yaml.safe_load(path.read_text())
    assert ("model_preset" in saved["model"]) == (edit == "unchanged")
    c._RAW_CONFIG_CACHE.clear(); c._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    reloaded = c.load_config()
    assert reloaded["agent"]["reasoning_effort"] == ("low" if edit == "reasoning" else "high")
    assert reloaded["fallback_providers"] == loaded["fallback_providers"]


@pytest.mark.parametrize("site,field,value", [("delegation", "reasoning_effort", "low"), ("auxiliary", "api_key", "nonsecret-fixture")])
def test_new_route_fields_are_saved_inline_not_as_conflicting_reference(tmp_path, monkeypatch, site, field, value):
    from hermes_cli import config as c
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw: dict = {"model_presets": {"plain": {"provider": "openrouter", "model": "test/model"}}}
    raw[site] = {"model_preset": "plain"} if site == "delegation" else {"vision": {"model_preset": "plain"}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    c._RAW_CONFIG_CACHE.clear(); c._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    loaded = c.load_config()
    target = loaded[site] if site == "delegation" else loaded[site]["vision"]
    target[field] = value
    c.save_config(loaded)
    saved = yaml.safe_load(path.read_text())
    target = saved[site] if site == "delegation" else saved[site]["vision"]
    assert "model_preset" not in target
    c._RAW_CONFIG_CACHE.clear(); c._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    reloaded = c.load_config()
    target = reloaded[site] if site == "delegation" else reloaded[site]["vision"]
    assert target[field] == value


def test_main_fallback_preset_reasoning_reaches_auxiliary_wire_request(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import config as c
    from agent import auxiliary_client as a
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw = {"model_presets": {"fallback": {"provider": "openrouter", "model": "test/fallback", "reasoning_effort": "low"}},
           "model": {"provider": "openrouter", "default": "test/main"}, "fallback_providers": [{"model_preset": "fallback"}]}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw))
    c._RAW_CONFIG_CACHE.clear(); c._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    client = SimpleNamespace(base_url="https://openrouter.ai/api/v1", _hermes_fallback_destination=a._FallbackDestination("openrouter", "https://openrouter.ai/api/v1", "chat_completions", "test/fallback"))
    monkeypatch.setattr(a, "_resolve_fallback_entry", lambda entry: (client, entry["model"]))
    monkeypatch.setattr(a, "_is_provider_unhealthy", lambda *args: False)
    monkeypatch.setattr(a, "_context_too_small", lambda *args, **kwargs: None)
    fb_client, model, label = a._try_main_fallback_chain("title_generation", "openrouter", failed_model="test/main")
    _, kwargs, _ = a._plan_fallback_candidate(fb_client, model, label, task="title_generation", effective_timeout=30,
        apply_fast_lane=False, messages=[{"role": "user", "content": "test"}], tools=None, temperature=None, max_tokens=50,
        effective_extra_body={}, reasoning_config={"enabled": True, "effort": "high"})
    assert kwargs["extra_body"]["reasoning"]["effort"] == "low"

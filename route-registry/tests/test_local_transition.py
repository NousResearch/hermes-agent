from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).parents[1]
ROOT = HERE.parent
import sys
sys.path.insert(0, str(HERE / "route_registry"))
sys.path.insert(0, str(ROOT))
from local_transition import (LocalTransitionError, apply_local_plan, build_local_plan,
                              rollback_local_plan, transform_config)


@pytest.fixture()
def spec():
    return yaml.safe_load((HERE / "registry/route-slots.yaml").read_text())["local_transition"]


def test_red_unknown_active_old_local_model_blocks(spec):
    with pytest.raises(LocalTransitionError, match="unknown model"):
        transform_config({"fallback_providers": [{"provider": "custom:turbohaul-local", "model": "mystery"}]}, spec)


def test_green_all_supported_shapes_and_embedded_json(spec):
    old_decl = {"name": "turbohaul-local", "base_url": spec["old"]["base_url"],
                "model": "qwen3.8-27b", "models": ["carwin-moe", "qwen3.8-27b", "unused"]}
    doc = {
        "model": {"provider": spec["old"]["provider"], "default": "qwen3.8-27b", "keep": 1},
        "fallback_providers": [{"provider": spec["old"]["provider"], "model": "qwen3.8-27b", "base_url": spec["old"]["base_url"]}],
        "auxiliary": {"compression": {"timeout": 17, "fallback_chain": [{"provider": spec["old"]["provider"], "model": "carwin-moe", "base_url": spec["old"]["base_url"]}]}},
        "moa": {"presets": {"x": {"reference_models": [{"provider": spec["old"]["provider"], "model": "carwin-moe"}], "aggregator": {"provider": spec["old"]["provider"], "model": "qwen3.8-27b"}}}},
        "council": {"chairman": json.dumps({"fallback": {"0": {"provider": spec["old"]["provider"], "model": "qwen3.8-27b"}}, "literal": "qwen3.8-27b"})},
        "custom_providers": [old_decl], "cloud": {"timeout": 19},
    }
    out, changes = transform_config(doc, spec)
    assert out["model"]["default"] == "active:main"
    assert out["fallback_providers"][0]["timeout"] == 1800
    assert out["auxiliary"]["compression"]["fallback_chain"][0]["model"] == "active:aux"
    assert out["auxiliary"]["compression"]["fallback_chain"][0]["timeout"] == 1800
    assert json.loads(out["council"]["chairman"])["literal"] == "qwen3.8-27b"
    assert out["custom_providers"][0] == old_decl
    assert out["custom_providers"][1]["models"] == ["active:main", "active:aux"]
    assert out["providers"]["custom:turbofit-local"]["request_timeout_seconds"] == 1800
    assert out["cloud"] == doc["cloud"]
    assert changes


def test_plan_apply_rollback_cas_and_raw_backup(tmp_path, spec):
    root = tmp_path / "tree"; root.mkdir(); cfg = root / "config.yaml"
    raw = b"fallback_providers:\n- provider: custom:turbohaul-local\n  model: qwen3.8-27b\n  base_url: http://127.0.0.1:11410/v1\n"
    cfg.write_bytes(raw)
    plan = build_local_plan(root, spec); backup = tmp_path / "backup"
    apply_local_plan(plan, spec, backup)
    assert b"active:main" in cfg.read_bytes(); assert (backup / "config.yaml").read_bytes() == raw
    rollback_local_plan(plan, backup); assert cfg.read_bytes() == raw
    cfg.write_bytes(raw + b"# stale\n")
    with pytest.raises(LocalTransitionError, match="stale input"):
        apply_local_plan(plan, spec, tmp_path / "backup2")


def test_actual_timeout_resolution_seams(monkeypatch, spec):
    from agent import auxiliary_client
    staged = {"timeout": 30, "fallback_chain": [{"provider": "cloud", "model": "x"},
              {"provider": spec["new"]["provider"], "model": "active:aux", "timeout": 1800}]}
    monkeypatch.setattr(auxiliary_client, "_get_auxiliary_task_config", lambda task: staged)
    assert auxiliary_client._effective_aux_timeout("title_generation", None) == 30
    assert auxiliary_client._effective_aux_timeout("compression", None) == 300
    assert auxiliary_client._fallback_entry_timeout("compression", "fallback_chain[0](cloud)") is None
    assert auxiliary_client._fallback_entry_timeout("compression", "fallback_chain[1](custom:turbofit-local)") == 1800

    import hermes_cli.config as config
    import hermes_cli.timeouts as timeouts
    cfg = {"providers": {spec["new"]["provider"]: {"request_timeout_seconds": 1800},
                         "cloud": {"request_timeout_seconds": 41}}}
    monkeypatch.setattr(config, "load_config_readonly", lambda: cfg)
    assert timeouts.get_provider_request_timeout(spec["new"]["provider"], "active:main") == 1800
    assert timeouts.get_provider_request_timeout("cloud", "x") == 41

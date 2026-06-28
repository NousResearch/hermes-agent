"""Regression: auxiliary client must resolve a MoA preset to its aggregator.

Bug: when the active main model is a MoA preset (provider="moa",
model="<preset-name>"), auxiliary side-tasks (title generation, compression,
vision) passed the *preset name* straight through as a model ID, producing
HTTP 400 "<preset> is not a valid model ID".

Fix lives in agent.auxiliary_client._resolve_auto: a "moa" main provider is
rewritten to the preset's aggregator (provider, model) before Step-1 builds a
client. These tests exercise the real _resolve_auto with resolve_provider_client
stubbed, asserting the (provider, model) actually handed to the router.
"""

import pytest

import agent.auxiliary_client as ac


@pytest.fixture(autouse=True)
def _clear_runtime():
    ac.clear_runtime_main()
    yield
    ac.clear_runtime_main()


def _capture_resolver(monkeypatch):
    """Stub resolve_provider_client; record the (provider, model) it receives."""
    calls = []

    def fake_resolve(provider, model, **kwargs):
        calls.append({"provider": provider, "model": model, "kwargs": kwargs})
        # Return a truthy sentinel client so Step-1 "wins" and returns.
        return object(), model

    monkeypatch.setattr(ac, "resolve_provider_client", fake_resolve)
    # Neutralize the unhealthy-cache short-circuit so Step-1 always runs.
    monkeypatch.setattr(ac, "_is_provider_unhealthy", lambda *_a, **_k: False)
    return calls


def _fake_config(monkeypatch, cfg):
    # Patch load_config in BOTH the config module and the auxiliary_client's
    # late-imported reference path. _resolve_auto does
    # `from hermes_cli.config import load_config` at call time, so patching the
    # source attribute is what matters.
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)


def test_moa_preset_resolves_to_aggregator(monkeypatch):
    calls = _capture_resolver(monkeypatch)
    cfg = {
        "model": {"provider": "moa", "default": "local-moa-minimal"},
        "moa": {
            "default_preset": "local-moa-minimal",
            "presets": {
                "local-moa-minimal": {
                    "reference_models": [
                        {"provider": "custom:ollama-local", "model": "qwen2.5:7b-instruct"}
                    ],
                    "aggregator": {
                        "provider": "custom:qwen3.6-35b-nvfp4",
                        "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
                    },
                }
            },
        },
    }
    _fake_config(monkeypatch, cfg)
    ac.set_runtime_main("moa", "local-moa-minimal")

    client, model = ac._resolve_auto(task="title_generation")

    assert client is not None
    # The preset name must NOT be used as a model id.
    assert model != "local-moa-minimal"
    assert model == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert calls, "resolve_provider_client was never called"
    assert calls[0]["provider"] == "custom:qwen3.6-35b-nvfp4"
    assert calls[0]["model"] == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    # No call should ever carry the bare preset name as the model id.
    assert all(c["model"] != "local-moa-minimal" for c in calls)


def test_moa_preset_without_aggregator_skips_step1(monkeypatch):
    """A preset whose aggregator slot is empty must fall through, not 400."""
    calls = _capture_resolver(monkeypatch)
    # Force the fallback chain to yield nothing so we can assert Step-1 was
    # skipped (no resolver call carrying the preset name).
    monkeypatch.setattr(ac, "_try_configured_fallback_chain",
                        lambda *_a, **_k: (None, None, None))
    monkeypatch.setattr(ac, "_try_main_fallback_chain",
                        lambda *_a, **_k: (None, None, None))
    monkeypatch.setattr(ac, "_get_provider_chain", lambda: [])

    cfg = {
        "model": {"provider": "moa", "default": "broken-preset"},
        "moa": {
            "default_preset": "broken-preset",
            "presets": {
                "broken-preset": {
                    "reference_models": [
                        {"provider": "custom:ollama-local", "model": "qwen2.5:7b-instruct"}
                    ],
                    # aggregator missing → normalize fills DEFAULT_MOA_AGGREGATOR,
                    # so to truly test "no aggregator" we blank it post-resolve.
                    "aggregator": {"provider": "", "model": ""},
                }
            },
        },
    }
    _fake_config(monkeypatch, cfg)
    ac.set_runtime_main("moa", "broken-preset")

    client, model = ac._resolve_auto(task="title_generation")

    # Either way, the bare preset name must never reach the router as a model.
    assert all(c["model"] != "broken-preset" for c in calls)

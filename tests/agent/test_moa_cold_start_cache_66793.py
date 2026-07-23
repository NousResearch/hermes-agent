"""Regression tests for MoA cold-start caching (#66793).

The preset switch used to re-parse + re-validate the full config and
re-resolve every slot's provider runtime on EACH create() call (once
per tool-loop iteration), serially before the parallel fan-out could
begin — 5-30s of "frozen" latency on complex presets.
Both the resolved preset and each (provider, model) runtime are now
cached for the process lifetime (config is immutable per turn), so the
underlying ``resolve_runtime_provider`` (real provider-catalog I/O)
runs once per distinct slot, not once per create() iteration.
"""

import types

import pytest


def _make_preset_config() -> dict:
    return {
        "moa": {
            "default_preset": "demo",
            "presets": {
                "demo": {
                    "enabled": True,
                    "aggregator": {"provider": "openai", "model": "gpt-5"},
                    "reference_models": [
                        {"provider": "deepseek", "model": "deepseek-v4"},
                        {"provider": "minimax", "model": "minimax-m3"},
                    ],
                }
            },
        }
    }


def test_preset_resolution_is_cached_across_create_calls(monkeypatch):
    """resolve_moa_preset must run once per (config-mtime, preset_name),
    not on every create() iteration."""
    import agent.moa_loop as moa

    moa._preset_cache.clear()

    calls = {"n": 0}
    import hermes_cli.moa_config as moa_cfg_mod
    real_resolve = moa_cfg_mod.resolve_moa_preset

    def counting_resolve(config, name=None):
        calls["n"] += 1
        return real_resolve(config, name)

    monkeypatch.setattr(moa_cfg_mod, "resolve_moa_preset", counting_resolve)
    import hermes_cli.config as cfg_mod
    monkeypatch.setattr(
        cfg_mod, "load_config",
        lambda: types.SimpleNamespace(
            mtime=1234,
            **{"get": lambda k, d=None: _make_preset_config().get(k, d)},
        ),
    )
    monkeypatch.setattr(moa, "call_llm", lambda **k: _fake_response())

    cc = moa.MoAChatCompletions("demo")
    for _ in range(3):
        cc.create(messages=[{"role": "user", "content": "hi"}])

    # One preset resolution for the whole turn (not 3).
    assert calls["n"] == 1, f"expected 1 preset resolution, got {calls['n']}"


def test_slot_runtime_is_cached_across_create_calls(monkeypatch):
    """resolve_runtime_provider (real I/O) must run once per
    (provider, model) across all create() iterations, not per call."""
    import agent.moa_loop as moa

    moa._runtime_cache.clear()
    moa._preset_cache.clear()

    calls = {"n": 0}

    def counting_resolve(*a, **k):
        calls["n"] += 1
        return {"base_url": None, "api_key": None, "api_mode": None}

    import hermes_cli.runtime_provider as rt_mod
    monkeypatch.setattr(rt_mod, "resolve_runtime_provider", counting_resolve)
    import hermes_cli.config as cfg_mod
    monkeypatch.setattr(
        cfg_mod, "load_config",
        lambda: types.SimpleNamespace(
            mtime=999,
            **{"get": lambda k, d=None: _make_preset_config().get(k, d)},
        ),
    )
    monkeypatch.setattr(moa, "call_llm", lambda **k: _fake_response())

    cc = moa.MoAChatCompletions("demo")
    for _ in range(2):
        cc.create(messages=[{"role": "user", "content": "hi"}])

    # aggregator(1) + 2 references = 3 distinct slots, resolved once
    # each regardless of 2 create() iterations.
    assert calls["n"] == 3, f"expected 3 slot resolutions, got {calls['n']}"


# ─── test harness helpers ──────────────────────────────────────────────

def _fake_response():
    ns = types.SimpleNamespace()
    ns.usage = None
    return ns

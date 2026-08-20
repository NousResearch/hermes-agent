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

import types  # noqa: F401  (used by _fake_response)

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


def test_preset_resolution_is_cached_across_create_calls(monkeypatch, tmp_path):
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
    # The cache keys on the config FILE's st_mtime_ns — give the test a real
    # stat-able file (no config file -> stamp=None -> caching fails open).
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("moa: {}\n")
    monkeypatch.setattr(cfg_mod, "get_config_path", lambda: cfg_file)
    monkeypatch.setattr(cfg_mod, "load_config", lambda: _make_preset_config())
    monkeypatch.setattr(moa, "call_llm", lambda **k: _fake_response())

    cc = moa.MoAChatCompletions("demo")
    for _ in range(3):
        cc.create(messages=[{"role": "user", "content": "hi"}])

    # One preset resolution for the whole turn (not 3).
    assert calls["n"] == 1, f"expected 1 preset resolution, got {calls['n']}"


def test_preset_cache_invalidates_on_config_edit(monkeypatch, tmp_path):
    """Editing config.yaml must invalidate the preset cache on the next
    create() — the original PR keyed on a nonexistent config-object mtime
    attribute, which never invalidated (review finding)."""
    import os

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
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("moa: {}\n")
    monkeypatch.setattr(cfg_mod, "get_config_path", lambda: cfg_file)
    monkeypatch.setattr(cfg_mod, "load_config", lambda: _make_preset_config())
    monkeypatch.setattr(moa, "call_llm", lambda **k: _fake_response())

    cc = moa.MoAChatCompletions("demo")
    cc.create(messages=[{"role": "user", "content": "hi"}])
    assert calls["n"] == 1

    # Simulate a config edit: bump the file's mtime past ns resolution.
    st = cfg_file.stat()
    os.utime(cfg_file, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    cc.create(messages=[{"role": "user", "content": "hi"}])
    assert calls["n"] == 2, "config edit must invalidate the preset cache"


def test_no_config_file_fails_open(monkeypatch, tmp_path):
    """No config.yaml (stat raises) -> caching disabled, create() still works."""
    import agent.moa_loop as moa

    moa._preset_cache.clear()

    import hermes_cli.config as cfg_mod
    monkeypatch.setattr(
        cfg_mod, "get_config_path", lambda: tmp_path / "missing.yaml"
    )
    monkeypatch.setattr(cfg_mod, "load_config", lambda: _make_preset_config())
    monkeypatch.setattr(moa, "call_llm", lambda **k: _fake_response())

    cc = moa.MoAChatCompletions("demo")
    cc.create(messages=[{"role": "user", "content": "hi"}])
    assert moa._preset_cache == {}, "must not cache under a None stamp"


def test_slot_runtime_is_cached_across_create_calls(monkeypatch, tmp_path):
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
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("moa: {}\n")
    monkeypatch.setattr(cfg_mod, "get_config_path", lambda: cfg_file)
    monkeypatch.setattr(cfg_mod, "load_config", lambda: _make_preset_config())
    monkeypatch.setattr(moa, "call_llm", lambda **k: _fake_response())

    cc = moa.MoAChatCompletions("demo")
    for _ in range(2):
        cc.create(messages=[{"role": "user", "content": "hi"}])

    # aggregator(1) + 2 references = 3 distinct slots, resolved once
    # each regardless of 2 create() iterations.
    assert calls["n"] == 3, f"expected 3 slot resolutions, got {calls['n']}"


def test_slot_runtime_cache_expires_after_ttl(monkeypatch):
    """A stale runtime entry (key rotation window) must re-resolve after
    the TTL — the original PR cached for the process lifetime, pinning
    rotated credentials forever (review finding)."""
    import agent.moa_loop as moa

    moa._runtime_cache.clear()

    calls = {"n": 0}

    def counting_resolve(*a, **k):
        calls["n"] += 1
        return {"base_url": "http://x", "api_key": f"key-{calls['n']}",
                "api_mode": None}

    import hermes_cli.runtime_provider as rt_mod
    monkeypatch.setattr(rt_mod, "resolve_runtime_provider", counting_resolve)

    slot = {"provider": "openai", "model": "gpt-5"}
    first = moa._slot_runtime(slot)
    assert calls["n"] == 1 and first["api_key"] == "key-1"

    # Within TTL: cached.
    assert moa._slot_runtime(slot)["api_key"] == "key-1"
    assert calls["n"] == 1

    # Age the entry past the TTL and confirm re-resolution. The cache key
    # includes the slot's explicit connection fields (empty for this slot).
    key = ("openai", "gpt-5", "", "", "")
    stamped_at, cached = moa._runtime_cache[key]
    moa._runtime_cache[key] = (
        stamped_at - moa._RUNTIME_CACHE_TTL_SECONDS - 1, cached
    )
    assert moa._slot_runtime(slot)["api_key"] == "key-2"
    assert calls["n"] == 2


def test_slot_runtime_resolution_error_is_not_cached(monkeypatch):
    """A transient resolution failure must not pin the bare-kwargs fallback
    for a TTL — the next call must retry the real resolver."""
    import agent.moa_loop as moa

    moa._runtime_cache.clear()

    calls = {"n": 0}

    def flaky_resolve(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("catalog hiccup")
        return {"base_url": "http://ok", "api_key": None, "api_mode": None}

    import hermes_cli.runtime_provider as rt_mod
    monkeypatch.setattr(rt_mod, "resolve_runtime_provider", flaky_resolve)

    slot = {"provider": "openai", "model": "gpt-5"}
    fallback = moa._slot_runtime(slot)
    assert "base_url" not in fallback  # bare kwargs on error
    assert moa._runtime_cache == {}, "error result must not be cached"

    recovered = moa._slot_runtime(slot)
    assert recovered.get("base_url") == "http://ok"
    assert calls["n"] == 2


# ─── test harness helpers ──────────────────────────────────────────────

def _fake_response():
    ns = types.SimpleNamespace()
    ns.usage = None
    return ns

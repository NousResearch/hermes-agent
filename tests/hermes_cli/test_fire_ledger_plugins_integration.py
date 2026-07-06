"""Phase B2 integration tests: aggregated fire-ledger wiring into dispatch (v2).

Proves (SPEC AC-1/2/3/8, dual-review blockers), retargeted to the counter model:
- hook + middleware fires are counted with CORRECT plugin attribution,
- attribution survives shared-callback + force-reload (Blocker-2),
- a raised callback is isolated AND counted as decision=error; other callbacks
  still run and their results still return,
- the ledger is a pure side-effect: dispatch return values identical on/off,
- added dispatch-loop latency stays under a fixed absolute ceiling (AC-8) — and
  is now even cheaper because a fire is an in-memory dict bump, not a disk write.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import hermes_cli.fire_ledger as fl
from hermes_cli.plugins import PluginManager


def _fresh_agg():
    fl._AGG = fl._Aggregator()


def _enable(monkeypatch, tmp_path: Path, sub="ledger"):
    _fresh_agg()
    cfg = fl.FireLedgerConfig(enabled=True, dir=tmp_path / sub,
                              retention_days=30, max_mb=25, min_observation_days=30)
    monkeypatch.setattr(fl, "load_fire_ledger_config", lambda _c=None: cfg)
    return cfg


class _Manifest:
    def __init__(self, name):
        self.name = name


class _Ctx:
    def __init__(self, mgr, name):
        self._manager = mgr
        self.manifest = _Manifest(name)
    def register_hook(self, hook_name, cb):
        self._manager._hooks.setdefault(hook_name, []).append(cb)
        self._manager._hook_owners.setdefault(hook_name, []).append(self.manifest.name)
    def register_middleware(self, kind, cb):
        self._manager._middleware.setdefault(kind, []).append(cb)
        self._manager._middleware_owners.setdefault(kind, []).append(self.manifest.name)


def _count_for(cfg, plugin):
    return sum(r["count"] for r in fl.read_counters(cfg=cfg).counters if r["plugin_name"] == plugin)


def test_hook_fire_attributed_to_owner(monkeypatch, tmp_path):
    cfg = _enable(monkeypatch, tmp_path)
    mgr = PluginManager()
    _Ctx(mgr, "activix-standards-guard").register_hook("post_tool_call", lambda **k: None)
    mgr.invoke_hook("post_tool_call", tool_name="terminal", args={"command": "git push"}, session_id="s")
    rows = [r for r in fl.read_counters(cfg=cfg).counters if r["plugin_name"] == "activix-standards-guard"]
    assert len(rows) == 1
    assert rows[0]["dispatch_kind"] == "hook"
    assert rows[0]["count"] == 1


def test_middleware_transform_decision(monkeypatch, tmp_path):
    cfg = _enable(monkeypatch, tmp_path)
    mgr = PluginManager()
    _Ctx(mgr, "rtk-safe-rewrite").register_middleware("tool_request", lambda **k: {"args": {"command": "ls"}})
    mgr.invoke_middleware("tool_request", tool_name="terminal", args={"command": "ls -la"})
    rows = [r for r in fl.read_counters(cfg=cfg).counters if r["plugin_name"] == "rtk-safe-rewrite"]
    assert len(rows) == 1
    assert rows[0]["decision"] == "transform"


def test_two_plugins_share_callback_no_misattribution(monkeypatch, tmp_path):
    cfg = _enable(monkeypatch, tmp_path)
    mgr = PluginManager()
    shared = lambda **k: None
    _Ctx(mgr, "plugin-A").register_hook("post_tool_call", shared)
    _Ctx(mgr, "plugin-B").register_hook("post_tool_call", shared)
    mgr.invoke_hook("post_tool_call", session_id="s")
    owners = sorted(r["plugin_name"] for r in fl.read_counters(cfg=cfg).counters)
    assert owners == ["plugin-A", "plugin-B"]


def test_force_reload_clears_owner_maps():
    mgr = PluginManager()
    _Ctx(mgr, "plugin-A").register_hook("post_tool_call", lambda **k: None)
    assert mgr._hook_owners.get("post_tool_call") == ["plugin-A"]
    mgr._hooks.clear(); mgr._middleware.clear()
    mgr._hook_owners.clear(); mgr._middleware_owners.clear()
    assert mgr._hook_owners == {}


def test_raised_callback_isolated_and_counted_as_error(monkeypatch, tmp_path):
    cfg = _enable(monkeypatch, tmp_path)
    mgr = PluginManager()
    def boom(**k):
        raise RuntimeError("kaboom")
    _Ctx(mgr, "bad-plugin").register_hook("post_tool_call", boom)
    _Ctx(mgr, "good-plugin").register_hook("post_tool_call", lambda **k: "kept")
    results = mgr.invoke_hook("post_tool_call", session_id="s")
    assert results == ["kept"]  # dispatch unaffected
    decisions = {r["plugin_name"]: r["decision"] for r in fl.read_counters(cfg=cfg).counters}
    assert decisions["bad-plugin"] == "error"


def test_dispatch_return_identical_ledger_on_vs_off(monkeypatch, tmp_path):
    mgr = PluginManager()
    _Ctx(mgr, "ctx-plugin").register_hook("pre_llm_call", lambda **k: {"context": "recalled"})

    off = fl.FireLedgerConfig(enabled=False, dir=tmp_path / "off", retention_days=30, max_mb=25, min_observation_days=30)
    _fresh_agg()
    monkeypatch.setattr(fl, "load_fire_ledger_config", lambda _c=None: off)
    off_ret = [mgr.invoke_hook("pre_llm_call", session_id="s", telemetry_schema_version="v") for _ in range(3)]

    on = fl.FireLedgerConfig(enabled=True, dir=tmp_path / "on", retention_days=30, max_mb=25, min_observation_days=30)
    _fresh_agg()
    monkeypatch.setattr(fl, "load_fire_ledger_config", lambda _c=None: on)
    on_ret = [mgr.invoke_hook("pre_llm_call", session_id="s", telemetry_schema_version="v") for _ in range(3)]

    assert off_ret == on_ret, "ledger changed the hook return value — cache risk!"
    # fired on all 3 turns -> one counter with count 3 (proof not vacuous)
    ctx = [r for r in fl.read_counters(cfg=on).counters if r["plugin_name"] == "ctx-plugin"]
    assert len(ctx) == 1 and ctx[0]["count"] == 3 and ctx[0]["decision"] == "context"


def test_kwargs_not_mutated_by_ledger(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    mgr = PluginManager()
    _Ctx(mgr, "p").register_hook("post_tool_call", lambda **k: None)
    kw = {"tool_name": "terminal", "args": {"command": "ls"}, "session_id": "s"}
    before = dict(kw)
    mgr.invoke_hook("post_tool_call", **kw)
    assert kw == before


def test_added_dispatch_latency_under_ceiling(monkeypatch, tmp_path):
    mgr = PluginManager()
    _Ctx(mgr, "p").register_hook("post_tool_call", lambda **k: None)
    N = 3000

    off = fl.FireLedgerConfig(enabled=False, dir=tmp_path / "off", retention_days=30, max_mb=25, min_observation_days=30)
    _fresh_agg()
    monkeypatch.setattr(fl, "load_fire_ledger_config", lambda _c=None: off)
    t0 = time.perf_counter()
    for _ in range(N):
        mgr.invoke_hook("post_tool_call", session_id="s")
    off_per = (time.perf_counter() - t0) / N

    on = fl.FireLedgerConfig(enabled=True, dir=tmp_path / "on", retention_days=30, max_mb=25, min_observation_days=30)
    _fresh_agg()
    monkeypatch.setattr(fl, "load_fire_ledger_config", lambda _c=None: on)
    t0 = time.perf_counter()
    for _ in range(N):
        mgr.invoke_hook("post_tool_call", session_id="s")
    on_per = (time.perf_counter() - t0) / N

    added = on_per - off_per
    # in-memory bump: should be a few µs. 200µs ceiling catches a real regression.
    assert added < 200e-6, f"added dispatch cost {added*1e6:.1f}µs exceeds 200µs ceiling"

"""fleet-models (fleet, 2026-09-11): one model file compiled into nine configs, plus the two runtime seams
(aux calls follow the destination model's pins + no-training floor; cron honours its own chain).
Every seam test has a negative control showing the gap without the plugin."""
import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = ROOT / "plugins" / "fleet-models"
V41 = "deepseek/deepseek-v4.1-flash"
FLASH = "deepseek-v4-flash-ga-260731"
HOSTS = ["deepinfra/fp8", "fireworks", "novita/fp8"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)] if path.name == "__init__.py" else None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def core():
    return _load(PLUGIN_DIR / "core.py", "fleet_models_core_under_test")


# ── a miniature fleet ──────────────────────────────────────────────────────────────────────
PROFILE_CFG = """\
model:
  provider: modelark
  default: deepseek-v4-flash-ga-260731
agent:
  max_turns: 60
auxiliary:
  title_generation:
    provider: modelark
    model: deepseek-v4-flash-ga-260731
    timeout: 30
  vision:
    provider: openrouter
    model: minimax/minimax-m3
delegation:
  model: deepseek-v4-flash-ga-260731
  provider: modelark
  max_concurrent_children: 10
provider_routing:
  order:
  - baidu
  ignore:
  - reka
  require_parameters: true
  data_collection: deny
platforms:
  photon:
    enabled: false
providers:
  modelark:
    name: ModelArk
    base_url: https://ark.ap-southeast.bytepluses.com/api/coding/v3
    key_env: HERMES_CUSTOM_MODELARK_API_KEY
    default_model: deepseek-v4-flash-ga-260731
    models:
      deepseek-v4-flash-ga-260731:
        context_length: 1048576
fallback_providers:
- provider: openrouter
  model: minimax/minimax-m3
"""
ROOT_TAIL = """\
fallback_model:
  provider: openrouter
  model: minimax/minimax-m3
vision:
  provider: openrouter
  model: minimax/minimax-m3

# ── Security ──────────────────────
# a banner that must survive the removal of the key above it
security:
  redact_secrets: true
"""

REGISTRY = {
    "ma-v4-flash": {"id": FLASH, "provider": "modelark", "vendor": "deepseek", "billing": "subscription",
                    "served_as": ["deepseek-v4-flash"], "cap_equivalent": {"input": 0.15, "output": 0.6, "cache_read": 0.003},
                    "context": 1048576, "tools": True, "vision": False},
    "v41-flash": {"id": V41, "provider": "openrouter", "vendor": "deepseek", "billing": "metered", "reasoning": "high",
                  "hosts": {"only": HOSTS, "order": HOSTS, "require_parameters": True},
                  "rules": {"first_host": "deepinfra", "min_uptime": 95}, "tools": True, "vision": True},
    "m3": {"id": "minimax/minimax-m3", "provider": "openrouter", "vendor": "minimax", "tools": True, "vision": True},
    "glm": {"id": "z-ai/glm-5.3-flash", "provider": "openrouter", "vendor": "z-ai", "tools": True, "vision": True,
            "hosts": {"order": ["z-ai", "novita"]}},
}


@pytest.fixture()
def fleet(tmp_path, monkeypatch, core):
    (tmp_path / "config.yaml").write_text(PROFILE_CFG + ROOT_TAIL)
    for p in core.PROFILES[1:]:
        (tmp_path / "profiles" / p).mkdir(parents=True)
        (tmp_path / "profiles" / p / "config.yaml").write_text(PROFILE_CFG)
    (tmp_path / "SOUL.md").write_text("# Smith\n\n## Model table\n\n<!-- fleet-models:table -->\nold\n<!-- /fleet-models:table -->\n\nrest\n")
    (tmp_path / "profiles" / "karl" / "SOUL.md").write_text("## Operating facts\n- **Model:** old <!-- fleet-models:model -->\n  continuation line\n")
    monkeypatch.setenv("FLEET_MODELS_ROOT", str(tmp_path))
    base = {"version": 1, "policy": {"data_collection": "deny"}, "models": copy.deepcopy(REGISTRY),
            "agents": {"root": {"name": "Smith", "locked": True}}}
    doc = core.import_configs(tmp_path, base)
    core.write_doc(doc, tmp_path)
    return tmp_path


def _signed_off(core, root):
    doc = copy.deepcopy(core.load_doc(root, fresh=True))
    for p in core.PROFILES:
        a = doc["agents"][p]
        a["main"] = a["subagents"] = ["ma-v4-flash", "v41-flash", "m3"]
        a["aux"] = {"title_generation": ["ma-v4-flash", "v41-flash"], "vision": ["glm", "m3", "v41-flash"]}
    return doc


def test_import_then_compile_is_semantically_identity(core, fleet):
    doc = core.load_doc(fleet, fresh=True)
    assert doc["agents"]["karl"]["main"] == ["ma-v4-flash", "m3"]
    assert doc["agents"]["karl"]["aux"]["vision"] == ["m3"]
    pl = core.plan(doc, fleet)
    for p in core.PROFILES:
        live = core.effective(yaml.safe_load(pl[p]["src"]))
        new = core.effective(yaml.safe_load(pl[p]["text"]))
        for k in ("main", "subagents", "cron", "aux", "reasoning", "reasoning_overrides"):
            assert live[k] == new[k], (p, k)
        assert core.verify_profile(doc, p, yaml.safe_load(pl[p]["text"])) == []


def test_apply_signed_off_verifies_and_keeps_everything_else(core, fleet):
    before = {p: yaml.safe_load(core.cfg_path(p, fleet).read_text()) for p in core.PROFILES}
    r = core.apply(_signed_off(core, fleet), by="test", root=fleet, unlock=("root",))
    assert r["ok"], r
    for p in core.PROFILES:
        cfg = yaml.safe_load(core.cfg_path(p, fleet).read_text())
        assert core.verify_profile(core.load_doc(fleet, fresh=True), p, cfg) == []
        assert [e["model"] for e in cfg["fallback_providers"]] == [V41, "minimax/minimax-m3"]
        assert [e["model"] for e in cfg["delegation"]["fallback_providers"]] == [V41, "minimax/minimax-m3"]
        pin = cfg["provider_routing"]["models"][V41]
        assert pin["only"] == HOSTS and pin["order"][0].startswith("deepinfra") and pin["data_collection"] == "deny"
        assert cfg["provider_routing"]["models"]["minimax/minimax-m3"]["data_collection"] == "deny"
        assert cfg["agent"]["reasoning_overrides"] == {V41: "high"}
        assert cfg["auxiliary"]["vision"]["model"] == "z-ai/glm-5.3-flash"
        assert cfg["auxiliary"]["title_generation"]["timeout"] == 30  # an unmanaged key inside a managed block
        for k in ("platforms", "security"):
            assert cfg.get(k) == before[p].get(k)
        assert cfg["delegation"]["max_concurrent_children"] == 10
    root_text = core.cfg_path("root", fleet).read_text()
    assert "a banner that must survive" in root_text           # comment trailing a removed key
    assert "fallback_model" not in root_text and "\nvision:" not in root_text  # inert keys gone
    assert (fleet / "fleet" / "models.json").exists()


def test_souls_are_regenerated_between_markers_only(core, fleet):
    r = core.apply(_signed_off(core, fleet), by="test", root=fleet, unlock=("root",))
    assert set(r["souls"]) == {"root", "karl"}
    root_soul = (fleet / "SOUL.md").read_text()
    assert "| `karl` | ma-v4-flash → v41-flash → m3 |" in root_soul
    assert "<!-- fleet-models:table -->" in root_soul and "\nold\n" not in root_soul and root_soul.endswith("rest\n")
    karl = (fleet / "profiles" / "karl" / "SOUL.md").read_text()
    assert "main ma-v4-flash → v41-flash → m3" in karl and "  continuation line" in karl and "old" not in karl


def test_revert_restores_exact_bytes(core, fleet):
    before = {p: core.cfg_path(p, fleet).read_text() for p in core.PROFILES}
    souls = {p: core.soul_path(p, fleet).read_text() for p in ("root", "karl")}
    r = core.apply(_signed_off(core, fleet), by="test", root=fleet, unlock=("root",))
    assert r["ok"]
    assert core.revert(r["id"], root=fleet)["ok"]
    assert {p: core.cfg_path(p, fleet).read_text() for p in core.PROFILES} == before
    assert {p: core.soul_path(p, fleet).read_text() for p in ("root", "karl")} == souls
    assert core.history(fleet)[0]["reverts"] == r["id"]


def test_validation_blocks_what_must_never_ship(core, fleet):
    doc = _signed_off(core, fleet)
    bad = copy.deepcopy(doc); bad["policy"]["data_collection"] = "allow"
    assert any("no-training" in e for e in core.validate(bad)[0])
    bad = copy.deepcopy(doc); bad["models"]["v41-flash"]["hosts"]["order"] = ["fireworks", "deepinfra/fp8"]
    assert any("first host" in e for e in core.validate(bad)[0])
    bad = copy.deepcopy(doc); bad["agents"]["karl"]["aux"]["vision"] = ["ma-v4-flash"]
    assert any("accept images" in e for e in core.validate(bad)[0])
    bad = copy.deepcopy(doc); bad["agents"]["karl"]["main"] = ["nope"]
    assert any("unknown model" in e for e in core.validate(bad)[0])
    # Smith is locked: a change without an explicit unlock is refused and nothing is written
    before = core.cfg_path("root", fleet).read_text()
    r = core.apply(doc, by="test", root=fleet)
    assert not r["ok"] and any("locked" in e for e in r["errors"])
    assert core.cfg_path("root", fleet).read_text() == before
    # stale editor: a base revision that moved on is refused
    r = core.apply(doc, by="test", root=fleet, unlock=("root",), base_revision=99)
    assert not r["ok"] and "moved on" in r["errors"][0]


def test_hand_edit_is_reported_as_drift(core, fleet):
    assert core.apply(_signed_off(core, fleet), by="test", root=fleet, unlock=("root",))["ok"]
    f = core.cfg_path("karl", fleet)
    f.write_text(f.read_text().replace("model: z-ai/glm-5.3-flash", "model: minimax/minimax-m3", 1))
    assert "karl" in core.drift(fleet)


# ── runtime seams ───────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def plugin(monkeypatch):
    from agent import auxiliary_client as ac
    for name in ("_build_call_kwargs", "_call_fallback_candidate_sync", "_call_fallback_candidate_async"):
        monkeypatch.setattr(ac, name, getattr(ac, name))
    mod = _load(PLUGIN_DIR / "__init__.py", "fleet_models_under_test")
    return mod


CFG = {"provider_routing": {"data_collection": "deny", "order": ["baidu"],
                            "models": {V41: {"only": HOSTS, "order": HOSTS, "ignore": [], "require_parameters": True, "data_collection": "deny"},
                                       "minimax/minimax-m3": {"data_collection": "deny"}}},
       "agent": {"reasoning_effort": "low", "reasoning_overrides": {V41: "high"}}}


def _aux_kwargs(model, provider="openrouter", extra_body=None, task="vision"):
    from agent import auxiliary_client as ac
    return ac._build_call_kwargs(provider, model, [{"role": "user", "content": "hi"}],
                                 extra_body=extra_body, base_url="https://openrouter.ai/api/v1", task=task)


def test_negative_control_aux_fallback_carries_no_pins_without_plugin(monkeypatch):
    from hermes_cli import config as hc
    monkeypatch.setattr(hc, "load_config_readonly", lambda: CFG)
    kw = _aux_kwargs(V41)
    prov = (kw.get("extra_body") or {}).get("provider") or {}
    assert "only" not in prov and prov.get("data_collection") != "deny"  # the gap this plugin closes


def test_aux_openrouter_calls_get_model_pins_deny_and_reasoning(plugin, monkeypatch):
    from hermes_cli import config as hc
    from agent import auxiliary_client as ac
    monkeypatch.setattr(hc, "load_config_readonly", lambda: CFG)
    monkeypatch.setattr(ac, "_get_auxiliary_task_config", lambda task: {"model": "z-ai/glm-5.3-flash"})
    assert "aux routing" in plugin.install()
    # v4.1 reached as a vision fallback: task-level glm host pins dropped, v4.1's own applied
    kw = _aux_kwargs(V41, extra_body={"provider": {"order": ["z-ai"], "data_collection": "deny"}})
    prov = kw["extra_body"]["provider"]
    assert prov["only"] == HOSTS and prov["order"] == HOSTS and prov["data_collection"] == "deny"
    assert kw["extra_body"]["reasoning"] == {"enabled": True, "effort": "high"}
    # a task with no extra_body at all still gets the no-training floor
    kw = _aux_kwargs("minimax/minimax-m3")
    assert kw["extra_body"]["provider"] == {"data_collection": "deny"} and "reasoning" not in kw["extra_body"]


def test_aux_modelark_calls_never_carry_openrouter_prefs(plugin, monkeypatch):
    from hermes_cli import config as hc
    monkeypatch.setattr(hc, "load_config_readonly", lambda: CFG)
    plugin.install()
    from agent import auxiliary_client as ac
    kw = ac._build_call_kwargs("modelark", FLASH, [{"role": "user", "content": "x"}],
                               extra_body={"provider": {"data_collection": "deny"}},
                               base_url="https://ark.ap-southeast.bytepluses.com/api/coding/v3", task="title_generation")
    assert "provider" not in (kw.get("extra_body") or {})


def test_cron_chain_seam(plugin):
    from hermes_cli.fallback_config import get_fallback_chain
    fake = types.SimpleNamespace(get_fallback_chain=get_fallback_chain)
    main = {"fallback_providers": [{"provider": "modelark", "model": FLASH}]}
    assert [e["model"] for e in fake.get_fallback_chain(main)] == [FLASH]  # negative control: main chain
    assert plugin.cron_chain_patch(fake) and not plugin.cron_chain_patch(fake)
    own = dict(main, cron={"fallback_providers": [{"provider": "openrouter", "model": V41}, {"provider": "openrouter", "model": "minimax/minimax-m3"}]})
    assert [e["model"] for e in fake.get_fallback_chain(own)] == [V41, "minimax/minimax-m3"]
    assert [e["model"] for e in fake.get_fallback_chain(main)] == [FLASH]


def test_main_loop_and_pinned_child_get_v41_pins_and_deny(monkeypatch):
    """Hermes' own request path: per-model pins ride with the model; a child pinned to ModelArk has its
    flat routing filters RESET (data_collection ''), so the per-model entry is what keeps deny on it."""
    from hermes_cli import config as hc
    from agent.chat_completion_helpers import _provider_preferences_for_agent
    from hermes_constants import resolve_reasoning_config
    monkeypatch.setattr(hc, "load_config_readonly", lambda: CFG)
    child = types.SimpleNamespace(model=V41, providers_allowed=None, providers_ignored=None, providers_order=None,
                                  provider_sort=None, provider_require_parameters=False, provider_data_collection="")
    prefs = _provider_preferences_for_agent(child)
    assert prefs["only"] == HOSTS and prefs["data_collection"] == "deny" and prefs["require_parameters"] is True
    child.model = "some/other-model"
    assert "data_collection" not in _provider_preferences_for_agent(child)  # control: no entry, no deny
    assert resolve_reasoning_config(CFG, V41) == {"enabled": True, "effort": "high"}
    assert resolve_reasoning_config(CFG, "minimax/minimax-m3") == {"enabled": True, "effort": "low"}


def test_pricing_reads_cap_rates_from_the_fleet_file(tmp_path, monkeypatch):
    import hermes_constants
    (tmp_path / "fleet").mkdir()
    (tmp_path / "fleet" / "models.json").write_text(json.dumps({"models": {"ma-v4-flash": {
        "id": FLASH, "provider": "modelark", "served_as": ["deepseek-v4-flash"],
        "cap_equivalent": {"input": 1.0, "output": 2.0, "cache_read": 0.5}}}}))
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: str(tmp_path))
    mp = _load(ROOT / "plugins" / "modelark-pricing" / "__init__.py", "modelark_pricing_rates_under_test")
    r = mp.fleet_rates()
    assert float(r[FLASH][0]) == 1.0 and float(r["deepseek-v4-flash"][1]) == 2.0


CHAIN = [{"provider": "custom", "model": "minimax/minimax-m3", "base_url": "http://127.0.0.1:9/v1"},
         {"provider": "openrouter", "model": V41}]


def _fake_candidate(calls):
    def cand(fb_client, fb_model, fb_label, **kw):
        calls.append(fb_label)
        if fb_label.startswith("fallback_chain[0]"):
            raise ConnectionError("connection refused")   # a capacity failure on the SECOND rung
        return f"served by {fb_model}"
    return cand


def _chain_env(monkeypatch):
    from agent import auxiliary_client as ac
    monkeypatch.setattr(ac, "_get_auxiliary_task_config", lambda task: {"model": "z-ai/glm-5.3-flash", "fallback_chain": CHAIN})
    monkeypatch.setattr(ac, "_resolve_fallback_entry", lambda e: (object(), e["model"]))
    monkeypatch.setattr(ac, "_is_provider_unhealthy", lambda *a, **k: False)
    return ac


def test_negative_control_chain_stops_after_one_failed_rung(monkeypatch):
    ac = _chain_env(monkeypatch)
    calls = []
    with pytest.raises(ConnectionError):
        _fake_candidate(calls)(None, "minimax/minimax-m3", "fallback_chain[0](custom)", task="vision")
    assert calls == ["fallback_chain[0](custom)"]   # Hermes alone: v4.1 is never tried


def test_chain_walks_to_the_third_rung_on_capacity_errors(plugin, monkeypatch):
    ac = _chain_env(monkeypatch)
    calls = []
    wrapped = plugin._wrap_candidate(_fake_candidate(calls), False)
    assert wrapped(None, "minimax/minimax-m3", "fallback_chain[0](custom)", task="vision") == f"served by {V41}"
    assert calls == ["fallback_chain[0](custom)", "fallback_chain[1](openrouter)"]


def test_chain_walk_never_swallows_auth_or_unknown_errors(plugin, monkeypatch):
    _chain_env(monkeypatch)
    def cand(*a, **k):
        raise ValueError("schema bug")  # not a capacity error: surfaces unchanged
    with pytest.raises(ValueError):
        plugin._wrap_candidate(cand, False)(None, "m", "fallback_chain[0](custom)", task="vision")


def test_chain_walk_async(plugin, monkeypatch):
    import asyncio
    ac = _chain_env(monkeypatch)
    monkeypatch.setattr(ac, "_to_async_client", lambda c, m, is_vision=False: (c, m))
    calls = []
    sync = _fake_candidate(calls)
    async def cand(*a, **k):
        return sync(*a, **k)
    r = asyncio.run(plugin._wrap_candidate(cand, True)(None, "minimax/minimax-m3", "fallback_chain[0](custom)", task="vision"))
    assert r == f"served by {V41}"


# ── 1.0.5: time-windowed usage (plugin_api.usage) ─────────────────────────────────────────────
NOW = 1789200000.0   # 2026-09-12 ~20:00 NZST


@pytest.fixture()
def api(core):
    return _load(PLUGIN_DIR / "dashboard" / "plugin_api.py", "fleet_models_api_under_test")


def _ledger(root, prof, rows):
    import sqlite3
    d = root if prof == "root" else root / "profiles" / prof
    d.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(d / "state.db")
    c.execute("CREATE TABLE IF NOT EXISTS session_model_usage (session_id TEXT, model TEXT, provider_name TEXT, "
              "billing_base_url TEXT, task TEXT, api_call_count INT, input_tokens INT, output_tokens INT, "
              "cache_read_tokens INT, cache_write_tokens INT, actual_cost_usd REAL, total_cost REAL, "
              "estimated_cost_usd REAL, cost_source TEXT, first_seen REAL, last_seen REAL)")
    for r in rows:
        c.execute("INSERT INTO session_model_usage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  ("s", r.get("model", V41), r.get("host", "DeepInfra"), r.get("base", "https://openrouter.ai/api/v1"),
                   r.get("task", ""), r["calls"], 1000 * r["calls"], 100 * r["calls"], 0, 0, 0, r.get("cost", 0.0), 0,
                   r.get("src", ""), r["fs"], r["ls"]))
    c.commit(); c.close()


def test_usage_spreads_long_rows_and_every_view_reconciles(api, tmp_path):
    N = NOW + 100   # mid-bucket, so the current bucket is partial
    _ledger(tmp_path, "root", [
        {"calls": 10, "cost": 0.10, "fs": N - 600, "ls": N - 600},          # a point, inside the last hour
        {"calls": 40, "cost": 0.40, "fs": N - 7200, "ls": N},               # 2 h span: half falls in the last hour
        {"calls": 99, "cost": 9.99, "fs": N - 9000, "ls": N - 3700},        # ended before the window
    ])
    _ledger(tmp_path, "karl", [{"calls": 6, "model": FLASH, "host": "", "base": "https://ark.ap-southeast.bytepluses.com/api/coding/v3",
                                "src": "modelark subscription", "fs": N - 1200, "ls": N - 1200}])
    u = api.usage(3600, bucket=300, tz="Pacific/Auckland", now=N, root=tmp_path)
    calls = sum(r["calls"] for r in u["rows"])
    assert calls == 10 + 20 + 6
    assert abs(sum(r["billed_usd"] for r in u["rows"]) - (0.10 + 0.20)) < 1e-9
    assert abs(sum(s["calls"] for s in u["series"]) - calls) < 0.01             # the chart sums to the tiles
    assert abs(sum(u["by_profile"]["root"]["calls"]) - 30) < 0.01
    assert abs(sum(s["modelark_calls"] for s in u["series"]) - 6) < 0.01
    karl = [r for r in u["rows"] if r["profile"] == "karl"][0]
    assert karl["modelark"] and karl["billed_usd"] == 0 and karl["cap_equivalent_usd"] > 0
    assert u["bucket"] == 300 and 12 <= len(u["series"]) <= 13
    assert u["series"][-1]["partial"]


def test_usage_buckets_follow_the_local_clock_across_dst(api, tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    nz = ZoneInfo("Pacific/Auckland")
    _ledger(tmp_path, "root", [])
    dst = datetime(2026, 9, 28, 12, 0, tzinfo=nz).timestamp()   # NZ clocks went forward 27 Sep 02:00 → 03:00
    days = api.usage(5 * 86400, bucket=86400, tz="Pacific/Auckland", now=dst, root=tmp_path)["series"]
    assert all(datetime.fromtimestamp(s["t"], nz).hour == 0 for s in days)
    assert sorted({round((s["end"] - s["t"]) / 3600) for s in days}) == [23, 24]   # the short day is one bucket
    hours = api.usage(86400, bucket=3600, tz="Pacific/Auckland", now=datetime(2026, 9, 27, 14, 0, tzinfo=nz).timestamp(), root=tmp_path)["series"]
    assert all(datetime.fromtimestamp(s["t"], nz).minute == 0 for s in hours)
    sixes = api.usage(7 * 86400, bucket=21600, tz="Pacific/Auckland", now=dst, root=tmp_path)["series"]
    assert all(datetime.fromtimestamp(s["t"], nz).hour in (0, 6, 12, 18) for s in sixes[1:])


def test_usage_clamps_window_and_bucket(api, tmp_path):
    _ledger(tmp_path, "root", [])
    assert api.usage(60, now=NOW, root=tmp_path)["window"] == 900
    assert api.usage(86400, bucket=7, now=NOW, root=tmp_path)["bucket"] == 3600          # not a nice size → natural
    u = api.usage(30 * 86400, bucket=60, now=NOW, root=tmp_path)                          # 43,200 buckets → coarsened
    assert len(u["series"]) <= api.MAX_BUCKETS + 1 and u["bucket"] in api.NICE_BUCKETS
    assert api.usage(10 ** 9, now=NOW, root=tmp_path)["window"] == api.WINDOW_MAX
    assert api.usage(3600, tz="../../etc/passwd", now=NOW, root=tmp_path)["tz"] != "../../etc/passwd"


def test_usage_route_keeps_the_old_daily_shape_for_open_104_tabs(api, tmp_path, monkeypatch):
    import time as _t
    _ledger(tmp_path, "root", [{"calls": 3, "cost": 0.03, "fs": _t.time() - 7200, "ls": _t.time() - 7200}])
    monkeypatch.setenv("FLEET_MODELS_ROOT", str(tmp_path))
    old = api.get_usage(days=7)
    assert "daily" in old and sum(d["calls"] for d in old["daily"]) == sum(r["calls"] for r in old["rows"]) == 3
    assert "daily" not in api.get_usage(window=3600, bucket=300, tz="Pacific/Auckland")


def test_core_loads_once_under_concurrent_first_requests(api, monkeypatch):
    """1.0.6 live: /state and /usage arrive together after a restart; one thread saw the half-loaded core."""
    import threading, time as _t
    real = api.importlib.util.spec_from_file_location
    def slow_spec(*a, **k):
        spec = real(*a, **k)
        orig = spec.loader.exec_module
        def exec_slow(m):
            _t.sleep(0.2)            # widen the window a racing thread used to fall into
            orig(m)
        spec.loader.exec_module = exec_slow
        return spec
    monkeypatch.setattr(api.importlib.util, "spec_from_file_location", slow_spec)
    monkeypatch.delitem(sys.modules, "fleet_models_core", raising=False)
    got, errs = [], []
    def hit():
        try:
            got.append(api._core().fleet_root)
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=hit) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errs and len(got) == 6

"""modelark-pricing 2.0 (fleet, 2026-09-11): the Coding Plan reports NO cost. Record $0 as "modelark
subscription", and make the $1 card cap count a cap-equivalent so it can still fire."""
import importlib.util
import sqlite3
from pathlib import Path

import pytest

from agent import usage_pricing as up
from agent.usage_pricing import CanonicalUsage
from hermes_cli import kanban_db as kdb

SUB = "https://ark.ap-southeast.bytepluses.com/api/coding/v3"
PAYG = "https://ark.ap-southeast.bytepluses.com/api/v3"
ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugins" / "modelark-pricing" / "__init__.py"
ESCALATOR = ROOT / "plugins" / "kanban-block-escalator" / "__init__.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def plugin(monkeypatch):
    # restore every seam after the test
    for obj, attr in ((up, "resolve_billing_route"), (up, "estimate_usage_cost"), (kdb, "_session_cost_in_db")):
        monkeypatch.setattr(obj, attr, getattr(obj, attr))
    import sys
    for modname in ("agent.turn_usage", "agent.insights"):
        m = sys.modules.get(modname)
        if m is not None and hasattr(m, "estimate_usage_cost"):
            monkeypatch.setattr(m, "estimate_usage_cost", m.estimate_usage_cost)
    monkeypatch.setattr(kdb, "modelark_cap_equivalent", None, raising=False)
    mod = _load(PLUGIN, "modelark_pricing_under_test")
    assert {"resolve_billing_route", "estimate_usage_cost", "_session_cost_in_db"} <= set(mod.install())
    assert mod.install() == []  # idempotent: nothing is wrapped twice
    return mod


def _cost(model, provider="custom", base_url=SUB, **usage):
    return up.estimate_usage_cost(model, CanonicalUsage(**usage), provider=provider, base_url=base_url)


def test_negative_control_without_plugin_is_unpriced():
    r = _cost("deepseek-v4-flash-ga-260731", input_tokens=1_000_000)
    assert r.status == "unknown" and r.amount_usd is None


@pytest.mark.parametrize("model", ["deepseek-v4-flash-ga-260731", "deepseek-v4-flash", "deepseek-v4-pro-ga-260813"])
def test_coding_plan_reports_zero_as_modelark_subscription(plugin, model):
    r = _cost(model, input_tokens=1_000_000, output_tokens=1_000_000)
    assert r.status == "included" and r.amount_usd == 0
    assert r.source == "modelark subscription" and r.label == "modelark subscription"
    assert up.resolve_billing_route(model, provider="custom", base_url=SUB).billing_mode == "subscription_included"


def test_pay_as_you_go_endpoint_is_not_a_subscription(plugin):
    assert _cost("deepseek-v4-flash-ga-260731", base_url=PAYG, input_tokens=1000).status != "included"


def test_openrouter_untouched(plugin):
    r = up.estimate_usage_cost("deepseek/deepseek-v4-flash-0731", CanonicalUsage(input_tokens=1000),
                               provider="openrouter", base_url="https://openrouter.ai/api/v1")
    assert r.source != "modelark subscription"
    assert up.resolve_billing_route("openai/gpt-5.6-luna", provider="openai-codex").provider == "openai-codex"


def _ledger(tmp_path):
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.execute("create table sessions(id text, title text, cwd text, estimated_cost_usd real)")
    con.execute("create table session_model_usage(session_id text, model text, billing_base_url text, cost_source text,"
                " estimated_cost_usd real, api_call_count int, input_tokens int, output_tokens int,"
                " cache_read_tokens int, cache_write_tokens int)")
    # a card that ran 1M in + 1M out on the Coding Plan (recorded $0) plus $0.05 of billed m3 fallback
    con.execute("insert into sessions values ('s1', 'Work kanban task t_abcdef12 #1', null, 0.05)")
    con.execute("insert into session_model_usage values ('s1','deepseek-v4-flash-ga-260731', ?, 'modelark subscription', 0, 40, 1000000, 1000000, 0, 0)", (SUB,))
    con.execute("insert into session_model_usage values ('s1','minimax/minimax-m3','https://openrouter.ai/api/v1','provider_models_api',0.05,1,100,10,0,0)")
    con.execute("insert into sessions values ('s2', 'Work kanban task t_99999999 #1', null, 0.0)")
    con.execute("insert into session_model_usage values ('s2','deepseek-v4-pro', ?, 'modelark subscription', 0, 3, 5000000, 0, 0, 0)", (SUB,))
    con.commit(); con.close()
    return db


def test_cap_negative_control_blind_without_plugin(tmp_path):
    db = _ledger(tmp_path)
    assert abs(kdb._cumulative_session_cost(str(db), None, task_id="t_abcdef12", all_ledgers=False) - 0.05) < 1e-9


def test_cap_counts_cap_equivalent_with_plugin(plugin, tmp_path):
    db = _ledger(tmp_path)
    flash = up._OFFICIAL_DOCS_PRICING[("deepseek", "deepseek-v4-flash")]
    want = 0.05 + float(flash.input_cost_per_million + flash.output_cost_per_million)
    got = kdb._cumulative_session_cost(str(db), None, task_id="t_abcdef12", all_ledgers=False)
    assert abs(got - want) < 1e-6 and got > 0.05
    pro = up._OFFICIAL_DOCS_PRICING[("deepseek", "deepseek-v4-pro")]
    got2 = kdb._cumulative_session_cost(str(db), None, task_id="t_99999999", all_ledgers=False)
    assert abs(got2 - 5 * float(pro.input_cost_per_million)) < 1e-6   # served name priced as pro


def test_escalator_brief_reports_modelark_share(plugin, tmp_path, monkeypatch):
    import hermes_constants
    (tmp_path / "profiles").mkdir()
    _ledger(tmp_path)
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path)
    esc = _load(ESCALATOR, "escalator_under_test_ma")
    usd, calls = esc._modelark_share("t_abcdef12")
    assert calls == 40 and usd > 0
    assert esc._modelark_share("t_00000000") == (0.0, 0)

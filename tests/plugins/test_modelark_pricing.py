"""modelark-pricing: the $1 card cap must see a non-zero cost for ModelArk calls (fleet, 2026-09-11)."""
import importlib.util
from decimal import Decimal
from pathlib import Path

import pytest

from agent import usage_pricing as up
from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

ARK = "https://ark.ap-southeast.bytepluses.com/api/coding/v3"
PLUGIN = Path(__file__).resolve().parents[2] / "plugins" / "modelark-pricing" / "__init__.py"


@pytest.fixture()
def plugin(monkeypatch):
    monkeypatch.setattr(up, "_OFFICIAL_DOCS_PRICING", dict(up._OFFICIAL_DOCS_PRICING))
    spec = importlib.util.spec_from_file_location("modelark_pricing_under_test", PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.install() == 12
    return mod


def _cost(model, provider, **usage):
    return estimate_usage_cost(model, CanonicalUsage(**usage), provider=provider, base_url=ARK)


def test_negative_control_unpriced_without_plugin(monkeypatch):
    monkeypatch.setattr(up, "_OFFICIAL_DOCS_PRICING", dict(up._OFFICIAL_DOCS_PRICING))
    r = _cost("deepseek-v4-flash-ga-260731", "custom", input_tokens=1_000_000)
    assert r.status == "unknown" and not r.amount_usd  # the 2026-09-08 blind spot


@pytest.mark.parametrize("provider", ["custom", "modelark", "custom:modelark"])
def test_flash_priced_at_deepseek_list(plugin, provider):
    flash = up._OFFICIAL_DOCS_PRICING[("deepseek", "deepseek-v4-flash")]
    r = _cost("deepseek-v4-flash-ga-260731", provider, input_tokens=1_000_000, output_tokens=1_000_000)
    assert r.status == "estimated" and r.source == "modelark-proxy"
    assert r.amount_usd == flash.input_cost_per_million + flash.output_cost_per_million > 0


def test_pro_priced_at_deepseek_list(plugin):
    pro = up._OFFICIAL_DOCS_PRICING[("deepseek", "deepseek-v4-pro")]
    r = _cost("deepseek-v4-pro-ga-260813", "custom", input_tokens=2_000_000)
    assert r.amount_usd == 2 * pro.input_cost_per_million and r.source == "modelark-proxy"


def test_cache_write_never_turns_cost_unknown(plugin):
    r = _cost("deepseek-v4-flash-ga-260731", "custom", input_tokens=10, cache_write_tokens=1_000_000)
    assert r.status == "estimated" and r.amount_usd > 0


def test_other_models_untouched(plugin):
    assert _cost("some-other-ark-model", "custom", input_tokens=1000).status == "unknown"
    r = estimate_usage_cost("deepseek/deepseek-v4-flash-0731", CanonicalUsage(input_tokens=1000),
                            provider="openrouter", base_url="https://openrouter.ai/api/v1")
    assert r.source != "modelark-proxy"


def test_upstream_entry_wins(monkeypatch):
    table = dict(up._OFFICIAL_DOCS_PRICING)
    theirs = up.PricingEntry(input_cost_per_million=Decimal("9"), source="official_docs_snapshot")
    table[("custom", "deepseek-v4-flash-ga-260731")] = theirs
    monkeypatch.setattr(up, "_OFFICIAL_DOCS_PRICING", table)
    spec = importlib.util.spec_from_file_location("mp2", PLUGIN)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.install()
    assert up._OFFICIAL_DOCS_PRICING[("custom", "deepseek-v4-flash-ga-260731")] is theirs


ESCALATOR = Path(__file__).resolve().parents[2] / "plugins" / "kanban-block-escalator" / "__init__.py"


def test_escalator_brief_splits_modelark_share(tmp_path, monkeypatch):
    """Overwatch must see which part of a breach is notional subscription cost (2026-09-11)."""
    import sqlite3
    import hermes_constants
    (tmp_path / "profiles" / "karl").mkdir(parents=True)
    for db in (tmp_path / "state.db", tmp_path / "profiles" / "karl" / "state.db"):
        con = sqlite3.connect(db)
        con.execute("create table sessions(id text, title text)")
        con.execute("create table session_model_usage(session_id text, estimated_cost_usd real, api_call_count int, cost_source text)")
        con.commit(); con.close()
    con = sqlite3.connect(tmp_path / "profiles" / "karl" / "state.db")
    con.execute("insert into sessions values ('s1', 'Work kanban task t_abcdef12 #1')")
    con.execute("insert into session_model_usage values ('s1', 0.40, 12, 'modelark-proxy')")
    con.execute("insert into session_model_usage values ('s1', 0.10, 1, 'provider_models_api')")
    con.execute("insert into sessions values ('s2', 'Work kanban task t_99999999 #1')")
    con.execute("insert into session_model_usage values ('s2', 0.70, 5, 'modelark-proxy')")
    con.commit(); con.close()
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path)
    spec = importlib.util.spec_from_file_location("escalator_under_test_ma", ESCALATOR)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    total, calls = mod._modelark_share("t_abcdef12")
    assert calls == 12 and abs(total - 0.40) < 1e-9       # only this card, only the modelark rows
    assert mod._modelark_share("t_00000000") == (0.0, 0)  # negative control

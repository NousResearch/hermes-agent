"""Regression tests for the Hermes Agentic Trader repair safety boundary."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml


AGGREGATOR_BACKED_TOOLS = {
    "get_portfolio_tokens",
    "get_portfolio_balances",
    "get_portfolio_transactions",
    "get_swap_price",
    "get_swap_quote",
    "execute_swap",
    "get_supported_chains",
    "get_liquidity_sources",
    "get_gasless_price",
    "get_gasless_quote",
    "submit_gasless_swap",
    "get_gasless_status",
    "get_gasless_chains",
    "get_gasless_approval_tokens",
}


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv(
        "USER_ADDRESS",
        "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    monkeypatch.setenv("HERMES_TRADER_MANDATE_SECRET", "test-secret")
    monkeypatch.delenv("HERMES_TRADER_KILL_SWITCH", raising=False)
    yield hermes_home


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_plugin():
    plugin_dir = _repo_root() / "plugins" / "hermes-agentic-trader"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.hermes_agentic_trader",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.hermes_agentic_trader"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.hermes_agentic_trader"] = mod
    spec.loader.exec_module(mod)
    cache = sys.modules["hermes_plugins.hermes_agentic_trader.quote_cache"]
    cache.clear_quote_cache()
    return mod


def _write_config(home: Path, trader):
    (home / "config.yaml").write_text(
        yaml.safe_dump({"trader": trader}),
        encoding="utf-8",
    )


def _install_mandate():
    mandate = sys.modules["hermes_plugins.hermes_agentic_trader.mandate"]
    signed = mandate.sign_mandate(
        "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        allowed_chain_ids=(8453,),
        max_trade_usd=50,
        max_slippage_bps=100,
        max_writes_per_hour=10,
    )
    mandate.save_mandate(signed)


def _swap_quote():
    return {
        "chainId": 8453,
        "sellToken": "0x1111111111111111111111111111111111111111",
        "sellAmount": "1000000",
        "buyToken": "0x2222222222222222222222222222222222222222",
        "transaction": {
            "to": "0x3333333333333333333333333333333333333333",
            "data": "0xdeadbeef",
            "gas": "210000",
            "gasPrice": "1000000",
            "value": "0",
        },
    }


def _quote_request_args(quote):
    return {
        "chainId": 8453,
        "buyToken": quote.get(
            "buyToken",
            "0x2222222222222222222222222222222222222222",
        ),
        "sellToken": quote.get(
            "sellToken",
            "0x1111111111111111111111111111111111111111",
        ),
        "sellAmount": quote.get("sellAmount", "1000000"),
        "slippageBps": 50,
    }


def _gasless_quote():
    return {
        "trade": {
            "type": "trade",
            "eip712": {
                "domain": {"chainId": 8453, "name": "Trade"},
                "types": {"Trade": [{"name": "sellAmount", "type": "uint256"}]},
                "primaryType": "Trade",
                "message": {"sellAmount": "1000000"},
            },
        }
    }


class TestWriteQuarantine:
    @pytest.mark.parametrize("tool_name", ["execute_swap", "submit_gasless_swap"])
    def test_paper_mode_blocks_raw_write_tools(self, _isolate_env, tool_name):
        _write_config(_isolate_env, {"mode": "paper"})
        mod = _load_plugin()
        out = mod._on_pre_tool_call(
            tool_name=tool_name,
            args={"quoteData": {}},
            session_id="s1",
        )
        assert out["action"] == "block"
        assert "paper mode" in out["message"].lower()

    def test_live_mode_rejects_unobserved_quote(self, _isolate_env):
        _write_config(_isolate_env, {"mode": "live"})
        mod = _load_plugin()
        out = mod._on_pre_tool_call(
            tool_name="execute_swap",
            args={"quoteData": _swap_quote()},
            session_id="s1",
        )
        assert out["action"] == "block"
        assert "not observed" in out["message"]

    def test_observed_quote_without_mandate_fails_closed(self, _isolate_env):
        _write_config(_isolate_env, {"mode": "live"})
        mod = _load_plugin()
        quote = _swap_quote()
        mod._on_post_tool_call(
            tool_name="get_swap_quote",
            args=_quote_request_args(quote),
            result={"message": "ok", "data": quote},
            status="ok",
            session_id="s1",
        )
        out = mod._on_pre_tool_call(
            tool_name="execute_swap",
            args={"quoteData": quote},
            session_id="s1",
        )
        assert "MANDATE_INVALID" in out["message"]

    def test_valid_mandate_reaches_notional_fail_closed_gate(self, _isolate_env):
        _write_config(_isolate_env, {"mode": "live"})
        mod = _load_plugin()
        _install_mandate()
        quote = _swap_quote()
        result = json.dumps(
            {"result": json.dumps({"message": "ok", "data": quote})}
        )
        mod._on_post_tool_call(
            tool_name="get_swap_quote",
            args=_quote_request_args(quote),
            result=result,
            status="ok",
            session_id="s1",
        )
        out = mod._on_pre_tool_call(
            tool_name="execute_swap",
            args={"quoteData": quote},
            session_id="s1",
        )
        assert out["action"] == "block"
        assert "NOTIONAL_UNAVAILABLE" in out["message"]

    def test_quote_mutation_breaks_binding(self, _isolate_env):
        _write_config(_isolate_env, {"mode": "live"})
        mod = _load_plugin()
        quote = _swap_quote()
        mod._on_post_tool_call(
            tool_name="get_swap_quote",
            args=_quote_request_args(quote),
            result={"message": "ok", "data": quote},
            status="ok",
            session_id="s1",
        )
        changed = json.loads(json.dumps(quote))
        changed["transaction"]["to"] = "0x4444444444444444444444444444444444444444"
        out = mod._on_pre_tool_call(
            tool_name="execute_swap",
            args={"quoteData": changed},
            session_id="s1",
        )
        assert "not observed" in out["message"]

    def test_quote_is_session_bound(self, _isolate_env):
        _write_config(_isolate_env, {"mode": "live"})
        mod = _load_plugin()
        quote = _swap_quote()
        mod._on_post_tool_call(
            tool_name="get_swap_quote",
            args=_quote_request_args(quote),
            result={"message": "ok", "data": quote},
            status="ok",
            session_id="s1",
        )
        out = mod._on_pre_tool_call(
            tool_name="execute_swap",
            args={"quoteData": quote},
            session_id="s2",
        )
        assert "not observed" in out["message"]

    def test_gasless_quote_binds_to_gasless_write_only(self, _isolate_env):
        _write_config(_isolate_env, {"mode": "live"})
        mod = _load_plugin()
        _install_mandate()
        quote = _gasless_quote()
        mod._on_post_tool_call(
            tool_name="get_gasless_quote",
            args=_quote_request_args(quote),
            result={"message": "ok", "data": quote},
            status="ok",
            session_id="s1",
        )
        gasless = mod._on_pre_tool_call(
            tool_name="submit_gasless_swap",
            args={"quoteData": quote, "chainId": 8453},
            session_id="s1",
        )
        direct = mod._on_pre_tool_call(
            tool_name="execute_swap",
            args={"quoteData": quote},
            session_id="s1",
        )
        assert "NOTIONAL_UNAVAILABLE" in gasless["message"]
        assert "not observed" in direct["message"]

    def test_kill_switch_wins_over_live_mode(self, _isolate_env, monkeypatch):
        _write_config(_isolate_env, {"mode": "live"})
        monkeypatch.setenv("HERMES_TRADER_KILL_SWITCH", "1")
        mod = _load_plugin()
        out = mod._on_pre_tool_call(
            tool_name="execute_swap",
            args={"quoteData": {}},
            session_id="s1",
        )
        assert out["action"] == "block"
        assert "kill switch" in out["message"].lower()

    def test_unrelated_read_tool_passes(self, _isolate_env):
        _write_config(_isolate_env, {"mode": "paper"})
        mod = _load_plugin()
        assert mod._on_pre_tool_call(
            tool_name="get_swap_quote",
            args={"chainId": 8453},
            session_id="s1",
        ) is None

    def test_malformed_trader_config_fails_safe_to_paper(self, _isolate_env):
        (_isolate_env / "config.yaml").write_text(
            yaml.safe_dump({"trader": "not-a-mapping"}),
            encoding="utf-8",
        )
        mod = _load_plugin()
        out = mod._on_pre_tool_call(
            tool_name="execute_swap",
            args={"quoteData": {}},
            session_id="s1",
        )
        assert out["action"] == "block"
        assert "paper mode" in out["message"].lower()


class TestMcpContractFixture:
    def _contract(self):
        path = (
            _repo_root()
            / "tests"
            / "fixtures"
            / "defi_trading_mcp_2_1_3_contract.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_exact_version_and_source_commit_are_recorded(self):
        contract = self._contract()
        assert contract["version"] == "2.1.3"
        assert contract["source_commit"] == "5a5c47f6a34b93ee11cf701d17e171cea0f775ed"

    def test_pinned_aggregator_transport_is_plaintext_http(self):
        contract = self._contract()
        assert contract["aggregator"] == {
            "url": "http://44.252.136.98",
            "transport": "plaintext_http",
        }

    def test_write_tools_require_quote_data(self):
        contract = self._contract()
        assert contract["tools"]["execute_swap"]["required"] == ["quoteData"]
        assert contract["tools"]["submit_gasless_swap"]["required"] == ["quoteData"]

    def test_quote_tools_require_chain_and_token_amount_fields(self):
        contract = self._contract()
        expected = ["chainId", "buyToken", "sellToken", "sellAmount"]
        assert contract["tools"]["get_swap_quote"]["required"] == expected
        assert contract["tools"]["get_gasless_quote"]["required"] == expected


class TestMcpManifest:
    def _manifest(self):
        path = _repo_root() / "optional-mcps" / "defi-trading" / "manifest.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_exact_mcp_version_is_pinned(self):
        manifest = self._manifest()
        assert "defi-trading-mcp@2.1.3" in manifest["transport"]["args"]

    def test_aggregator_backed_tools_are_not_default_enabled(self):
        manifest = self._manifest()
        enabled = set(manifest["tools"]["default_enabled"])
        assert enabled.isdisjoint(AGGREGATOR_BACKED_TOOLS)

    def test_base_specific_trending_tool_is_default_enabled(self):
        manifest = self._manifest()
        assert "get_trending_pools_by_network" in manifest["tools"]["default_enabled"]

    def test_default_surface_does_not_request_wallet_credentials(self):
        manifest = self._manifest()
        names = {item["name"] for item in manifest["auth"]["env"]}
        assert names == {"COINGECKO_API_KEY"}


class TestPluginRegistration:
    def test_registers_tools_and_policy_hooks(self):
        mod = _load_plugin()
        hooks = []
        registered_tools = []

        class Ctx:
            def register_tool(self, **kwargs):
                registered_tools.append(kwargs)

            def register_hook(self, name, fn):
                hooks.append((name, fn))

        mod.register(Ctx())

        assert [item["name"] for item in registered_tools] == [
            "trader_uniswap_quote",
            "trader_uniswap_build_calls",
        ]
        assert [item["toolset"] for item in registered_tools] == [
            "trader",
            "trader",
        ]
        assert [name for name, _fn in hooks] == [
            "pre_tool_call",
            "post_tool_call",
        ]

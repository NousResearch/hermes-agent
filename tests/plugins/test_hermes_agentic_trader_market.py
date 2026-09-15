"""Market normalization tests for Hermes Agentic Trader."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_market():
    plugin_dir = _repo_root() / "plugins" / "hermes-agentic-trader"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    package_name = "hermes_plugins.hermes_agentic_trader"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(plugin_dir)]
        sys.modules[package_name] = pkg
    spec = importlib.util.spec_from_file_location(
        package_name + ".market",
        plugin_dir / "market.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[package_name + ".market"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fixture():
    path = _repo_root() / "tests" / "fixtures" / "coingecko_base_trending_sample.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_request_uses_network_specific_mcp_tool():
    market = _load_market()
    tool, args = market.build_trending_request("base")
    assert tool == "get_trending_pools_by_network"
    assert args["network"] == "base"
    assert args["include"] == "base_token,quote_token,dex"


def test_parser_keeps_pool_and_token_addresses_distinct():
    market = _load_market()
    pools = market.parse_trending_result(_fixture(), expected_network="base")
    assert len(pools) == 1
    pool = pools[0]
    assert pool.pool_address == "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert pool.base_token_address == "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    assert pool.quote_token_address == "0xcccccccccccccccccccccccccccccccccccccccc"
    assert pool.trade_target_address == pool.base_token_address
    assert pool.trade_target_address != pool.pool_address
    assert pool.base_symbol == "TOKEN"
    assert pool.quote_symbol == "WETH"
    assert pool.liquidity_usd == 250000.0
    assert pool.volume_24h_usd == 180000.0


def test_wrong_network_payload_is_rejected():
    market = _load_market()
    assert market.parse_trending_result(_fixture(), expected_network="eth") == []


def test_mcp_text_wrapper_is_unwrapped():
    market = _load_market()
    provider = _fixture()
    tool_service_result = {
        "message": "Trending pools for base retrieved successfully",
        "data": provider,
    }
    wrapped = json.dumps({"result": json.dumps(tool_service_result)})
    pools = market.parse_trending_result(wrapped, expected_network="base")
    assert len(pools) == 1


def test_token_address_falls_back_to_relationship_resource_id():
    market = _load_market()
    payload = _fixture()
    payload["included"] = []
    pools = market.parse_trending_result(payload, expected_network="base")
    assert len(pools) == 1
    assert pools[0].base_token_address == "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    assert pools[0].quote_token_address == "0xcccccccccccccccccccccccccccccccccccccccc"


def test_relationship_network_mismatch_is_rejected():
    market = _load_market()
    payload = _fixture()
    payload["data"][0]["relationships"]["network"] = {
        "data": {"id": "eth", "type": "network"}
    }
    assert market.parse_trending_result(payload, expected_network="base") == []

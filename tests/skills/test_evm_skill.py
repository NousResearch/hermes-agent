from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "evm"
    / "scripts"
    / "evm_client.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("evm_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_injective_chain_registry() -> None:
    mod = load_module()

    assert mod.CHAINS["injective"] == {
        "chain_id": 1776,
        "rpc": "https://sentry.evm-rpc.injective.network/",
        "native": "INJ",
        "coingecko": "injective-protocol",
        "explorer": "https://blockscout.injective.network",
        "decimals": 18,
    }


def test_injective_known_tokens_are_verified_mainnet_contracts() -> None:
    mod = load_module()

    assert mod.KNOWN_TOKENS["injective"] == {
        "USDC": "0x2a25fbD67b3aE485e461fe55d9DbeF302B7D3989",
        "WETH": "0x83A15000b753AC0EeE06D2Cb41a69e76D0D5c7F7",
    }
    assert "INJ" not in mod.KNOWN_TOKENS["injective"]


def test_inj_symbol_uses_native_coingecko_coin_id() -> None:
    mod = load_module()

    assert mod.COINGECKO_IDS["INJ"] == "injective-protocol"


def test_contract_price_uses_injective_asset_platform() -> None:
    mod = load_module()
    calls: list[str] = []
    usdc = mod.KNOWN_TOKENS["injective"]["USDC"]

    def fake_http_get(url: str, timeout: int = 20):
        calls.append(url)
        return {usdc.lower(): {"usd": 1.0}}

    with patch.object(mod, "_http_get", side_effect=fake_http_get):
        price = mod.cg_price_by_contract("injective", usdc)

    assert price == 1.0
    assert calls == [
        f"{mod.COINGECKO_BASE}/simple/token_price/injective"
        f"?contract_addresses={usdc}&vs_currencies=usd"
    ]


def test_parser_accepts_injective_chain() -> None:
    mod = load_module()

    args = mod.build_parser().parse_args(["stats", "--chain", "injective"])

    assert args.command == "stats"
    assert args.chain == "injective"


def test_stats_command_uses_injective_chain_without_live_network(capsys) -> None:
    mod = load_module()

    def fake_rpc_call(chain: str, method: str, params: list):
        assert chain == "injective"
        if method == "eth_getBlockByNumber":
            assert params == ["latest", False]
            return {
                "parentHash": "0xparent",
                "timestamp": "0x20",
                "transactions": ["0x1", "0x2", "0x3", "0x4"],
            }
        if method == "eth_getBlockByHash":
            assert params == ["0xparent", False]
            return {"timestamp": "0x10"}
        raise AssertionError(f"unexpected RPC method: {method}")

    with (
        patch.object(mod, "rpc_batch", return_value=["0x2a", "0x3b9aca00"]) as rpc_batch,
        patch.object(mod, "rpc_call", side_effect=fake_rpc_call),
        patch.object(mod, "get_native_price", return_value=7.5),
    ):
        mod.cmd_stats(argparse.Namespace(chain="injective"))

    rpc_batch.assert_called_once_with(
        "injective",
        [
            ("eth_blockNumber", []),
            ("eth_gasPrice", []),
        ],
    )
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["chain"] == "injective"
    assert rendered["block_number"] == 42
    assert rendered["gas_price_gwei"] == 1.0
    assert rendered["native_token"] == "INJ"
    assert rendered["native_price_usd"] == 7.5
    assert rendered["tps_estimate"] == 0.25
    assert rendered["explorer"] == "https://blockscout.injective.network"

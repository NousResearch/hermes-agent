from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "solana"
    / "scripts"
    / "solana_client.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("solana_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _token_account(mint: str, amount: str, *, decimals: int = 6) -> dict:
    return {
        "account": {
            "data": {
                "parsed": {
                    "info": {
                        "mint": mint,
                        "tokenAmount": {
                            "uiAmountString": amount,
                            "decimals": decimals,
                        },
                    }
                }
            }
        }
    }


def test_wallet_total_includes_priced_tokens_hidden_by_display_limit():
    mod = load_module()
    args = SimpleNamespace(address="wallet-1", all=False, limit=1, no_prices=False)

    token_rows = [
        _token_account("mint-a", "5"),
        _token_account("mint-b", "4"),
        _token_account("mint-c", "3"),
    ]

    def fake_rpc(method, _params=None):
        if method == "getBalance":
            return {"value": 2 * mod.LAMPORTS_PER_SOL}
        if method == "getTokenAccountsByOwner":
            return {"value": token_rows}
        raise AssertionError(f"Unexpected RPC method: {method}")

    with (
        patch.object(mod, "rpc", side_effect=fake_rpc),
        patch.object(mod, "fetch_sol_price", return_value=100.0),
        patch.object(
            mod,
            "fetch_prices",
            return_value={"mint-a": 10.0, "mint-b": 10.0, "mint-c": 10.0},
        ),
        patch.object(mod, "print_json") as mock_print_json,
    ):
        mod.cmd_wallet(args)

    output = mock_print_json.call_args.args[0]
    assert output["tokens_shown"] == 1
    assert output["tokens_hidden"] == 2
    assert output["spl_tokens"][0]["value_usd"] == 50.0
    assert output["portfolio_total_usd"] == 320.0

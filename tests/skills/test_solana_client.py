"""Offline tests for the optional ``solana`` skill helper script.

Every RPC / CoinGecko call is faked; nothing touches the network.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "solana"
    / "scripts"
    / "solana_client.py"
)

LAMPORTS = 1_000_000_000
V1_ERROR = {
    "code": -32015,
    "message": (
        "Transaction version (1) is not supported by the requesting client. "
        "Please try the request again with the following configuration parameter: "
        '"maxSupportedTransactionVersion": 1'
    ),
}


def load_module():
    spec = importlib.util.spec_from_file_location("solana_skill_client", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    module = load_module()
    with patch.object(module.time, "sleep", lambda *_: None):
        yield module


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(handler, calls=None):
    """Build a urlopen replacement; ``handler(url, body)`` returns the JSON reply."""

    def urlopen(req, timeout=None):
        body = json.loads(req.data) if req.data else None
        if calls is not None:
            calls.append(body)
        return _FakeResponse(json.dumps(handler(req.full_url, body)).encode())

    return urlopen


def _b58decode(value: str) -> bytes:
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    num = 0
    for char in value:
        num = num * 58 + alphabet.index(char)
    raw = num.to_bytes((num.bit_length() + 7) // 8, "big")
    return b"\0" * (len(value) - len(value.lstrip("1"))) + raw


# ---------------------------------------------------------------------------
# Known tokens / price lookup
# ---------------------------------------------------------------------------


def test_known_token_mints_are_valid_pubkeys(mod):
    for mint, (symbol, _name) in mod.KNOWN_TOKENS.items():
        assert len(_b58decode(mint)) == 32, f"{symbol} mint {mint} is not a 32-byte pubkey"


def test_pengu_and_wen_resolve_to_their_real_mints(mod):
    assert mod._SYMBOL_TO_MINT["PENGU"] == "2zMMhcVQEXDtdE6vsFS7S7D5oUodfJHE8vd1gnBouauv"
    assert mod._SYMBOL_TO_MINT["WEN"] == "WENWENvqqNya429ubCdR81ZmD69brwQaaBYY6p3LCpk"


def test_legacy_rndr_symbol_is_an_alias_for_render(mod):
    assert mod._SYMBOL_TO_MINT["RNDR"] == mod._SYMBOL_TO_MINT["RENDER"]


def test_price_rejects_unknown_symbol_without_querying_it_as_a_mint(mod):
    with patch.object(mod, "fetch_prices") as fetch_prices:
        with pytest.raises(SystemExit) as excinfo:
            mod.cmd_price(SimpleNamespace(token="RAY"))
    assert "Unknown symbol 'RAY'" in str(excinfo.value)
    fetch_prices.assert_not_called()


# ---------------------------------------------------------------------------
# Versioned transactions (-32015)
# ---------------------------------------------------------------------------


def test_requested_tx_version_is_parsed_from_rpc_error(mod):
    assert mod._requested_tx_version(V1_ERROR) == 1
    assert mod._requested_tx_version({"code": -32602, "message": "bad"}) is None


def test_whales_retries_get_block_with_the_requested_tx_version(mod, capsys):
    calls = []

    def handler(url, body):
        if "coingecko" in url:
            return {"solana": {"usd": 100.0}}
        if body["method"] == "getSlot":
            return {"result": 500}
        if body["params"][1]["maxSupportedTransactionVersion"] < 1:
            return {"error": V1_ERROR}
        return {"result": {"transactions": [{
            "meta": {"err": None, "preBalances": [700 * LAMPORTS, 0],
                     "postBalances": [100 * LAMPORTS, 600 * LAMPORTS]},
            "transaction": {"message": {"accountKeys": [{"pubkey": "A"}, {"pubkey": "B"}]}},
        }]}}

    with patch.object(mod.urllib.request, "urlopen", _fake_urlopen(handler, calls)):
        mod.cmd_whales(SimpleNamespace(min_sol=100))

    out = json.loads(capsys.readouterr().out)
    versions = [c["params"][1]["maxSupportedTransactionVersion"]
                for c in calls if c and c["method"] == "getBlock"]
    assert versions == [0, 1]
    assert out["large_transfers"] == [
        {"sender": "A", "receiver": "B", "amount_SOL": 600.0, "amount_USD": 60000.0}
    ]


def test_tx_retries_get_transaction_with_the_requested_tx_version(mod, capsys):
    def handler(url, body):
        if "coingecko" in url:
            return {"solana": {"usd": 100.0}}
        if body["params"][1]["maxSupportedTransactionVersion"] < 1:
            return {"error": V1_ERROR}
        return {"result": {
            "slot": 7, "blockTime": 1,
            "meta": {"err": None, "fee": 5000,
                     "preBalances": [LAMPORTS, 0], "postBalances": [LAMPORTS // 2 - 5000, LAMPORTS // 2]},
            "transaction": {"message": {
                "accountKeys": [{"pubkey": "A"}, {"pubkey": "B"}],
                "instructions": [{"programId": "11111111111111111111111111111111"}],
            }},
        }}

    with patch.object(mod.urllib.request, "urlopen", _fake_urlopen(handler)):
        mod.cmd_tx(SimpleNamespace(signature="sig"))

    out = json.loads(capsys.readouterr().out)
    assert out["slot"] == 7
    assert out["status"] == "success"


# ---------------------------------------------------------------------------
# Whale detector robustness
# ---------------------------------------------------------------------------


def test_whales_walks_back_when_latest_slot_was_skipped(mod, capsys):
    def handler(url, body):
        if "coingecko" in url:
            return {}
        if body["method"] == "getSlot":
            return {"result": 900}
        if body["params"][0] == 900:
            return {"error": {"code": -32007, "message": "Slot 900 was skipped"}}
        return {"result": {"transactions": []}}

    with patch.object(mod.urllib.request, "urlopen", _fake_urlopen(handler)):
        mod.cmd_whales(SimpleNamespace(min_sol=100))

    out = json.loads(capsys.readouterr().out)
    assert out["slot"] == 899
    assert out["large_transfers"] == []


def test_whales_attributes_sender_to_largest_balance_drop(mod, capsys):
    def handler(url, body):
        if "coingecko" in url:
            return {}
        if body["method"] == "getSlot":
            return {"result": 1}
        # Two large outflows in one tx: the 900 SOL receiver must not be
        # attributed to the first (smaller) sender just because it comes first.
        return {"result": {"transactions": [{
            "meta": {"err": None,
                     "preBalances": [600 * LAMPORTS, 900 * LAMPORTS, 0, 0],
                     "postBalances": [0, 0, 600 * LAMPORTS, 900 * LAMPORTS]},
            "transaction": {"message": {"accountKeys": ["small", "whale", "recv-600", "recv-900"]}},
        }]}}

    with patch.object(mod.urllib.request, "urlopen", _fake_urlopen(handler)):
        mod.cmd_whales(SimpleNamespace(min_sol=500))

    transfers = {t["receiver"]: t for t in json.loads(capsys.readouterr().out)["large_transfers"]}
    assert transfers["recv-900"]["sender"] == "whale"
    assert transfers["recv-900"]["amount_SOL"] == 900.0


# ---------------------------------------------------------------------------
# Input validation / provider quirks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "account_data",
    [
        ["", "base64"],  # system account (a wallet address)
        {"parsed": {"type": "account", "info": {"mint": "X"}}, "program": "spl-token"},  # token account
    ],
)
def test_token_rejects_addresses_that_are_not_mints(mod, account_data):
    def fake_rpc(method, _params=None):
        assert method == "getAccountInfo"
        return {"value": {"data": account_data}}

    with patch.object(mod, "rpc", side_effect=fake_rpc):
        with pytest.raises(SystemExit) as excinfo:
            mod.cmd_token(SimpleNamespace(mint="not-a-mint"))
    assert "not an SPL token mint" in str(excinfo.value)


def test_stats_falls_back_to_single_calls_when_batch_is_rejected(mod, capsys):
    results = {
        "getSlot": 42,
        "getEpochInfo": {"epoch": 3, "slotIndex": 4},
        "getRecentPerformanceSamples": [
            {"numTransactions": 600, "numNonVoteTransactions": 120, "samplePeriodSecs": 60}
        ],
        "getSupply": {"value": {"total": 10 * LAMPORTS, "circulating": 5 * LAMPORTS}},
        "getVersion": {"solana-core": "9.9.9"},
    }

    def handler(url, body):
        if "coingecko" in url:
            return {"solana": {"usd": 2.0}}
        if isinstance(body, list):
            return {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32600, "message": "batch requests are disabled"}}
        return {"result": results[body["method"]]}

    with patch.object(mod.urllib.request, "urlopen", _fake_urlopen(handler)):
        mod.cmd_stats(SimpleNamespace())

    out = json.loads(capsys.readouterr().out)
    assert out["slot"] == 42
    assert out["tps"] == 10.0
    assert out["tps_non_vote"] == 2.0
    assert out["validator_version"] == "9.9.9"


def test_wallet_warns_when_no_price_data_is_available(mod, capsys):
    def fake_rpc(method, _params=None):
        if method == "getBalance":
            return {"value": 2 * LAMPORTS}
        return {"value": [{"account": {"data": {"parsed": {"info": {
            "mint": "mint-a", "tokenAmount": {"uiAmountString": "5", "decimals": 6},
        }}}}}]}

    with (
        patch.object(mod, "rpc", side_effect=fake_rpc),
        patch.object(mod, "fetch_sol_price", return_value=None),
        patch.object(mod, "fetch_prices", return_value={}),
    ):
        mod.cmd_wallet(SimpleNamespace(address="wallet-1", all=False, limit=20, no_prices=False))

    out = json.loads(capsys.readouterr().out)
    assert "warning" in out
    assert out["spl_tokens"][0]["mint"] == "mint-a"


def test_activity_limit_is_clamped_to_at_least_one(mod, capsys):
    seen = {}

    def fake_rpc(method, params=None):
        seen["limit"] = params[1]["limit"]
        return []

    with patch.object(mod, "rpc", side_effect=fake_rpc):
        mod.cmd_activity(SimpleNamespace(address="wallet-1", limit=0))

    assert seen["limit"] == 1
    assert json.loads(capsys.readouterr().out)["transactions"] == []

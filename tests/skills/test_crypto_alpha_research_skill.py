from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "crypto-alpha-research"
    / "scripts"
    / "price.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("crypto_alpha_price", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pair(address, name, symbol, liquidity_usd):
    return {
        "chainId": "solana",
        "dexId": "raydium",
        "baseToken": {"address": address, "name": name, "symbol": symbol},
        "priceUsd": "1.5",
        "fdv": 1000000,
        "marketCap": 750000,
        "liquidity": {"usd": liquidity_usd},
        "volume": {"h24": 42.5},
        "priceChange": {"h24": -1.2},
        "pairAddress": "pair-" + address,
        "url": "https://dexscreener.com/solana/" + address,
    }


def test_token_facts_picks_highest_liquidity_pair():
    mod = load_module()
    payload = {
        "pairs": [
            _pair("addrLow", "Alpha", "ALP", 1000),
            _pair("addrHigh", "Alpha", "ALP", 90000),
        ]
    }

    with patch.object(mod, "_get", return_value=payload):
        facts = mod.token_facts("addrHigh")

    assert facts["found"] is True
    assert facts["address"] == "addrHigh"
    assert facts["liquidityUsd"] == 90000.0
    assert facts["pairsCount"] == 2
    assert [p["address"] for p in facts["otherPairs"]] == ["addrLow"]


def test_token_facts_without_pairs_reports_not_found():
    mod = load_module()

    with patch.object(mod, "_get", return_value={"pairs": []}):
        facts = mod.token_facts("0xmissing")

    assert facts["found"] is False
    assert facts["address"] == "0xmissing"
    assert "no DEX pairs" in facts["note"]


def test_search_flags_normalized_exact_matches_only():
    mod = load_module()
    payload = {
        "pairs": [
            _pair("addr1", "  PEPE ", "PePe", 50000),
            _pair("addr2", "Pepe Classic", "PEPEC", 40000),
            _pair("addr3", "Wrapped Thing", "pepe", 30000),
        ]
    }

    with patch.object(mod, "_get", return_value=payload):
        result = mod.search_name("pepe")

    by_addr = {c["address"]: c for c in result["candidates"]}
    assert by_addr["addr1"]["exactMatch"] is True
    assert by_addr["addr3"]["exactMatch"] is True
    assert by_addr["addr2"]["exactMatch"] is False
    assert result["exactMatchCount"] == 2
    assert result["candidatesCount"] == 3
    assert "fuzzy" in result["note"]
    # exact matches first, fuzzy hits after, liquidity order within each group
    assert [c["address"] for c in result["candidates"]] == ["addr1", "addr3", "addr2"]


def test_search_dedupes_addresses_and_caps_candidates():
    mod = load_module()
    pairs = [
        _pair("dup", "Thing", "THG", 99000),
        _pair("dup", "Thing", "THG", 98000),
    ]
    pairs += [_pair("addr%d" % i, "Thing %d" % i, "T%d" % i, 90000 - i) for i in range(20)]

    with patch.object(mod, "_get", return_value={"pairs": pairs}):
        result = mod.search_name("Thing")

    addresses = [c["address"] for c in result["candidates"]]
    assert len(addresses) == len(set(addresses)) == 15
    assert result["candidatesCount"] == 15
    assert addresses[0] == "dup"


def test_main_search_network_error_prints_json_error(capsys, monkeypatch):
    mod = load_module()

    def boom(url):
        raise urllib.error.URLError("dns down")

    monkeypatch.setattr(mod, "_get", boom)
    mod.main(["--search", "pepe"])

    rendered = json.loads(capsys.readouterr().out)
    assert rendered["error"].startswith("network error:")


def test_main_without_arguments_prints_usage_error(capsys):
    mod = load_module()

    mod.main([])

    rendered = json.loads(capsys.readouterr().out)
    assert "error" in rendered


def test_user_agent_carries_no_attribution_url():
    mod = load_module()
    assert "http" not in mod.UA.lower()

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "payments"
    / "x402-reader"
    / "scripts"
    / "quote.py"
)


def load_mod():
    spec = importlib.util.spec_from_file_location("x402_reader_quote", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prefers_solana_v1_network_name():
    mod = load_mod()
    body = {
        "accepts": [
            {
                "scheme": "exact",
                "network": "base",
                "maxAmountRequired": "5000",
                "payTo": "0x14df772BD496bBb7f49Bc3E992Ce13B2c441177F",
                "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            },
            {
                "scheme": "exact",
                "network": "solana",
                "maxAmountRequired": "5000",
                "payTo": "F1AbWuXJcBT9arW9wc6Xr2vom5NBtngWsz6Ht16jRBLM",
                "asset": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            },
        ]
    }
    rows = mod.quote_rows(body)
    chosen = mod.pick_row(rows)
    assert chosen["network"] == "solana"
    assert chosen["amount_usdc"] == 0.005


def test_accepts_caip_solana_network():
    mod = load_mod()
    rows = mod.quote_rows(
        {
            "accepts": [
                {
                    "network": "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp",
                    "amount": "50000",
                    "payTo": "F1AbWuXJcBT9arW9wc6Xr2vom5NBtngWsz6Ht16jRBLM",
                }
            ]
        }
    )
    chosen = mod.pick_row(rows)
    assert chosen["preferred"] is True
    assert chosen["amount_usdc"] == 0.05


def test_empty_accepts_is_not_ok(capsys):
    mod = load_mod()
    old = sys.stdin
    sys.stdin = type("S", (), {"read": lambda self: json.dumps({"accepts": []})})()
    try:
        code = mod.main([])
    finally:
        sys.stdin = old
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["ok"] is False
    assert out["accepts"] == []


def test_strips_curl_header_prefix():
    mod = load_mod()
    raw = (
        "HTTP/2 402\r\n"
        "payment-required: ignore\r\n"
        "content-type: application/json\r\n"
        "\r\n"
        + json.dumps(
            {
                "accepts": [
                    {
                        "network": "solana",
                        "maxAmountRequired": "5000",
                        "payTo": "F1AbWuXJcBT9arW9wc6Xr2vom5NBtngWsz6Ht16jRBLM",
                    }
                ]
            }
        )
    )
    body = mod.load_body(raw)
    chosen = mod.pick_row(mod.quote_rows(body))
    assert chosen["amount_usdc"] == 0.005


def test_main_prints_json(capsys):
    mod = load_mod()
    body = json.dumps(
        {
            "accepts": [
                {
                    "network": "solana",
                    "maxAmountRequired": "5000",
                    "payTo": "F1AbWuXJcBT9arW9wc6Xr2vom5NBtngWsz6Ht16jRBLM",
                }
            ]
        }
    )
    old = sys.stdin
    sys.stdin = type("S", (), {"read": lambda self: body})()
    try:
        code = mod.main([])
    finally:
        sys.stdin = old
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["ok"] is True
    assert out["quote"]["amount_usdc"] == 0.005

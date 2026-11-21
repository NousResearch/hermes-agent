"""Tests for optional-skills/autonomous-ai-agents/brainchain/scripts/wallet_check.py."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "optional-skills" / "autonomous-ai-agents" / "brainchain" / "scripts" / "wallet_check.py"
spec = importlib.util.spec_from_file_location("brainchain_wallet_check", SCRIPT)
wallet_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wallet_check)

ADDRESS = "0x" + "ab" * 20


def chain(balance: int, allowance: int = 0, eth: int = 0):
    calls = []

    def call(url, method, params):
        calls.append((url, method, params))
        if method == "eth_getBalance":
            return hex(eth)
        data = params[0]["data"]
        if data.startswith(wallet_check.BALANCE_OF):
            return hex(balance)
        if data.startswith(wallet_check.ALLOWANCE):
            return hex(allowance)
        raise AssertionError(f"unexpected call {data[:10]}")

    return call, calls


def test_funded_and_approved_usdg_wallet_is_ready():
    call, calls = chain(balance=5_000_000, allowance=100_000_000)
    report = wallet_check.check(ADDRESS, "mainnet", "robinhood", 3, call=call)
    assert report["ready"] is True
    assert report["balance"] == 5 and report["needed"] == 3
    assert calls[0][0] == "https://rpc.mainnet.chain.robinhood.com"
    # balanceOf(address) carries the address as one left-padded word.
    assert calls[0][2][0]["data"] == wallet_check.BALANCE_OF + "0" * 24 + "ab" * 20


def test_short_balance_and_missing_approval_are_both_reported():
    call, _ = chain(balance=500_000, allowance=0, eth=0)
    report = wallet_check.check(ADDRESS, "mainnet", "robinhood", 1, call=call)
    assert report["ready"] is False
    assert len(report["problems"]) == 3
    assert "approve" in report["problems"][1]
    assert "no ETH" in report["problems"][2]


def test_base_needs_no_allowance_or_eth():
    call, calls = chain(balance=1_000_000)
    report = wallet_check.check(ADDRESS, "testnet", "base", 1, call=call)
    assert report["ready"] is True
    assert report["chain"] == "Base Sepolia"
    assert [method for _, method, _ in calls] == ["eth_call"]


def test_cli_refuses_anything_but_an_address(capsys):
    assert wallet_check.main(["0x" + "11" * 32]) == 2
    assert "never its key" in json.loads(capsys.readouterr().out)["error"]


def test_cli_exit_codes_follow_readiness(monkeypatch, capsys):
    call, _ = chain(balance=0)
    monkeypatch.setattr(wallet_check, "rpc", call)
    assert wallet_check.main([ADDRESS, "--pay-on", "base"]) == 1
    assert json.loads(capsys.readouterr().out)["ready"] is False

    def unreachable(*_):
        raise OSError("connection refused")

    monkeypatch.setattr(wallet_check, "rpc", unreachable)
    assert wallet_check.main([ADDRESS]) == 2
    assert "Could not read the chain" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.parametrize("rail", sorted(wallet_check.RAILS))
def test_every_rail_names_a_chain_and_token(rail):
    entry = wallet_check.RAILS[rail]
    assert entry["token"] in {"USDG", "USDC"}
    assert entry["rpc"].startswith("https://")

"""Direct Base Uniswap V3 adapter tests."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_module(name: str):
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
    full_name = package_name + "." + name
    spec = importlib.util.spec_from_file_location(
        full_name,
        plugin_dir / f"{name}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def u():
    module = _load_module("uniswap_v3")
    module.clear_direct_quotes()
    return module


TOKEN_IN = "0x1111111111111111111111111111111111111111"
TOKEN_OUT = "0x2222222222222222222222222222222222222222"
OWNER = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_official_base_deployments_are_pinned(u):
    assert u.BASE_CHAIN_ID == 8453
    assert u.BASE_CHAIN_HEX == "0x2105"
    assert u.SWAP_ROUTER_02 == "0x2626664c2603336e57b271c5c0b26f421741e481"
    assert u.QUOTER_V2 == "0x3d4e44eb1374240ce5f1b871ab261cd16335b76a"
    assert (
        u.UNISWAP_DEPLOYMENT_SOURCE_COMMIT
        == "e34ba78b05663c8c342cce69b0039067ace75691"
    )


def test_quote_calldata_is_local_and_static(u):
    data = u.quote_calldata(
        token_in=TOKEN_IN,
        token_out=TOKEN_OUT,
        amount_in=1_000_000,
        fee=500,
    )
    assert data.startswith("0xc6a5026a")
    assert len(data) == 2 + 8 + 64 * 5
    assert TOKEN_IN[2:] in data
    assert TOKEN_OUT[2:] in data


def test_decode_quoter_v2_result(u):
    raw = "0x" + "".join(
        f"{value:064x}"
        for value in [2_000_000, 12345, 7, 180000]
    )
    decoded = u.decode_quote_result(raw)
    assert decoded == {
        "amount_out": 2_000_000,
        "sqrt_price_x96_after": 12345,
        "initialized_ticks_crossed": 7,
        "gas_estimate": 180000,
    }


def test_direct_quote_checks_base_chain_and_binds_session(u, monkeypatch):
    raw = "0x" + "".join(
        f"{value:064x}"
        for value in [2_000_000, 12345, 7, 180000]
    )
    responses = iter(["0x2105", raw])
    monkeypatch.setattr(u, "rpc_call", lambda *a, **k: next(responses))
    monkeypatch.setattr(u.secrets, "token_hex", lambda n: "00" * n)

    record = u.fetch_direct_quote(
        token_in=TOKEN_IN,
        token_out=TOKEN_OUT,
        amount_in="1000000",
        fee=500,
        session_id="s1",
        now=10.0,
    )
    assert record.amount_out == 2_000_000
    assert u.resolve_direct_quote(
        record.quote_id,
        session_id="s1",
        now=10.0,
    ) == record
    assert u.resolve_direct_quote(
        record.quote_id,
        session_id="s2",
        now=10.0,
    ) is None


def test_direct_quote_rejects_wrong_chain(u, monkeypatch):
    monkeypatch.setattr(u, "rpc_call", lambda *a, **k: "0x1")
    with pytest.raises(RuntimeError, match="not Base mainnet"):
        u.fetch_direct_quote(
            token_in=TOKEN_IN,
            token_out=TOKEN_OUT,
            amount_in=1_000_000,
            fee=500,
            session_id="s1",
        )


def test_expired_quote_cannot_build(u, monkeypatch):
    raw = "0x" + "".join(
        f"{value:064x}"
        for value in [2_000_000, 12345, 7, 180000]
    )
    responses = iter(["0x2105", raw])
    monkeypatch.setattr(u, "rpc_call", lambda *a, **k: next(responses))
    record = u.fetch_direct_quote(
        token_in=TOKEN_IN,
        token_out=TOKEN_OUT,
        amount_in=1_000_000,
        fee=500,
        session_id="s1",
        now=10.0,
    )
    assert u.resolve_direct_quote(
        record.quote_id,
        session_id="s1",
        now=10.0 + u.QUOTE_TTL_SECONDS + 1,
    ) is None


def test_wallet_bundle_is_unsigned_atomic_and_allowlisted(u):
    record = u.DirectQuote(
        quote_id="q1",
        session_id="s1",
        token_in=TOKEN_IN,
        token_out=TOKEN_OUT,
        amount_in=1_000_000,
        fee=500,
        amount_out=2_000_000,
        sqrt_price_x96_after=12345,
        initialized_ticks_crossed=7,
        gas_estimate=180000,
        captured_at=10.0,
    )
    payload = u.build_wallet_send_calls(
        owner=OWNER,
        quote=record,
        slippage_bps=50,
    )
    request = payload["params"][0]
    assert payload["method"] == "wallet_sendCalls"
    assert request["version"] == "2.0.0"
    assert request["chainId"] == "0x2105"
    assert request["atomicRequired"] is True
    assert request["from"] == OWNER

    approval, swap = request["calls"]
    assert approval["to"] == TOKEN_IN
    assert approval["data"].startswith("0x095ea7b3")
    assert swap["to"] == u.SWAP_ROUTER_02
    assert swap["data"].startswith("0x04e45aaf")
    assert payload["quote"]["amount_out_minimum"] == "1990000"

    serialized = json.dumps(payload).lower()
    assert "private_key" not in serialized
    assert "user_private_key" not in serialized


def test_wallet_capability_preflight_uses_eip5792_shape(u):
    request = u.build_wallet_capabilities_request(OWNER)
    assert request == {
        "method": "wallet_getCapabilities",
        "params": [OWNER, ["0x2105"]],
    }


@pytest.mark.parametrize("fee", [1, 250, 1000, 2500])
def test_unsupported_fee_is_rejected(u, fee):
    with pytest.raises(ValueError, match="fee must be one of"):
        u.quote_calldata(
            token_in=TOKEN_IN,
            token_out=TOKEN_OUT,
            amount_in=1_000_000,
            fee=fee,
        )

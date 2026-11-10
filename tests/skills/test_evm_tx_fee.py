from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "evm"
    / "scripts"
    / "evm_client.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("evm_skill_tx_fee", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_tx(monkeypatch, capsys, receipt_extra, chain):
    mod = load_module()
    tx = {
        "blockNumber": "0x10",
        "value": "0x0",
        "gasPrice": hex(10**9),
        "gas": "0x5208",
        "input": "0x",
        "nonce": "0x1",
    }
    receipt = {"gasUsed": "0x5208", "status": "0x1", **receipt_extra}
    monkeypatch.setattr(mod, "rpc_batch", lambda *_args, **_kwargs: [tx, receipt])
    monkeypatch.setattr(mod, "rpc_call", lambda *_args, **_kwargs: {"timestamp": "0x1"})
    monkeypatch.setattr(mod, "get_native_price", lambda _chain: None)
    args = mod.build_parser().parse_args(["tx", "0x" + "ab" * 32, "--chain", chain])
    mod.cmd_tx(args)
    return json.loads(capsys.readouterr().out)


def test_tx_fee_includes_op_stack_l1_data_fee(monkeypatch, capsys):
    # 21000 gas at 1 gwei, plus a 0.0001 ETH L1 data fee reported in the receipt.
    out = run_tx(monkeypatch, capsys, {"l1Fee": hex(10**14)}, chain="base")

    assert out["fee_native"] == round((21000 * 10**9 + 10**14) / 10**18, 8)


def test_tx_fee_uses_receipt_effective_gas_price(monkeypatch, capsys):
    out = run_tx(monkeypatch, capsys, {"effectiveGasPrice": hex(2 * 10**9)}, chain="ethereum")

    assert out["gas_price_gwei"] == 2.0
    assert out["fee_native"] == round(21000 * 2 * 10**9 / 10**18, 8)


def test_tx_fee_without_receipt_extras_is_unchanged(monkeypatch, capsys):
    out = run_tx(monkeypatch, capsys, {}, chain="ethereum")

    assert out["gas_price_gwei"] == 1.0
    assert out["fee_native"] == round(21000 * 10**9 / 10**18, 8)

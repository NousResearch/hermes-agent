"""Tests for optional-skills/blockchain/evm (whitelist raw RPC extension)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "evm"
    / "scripts"
    / "evm_client.py"
)


@pytest.fixture(scope="module")
def evm():
    spec = importlib.util.spec_from_file_location("evm_client_under_test", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_redact_rpc_url_strips_query_userinfo_and_path(evm):
    redacted = evm.redact_rpc_url(
        "https://user:secret@rpc.example.com/v1/xyz?apikey=abc123#frag"
    )
    assert redacted == "https://***@rpc.example.com"
    assert "secret" not in redacted
    assert "abc123" not in redacted
    assert "/v1" not in redacted
    assert "frag" not in redacted


def test_rpc_rejects_disallowed_method(evm, capsys):
    args = argparse.Namespace(method="eth_sendRawTransaction", params="[]", chain="ethereum")
    with mock.patch.object(evm, "get_rpc_url", return_value="https://key@rpc.example/v1"):
        with pytest.raises(SystemExit) as exited:
            evm.cmd_rpc(args)
    assert exited.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "disallowed" in payload["error"]
    assert "key@" not in payload["endpoint"]


def test_rpc_allows_chain_id(evm, capsys):
    args = argparse.Namespace(method="eth_chainId", params="[]", chain="ethereum")
    with mock.patch.object(evm, "get_rpc_url", return_value="https://rpc.example/v1?token=xyz"):
        with mock.patch.object(evm, "rpc_call", return_value="0x1") as mocked:
            evm.cmd_rpc(args)
            mocked.assert_called_once_with("ethereum", "eth_chainId", [])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["result"] == "0x1"
    assert "token=xyz" not in payload["endpoint"]


def test_rpc_eth_call_rejects_oversize_data(evm, capsys):
    huge = "0x" + ("ab" * (evm._ETH_CALL_DATA_MAX_HEX_CHARS + 1))
    args = argparse.Namespace(
        method="eth_call",
        params=json.dumps([{"to": "0x" + "11" * 20, "data": huge}, "latest"]),
        chain="ethereum",
    )
    with mock.patch.object(evm, "get_rpc_url", return_value="https://rpc.example/v1"):
        with pytest.raises(SystemExit) as exited:
            evm.cmd_rpc(args)
    assert exited.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "maximum hex length" in payload["error"]

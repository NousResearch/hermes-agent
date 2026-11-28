from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "utxo"
    / "scripts"
    / "utxo_client.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("utxo_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ok(data):
    return _Response(json.dumps({"status": "success", "data": data}).encode("utf-8"))


def _http_error(code, body=None, headers=None):
    msg = Message()
    for key, value in (headers or {}).items():
        msg[key] = value
    payload = io.BytesIO(json.dumps(body or {}).encode("utf-8"))
    return urllib.error.HTTPError("https://sochain.com/api/v3/x", code, "error", msg, payload)


@pytest.mark.parametrize(
    ("address", "network"),
    [
        ("bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh", "BTC"),
        ("BC1QXY2KGDYGJRSQTZQ2N0YRF2493P83KKFJHX0WLH", "BTC"),
        ("1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "BTC"),
        ("ltc1qg82tzqyx0s5xz2dvdtnq5drj7ynd2ekmv0x9e2", "LTC"),
        ("LZ1ESUXcGqTJGKBNVHSE5dW3avSfH6SX8q", "LTC"),
        ("DLCDJhnh6aGotar6b182jpzbNEyXb3C361", "DOGE"),
        ("t1Ma2z4QAVqfxfkkiw6XNM3WC7VJkYz6S5G", "ZEC"),
        ("tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx", "BTCTEST"),
        ("tltc1qw508d6qejxtdg4y5r3zarvary0c5xw7k5kg0g0", "LTCTEST"),
    ],
)
def test_infer_network_from_prefix(address, network):
    mod = load_module()
    assert mod.infer_network(address) == network


def test_ambiguous_testnet_prefix_asks_for_network():
    mod = load_module()
    with pytest.raises(mod.SoChainError, match="--network"):
        mod.infer_network("mipcBbFg9gMiCh81Kj8tqqdgoZub1ZJRfn")


def test_keyless_request_sends_no_api_key(monkeypatch):
    mod = load_module()
    monkeypatch.delenv("SOCHAIN_API_KEY", raising=False)
    with patch.object(mod.urllib.request, "urlopen", return_value=_ok({"confirmed": "1.0"})) as urlopen:
        assert mod._get("/balance/BTC/addr") == {"confirmed": "1.0"}
    request = urlopen.call_args.args[0]
    assert request.full_url == "https://sochain.com/api/v3/balance/BTC/addr"
    assert request.get_header("Api-key") is None


def test_api_key_is_sent_as_api_key_header(monkeypatch):
    mod = load_module()
    monkeypatch.setenv("SOCHAIN_API_KEY", "  test-key  ")
    with patch.object(mod.urllib.request, "urlopen", return_value=_ok({})) as urlopen:
        mod._get("/network_info/BTC")
    assert urlopen.call_args.args[0].get_header("Api-key") == "test-key"


def test_fail_status_raises_with_error_message():
    mod = load_module()
    body = _Response(json.dumps({"status": "fail", "data": {"error_message": "Invalid address."}}).encode())
    with patch.object(mod.urllib.request, "urlopen", return_value=body):
        with pytest.raises(mod.SoChainError, match="Invalid address"):
            mod._get("/balance/BTC/nope")


def test_rate_limit_retries_after_retry_after_then_succeeds():
    mod = load_module()
    responses = [_http_error(429, headers={"Retry-After": "3"}), _ok({"block_count": 1})]

    def fake_urlopen(request, timeout):
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    with patch.object(mod.urllib.request, "urlopen", side_effect=fake_urlopen), \
            patch.object(mod.time, "sleep") as sleep:
        assert mod._get("/network_info/BTC") == {"block_count": 1}
    sleep.assert_called_once_with(3.0)


def test_rate_limit_exhausted_points_to_api_key(monkeypatch):
    mod = load_module()
    with patch.object(mod.urllib.request, "urlopen", side_effect=_http_error(429)), \
            patch.object(mod.time, "sleep"):
        with pytest.raises(mod.SoChainError, match="SOCHAIN_API_KEY"):
            mod._get("/network_info/BTC")


def test_bad_key_names_the_env_var():
    mod = load_module()
    with patch.object(mod.urllib.request, "urlopen", side_effect=_http_error(401)):
        with pytest.raises(mod.SoChainError, match="SOCHAIN_API_KEY"):
            mod._get("/account_info")


def test_main_balance_at_date_prints_data(capsys):
    mod = load_module()
    with patch.object(mod, "_get", return_value={"confirmed": "12.5"}) as get:
        exit_code = mod.main(["balance", "DLCDJhnh6aGotar6b182jpzbNEyXb3C361", "--time", "2025-12-31"])
    assert exit_code == 0
    get.assert_called_once_with(
        "/balance/DOGE/DLCDJhnh6aGotar6b182jpzbNEyXb3C361", {"time": "2025-12-31", "height": None}
    )
    assert json.loads(capsys.readouterr().out) == {"confirmed": "12.5"}


def test_main_filtered_history_uses_query_not_page(capsys):
    mod = load_module()
    with patch.object(mod, "_get", return_value={"transactions": []}) as get:
        assert mod.main(["txs", "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "--direction", "in"]) == 0
    path, params = get.call_args.args
    assert path == "/transactions/BTC/1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2"
    assert params["direction"] == "in"

    assert mod.main(["txs", "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2", "--direction", "in", "--page", "2"]) == 1
    assert "--cursor" in capsys.readouterr().err


def test_main_unknown_network_is_an_error(capsys):
    mod = load_module()
    assert mod.main(["tx", "ab" * 32, "--network", "ETH"]) == 1
    assert "unknown network" in capsys.readouterr().err

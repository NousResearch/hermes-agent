from __future__ import annotations

import base64
import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "hedera"
    / "scripts"
    / "hedera_client.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("hedera_skill", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BLOCKS_FIXTURE = {
    "blocks": [{"number": 12345678, "consensus_end_timestamp": "1751234567.000000000"}]
}
SUPPLY_FIXTURE = {
    "released_supply": "381223410000000000",
    "total_supply": "500000000000000000",
}
NODES_FIXTURE = {"nodes": [{}] * 39}
EXCHANGE_RATE_FIXTURE = {
    "current_rate": {
        "cent_equivalent": 12,
        "hbar_equivalent": 1,
        "expiration_time": "1751234567",
    }
}
ACCOUNT_FIXTURE = {
    "account": "0.0.12345",
    "evm_address": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
    "memo": "",
    "receiver_sig_required": False,
    "balance": {
        "balance": 500_000_000,  # 5 HBAR
        "tokens": [
            {"token_id": "0.0.999", "balance": 1000},
            {"token_id": "0.0.888", "balance": 500},
        ],
    },
}
TOKEN_999_FIXTURE = {
    "token_id": "0.0.999",
    "name": "Alpha Token",
    "symbol": "ALPHA",
    "decimals": "2",
}
TOKEN_888_FIXTURE = {
    "token_id": "0.0.888",
    "name": "Beta Token",
    "symbol": "BETA",
    "decimals": "0",
}
FUNGIBLE_TOKEN_FIXTURE = {
    "token_id": "0.0.5678",
    "name": "My Token",
    "symbol": "MTK",
    "type": "FUNGIBLE_COMMON",
    "decimals": "6",
    "total_supply": "1000000000000",
    "initial_supply": "500000000000",
    "max_supply": "0",
    "supply_type": "INFINITE",
    "treasury_account_id": "0.0.100",
    "memo": "",
    "freeze_default": False,
    "admin_key": {"key": "abc123"},
    "supply_key": None,
    "freeze_key": None,
    "kyc_key": None,
    "wipe_key": None,
    "pause_key": {"key": "def456"},
    "pause_status": "NOT_APPLICABLE",
    "custom_fees": {"fixed_fees": [], "fractional_fees": []},
    "created_timestamp": "1234567890.000000000",
}
TX_FIXTURE = {
    "transactions": [
        {
            "transaction_id": "0.0.1234-1234567890-000000000",
            "name": "CRYPTOTRANSFER",
            "result": "SUCCESS",
            "consensus_timestamp": "1234567890.000000000",
            "valid_start_timestamp": "1234567880.000000000",
            "charged_tx_fee": "10000",
            "memo_base64": "",
            "transfers": [
                {"account": "0.0.1234", "amount": -110000},
                {"account": "0.0.5678", "amount": 100000},
                {"account": "0.0.98",   "amount": 10000},
            ],
            "token_transfers": [],
            "nft_transfers": [],
        }
    ]
}
TOPIC_FIXTURE = {
    "topic_id": "0.0.1001",
    "memo": "test topic",
    "admin_key": {"key": "abc"},
    "submit_key": None,
    "auto_renew_period": 7776000,
    "auto_renew_account": "0.0.100",
    "created_timestamp": "1234567890.000000000",
    "deleted": False,
}
_MSG_TEXT = b"hello from HCS"
_MSG_B64 = base64.b64encode(_MSG_TEXT).decode()
MESSAGES_FIXTURE = {
    "messages": [
        {
            "consensus_timestamp": "1234567900.000000000",
            "sequence_number": 42,
            "message": _MSG_B64,
            "running_hash": "abcdef0123456789abcdef0123456789",
            "topic_id": "0.0.1001",
        }
    ]
}
CONTRACT_FIXTURE = {
    "contract_id": "0.0.5678",
    "evm_address": "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
    "memo": "my contract",
    "admin_key": {"key": "xyz"},
    "auto_renew_account": "0.0.100",
    "auto_renew_period": 7776000,
    "bytecode": "0x" + "60" * 100,  # 100 bytes
    "balance": 50_000_000,
    "created_timestamp": "1234567890.000000000",
}
ACTIVITY_FIXTURE = {
    "transactions": [
        {
            "transaction_id": f"0.0.1-{1000 + i}-0",
            "name": "CRYPTOTRANSFER",
            "result": "SUCCESS",
            "consensus_timestamp": f"{1000 + i}.0",
            "charged_tx_fee": "10000",
        }
        for i in range(5)
    ]
}
_IPFS_URI = b"ipfs://QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco"
_IPFS_B64 = base64.b64encode(_IPFS_URI).decode()
_BINARY_RAW = bytes([0x80, 0x81])
_BINARY_B64 = base64.b64encode(_BINARY_RAW).decode()
NFT_FIXTURE = {
    "nfts": [
        {"token_id": "0.0.100", "serial_number": 1, "metadata": _IPFS_B64, "created_timestamp": "1234.0"},
        {"token_id": "0.0.100", "serial_number": 2, "metadata": _BINARY_B64, "created_timestamp": "1235.0"},
        {"token_id": "0.0.200", "serial_number": 1, "metadata": _IPFS_B64, "created_timestamp": "1236.0"},
    ]
}


# ===========================================================================
# Group A — Input validation
# ===========================================================================


def test_valid_account_id_dotted():
    mod = load_module()
    for valid in ["0.0.1234", "0.0.0", "1.2.3", "0.0.4294967296"]:
        assert mod.require_account_id(valid) == valid


def test_valid_account_id_evm_address():
    mod = load_module()
    addr = "0x1234567890abcdef1234567890abcdef12345678"
    assert mod.require_account_id(addr) == addr


def test_invalid_account_id_exits_2():
    mod = load_module()
    for bad in ["1234", "0.0", "abc", "0.0.abc", "", "0.0.1.2"]:
        with pytest.raises(SystemExit) as exc:
            mod.require_account_id(bad)
        assert exc.value.code == 2


def test_valid_topic_id():
    mod = load_module()
    assert mod.require_topic_id("0.0.5678") == "0.0.5678"


def test_invalid_topic_id_exits_2():
    mod = load_module()
    for bad in ["notanid", "0.0", "abc123"]:
        with pytest.raises(SystemExit) as exc:
            mod.require_topic_id(bad)
        assert exc.value.code == 2


def test_tx_id_canonical_form_accepted():
    mod = load_module()
    tx = "0.0.1234-1234567890-000000000"
    assert mod.require_tx_id(tx) == tx


def test_tx_id_at_form_normalized_to_canonical():
    mod = load_module()
    result = mod.require_tx_id("0.0.1234@1234567890.000000000")
    assert result == "0.0.1234-1234567890-000000000"


def test_invalid_tx_id_exits_2():
    mod = load_module()
    for bad in ["abc123", "0.0.1234", "0.0.1234-only-two", ""]:
        with pytest.raises(SystemExit) as exc:
            mod.require_tx_id(bad)
        assert exc.value.code == 2


def test_valid_contract_id_dotted():
    mod = load_module()
    assert mod.require_contract_id("0.0.5678") == "0.0.5678"


def test_valid_contract_id_evm():
    mod = load_module()
    addr = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
    assert mod.require_contract_id(addr) == addr


def test_invalid_contract_id_exits_2():
    mod = load_module()
    with pytest.raises(SystemExit) as exc:
        mod.require_contract_id("notvalid")
    assert exc.value.code == 2


# ===========================================================================
# Group B — Unit conversions
# ===========================================================================


def test_tinybar_to_hbar_one_hbar():
    mod = load_module()
    assert mod.tinybar_to_hbar(100_000_000) == 1.0


def test_tinybar_to_hbar_zero():
    mod = load_module()
    assert mod.tinybar_to_hbar(0) == 0.0


def test_tinybar_to_hbar_large():
    mod = load_module()
    assert mod.tinybar_to_hbar(5_000_000_000_000) == 50_000.0


def test_decode_b64_utf8_string():
    mod = load_module()
    encoded = base64.b64encode(b"hello hedera").decode()
    assert mod.decode_b64(encoded) == "hello hedera"


def test_decode_b64_ipfs_uri():
    mod = load_module()
    uri = b"ipfs://QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco"
    encoded = base64.b64encode(uri).decode()
    assert mod.decode_b64(encoded) == uri.decode()


def test_decode_b64_binary_returns_hex():
    mod = load_module()
    raw = bytes([0x80, 0x81, 0x82])
    encoded = base64.b64encode(raw).decode()
    assert mod.decode_b64(encoded) == raw.hex()


def test_decode_b64_empty_string():
    mod = load_module()
    assert mod.decode_b64("") == ""


# ===========================================================================
# Group C — Network selection
# ===========================================================================


def test_default_network_is_mainnet():
    mod = load_module()
    assert mod._NETWORK == "mainnet"
    assert "mainnet" in mod._mirror_base()


def test_env_var_overrides_network(monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HEDERA_MIRROR_URL", "https://custom.mirror.example.com")
    assert mod._mirror_base() == "https://custom.mirror.example.com"


def test_env_var_strips_trailing_slash(monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HEDERA_MIRROR_URL", "https://custom.example.com/")
    assert mod._mirror_base() == "https://custom.example.com"


def test_testnet_flag_routes_to_testnet_url(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    seen_bases: list = []

    def fake_http(path, params=None, **kw):
        seen_bases.append(mod._mirror_base())
        if "blocks" in path:
            return BLOCKS_FIXTURE
        if "supply" in path:
            return SUPPLY_FIXTURE
        if "nodes" in path:
            return NODES_FIXTURE
        return None

    with patch.object(mod, "_http_get", side_effect=fake_http):
        with patch.object(mod, "fetch_hbar_price", return_value=None):
            mod.main(["--network", "testnet", "stats"])

    assert all("testnet" in b for b in seen_bases)
    assert not any("mainnet" in b for b in seen_bases)


# ===========================================================================
# Group D — stats command
# ===========================================================================


def test_stats_output_shape(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    def fake_http(path, params=None, **kw):
        if "blocks" in path:
            return BLOCKS_FIXTURE
        if "supply" in path:
            return SUPPLY_FIXTURE
        if "nodes" in path:
            return NODES_FIXTURE
        return None

    with patch.object(mod, "_http_get", side_effect=fake_http):
        with patch.object(mod, "fetch_hbar_price", return_value=0.0812):
            exit_code = mod.main(["stats"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["latest_block"] == 12345678
    assert out["hbar_price_usd"] == pytest.approx(0.0812)
    assert out["node_count"] == 39
    assert out["released_supply_hbar"] > 0
    assert out["total_supply_hbar"] > out["released_supply_hbar"]
    assert "market_cap_usd" in out


def test_stats_omits_market_cap_when_no_price(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    def fake_http(path, params=None, **kw):
        if "blocks" in path:
            return BLOCKS_FIXTURE
        if "supply" in path:
            return SUPPLY_FIXTURE
        if "nodes" in path:
            return NODES_FIXTURE
        return None

    with patch.object(mod, "_http_get", side_effect=fake_http):
        with patch.object(mod, "fetch_hbar_price", return_value=None):
            exit_code = mod.main(["stats"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out.get("hbar_price_usd") is None
    assert "market_cap_usd" not in out


# ===========================================================================
# Group E — account command
# ===========================================================================


def _account_http(path, params=None, **kw):
    if "/accounts/0.0.12345" in path and "nfts" not in path:
        return ACCOUNT_FIXTURE
    if "/tokens/0.0.999" in path:
        return TOKEN_999_FIXTURE
    if "/tokens/0.0.888" in path:
        return TOKEN_888_FIXTURE
    return None


def test_account_hbar_balance_converted(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", side_effect=_account_http):
        with patch.object(mod, "fetch_hbar_price", return_value=0.08):
            exit_code = mod.main(["account", "0.0.12345"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["hbar_balance"] == 5.0
    assert out["hbar_value_usd"] == pytest.approx(0.40, rel=1e-4)


def test_account_token_list_decimals_applied(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", side_effect=_account_http):
        with patch.object(mod, "fetch_hbar_price", return_value=None):
            exit_code = mod.main(["account", "0.0.12345", "--no-prices"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert len(out["tokens"]) == 2

    alpha = next(t for t in out["tokens"] if t["symbol"] == "ALPHA")
    assert alpha["balance"] == pytest.approx(10.0)  # 1000 / 10^2

    beta = next(t for t in out["tokens"] if t["symbol"] == "BETA")
    assert beta["balance"] == 500  # 500 / 10^0


def test_account_no_prices_skips_coingecko(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    cg_calls: list = []

    with patch.object(mod, "_http_get", side_effect=_account_http):
        with patch.object(mod, "_cg_get", side_effect=lambda *a, **kw: cg_calls.append(a) or None):
            mod.main(["account", "0.0.12345", "--no-prices"])

    assert cg_calls == [], "CoinGecko must not be called with --no-prices"


def test_account_not_found_returns_exit_1(monkeypatch):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=None):
        exit_code = mod.main(["account", "0.0.99999"])

    assert exit_code == 1


def test_account_token_cap_limits_metadata_calls(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    many_tokens = [{"token_id": f"0.0.{1000 + i}", "balance": 100} for i in range(15)]
    account_many = {
        **ACCOUNT_FIXTURE,
        "balance": {**ACCOUNT_FIXTURE["balance"], "tokens": many_tokens},
    }

    meta_calls: list = []

    def fake_http(path, params=None, **kw):
        if "/accounts/" in path and "nfts" not in path:
            return account_many
        if "/tokens/" in path:
            meta_calls.append(path)
            tid = path.rstrip("/").split("/")[-1]
            return {"token_id": tid, "name": "T", "symbol": "T", "decimals": "0"}
        return None

    with patch.object(mod, "_http_get", side_effect=fake_http):
        with patch.object(mod, "fetch_hbar_price", return_value=None):
            exit_code = mod.main(["account", "0.0.12345", "--no-prices"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert len(meta_calls) == mod.ACCOUNT_TOKEN_CAP
    assert out["tokens_omitted"] == 5  # 15 total - 10 cap


def test_account_hashscan_url_present(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", side_effect=_account_http):
        with patch.object(mod, "fetch_hbar_price", return_value=None):
            mod.main(["account", "0.0.12345", "--no-prices"])

    out = json.loads(capsys.readouterr().out)
    assert "hashscan.io" in out["hashscan_url"]
    assert "0.0.12345" in out["hashscan_url"]


# ===========================================================================
# Group F — token command
# ===========================================================================


def test_token_fungible_basic_fields(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=FUNGIBLE_TOKEN_FIXTURE):
        exit_code = mod.main(["token", "0.0.5678"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["type"] == "FUNGIBLE_COMMON"
    assert out["decimals"] == 6
    assert out["total_supply"] == pytest.approx(1_000_000.0)   # 1e12 / 10^6
    assert out["initial_supply"] == pytest.approx(500_000.0)
    assert out["symbol"] == "MTK"


def test_token_key_presence_shown_as_bool(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=FUNGIBLE_TOKEN_FIXTURE):
        mod.main(["token", "0.0.5678"])

    out = json.loads(capsys.readouterr().out)
    assert out["admin_key"] is True    # fixture has {"key": "abc123"}
    assert out["supply_key"] is False  # fixture has None
    assert out["pause_key"] is True    # fixture has {"key": "def456"}


def test_token_raw_key_bytes_not_in_output(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=FUNGIBLE_TOKEN_FIXTURE):
        mod.main(["token", "0.0.5678"])

    raw_output = capsys.readouterr().out
    assert "abc123" not in raw_output
    assert "def456" not in raw_output


def test_token_nft_type(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    nft_fixture = {**FUNGIBLE_TOKEN_FIXTURE, "type": "NON_FUNGIBLE_UNIQUE", "decimals": "0"}

    with patch.object(mod, "_http_get", return_value=nft_fixture):
        mod.main(["token", "0.0.5678"])

    out = json.loads(capsys.readouterr().out)
    assert out["type"] == "NON_FUNGIBLE_UNIQUE"


def test_token_not_found_returns_exit_1(monkeypatch):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=None):
        assert mod.main(["token", "0.0.9999"]) == 1


# ===========================================================================
# Group G — tx command
# ===========================================================================


def test_tx_required_fields_present(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=TX_FIXTURE):
        with patch.object(mod, "fetch_hbar_price", return_value=None):
            exit_code = mod.main(["tx", "0.0.1234-1234567890-000000000"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["type"] == "CRYPTOTRANSFER"
    assert out["result"] == "SUCCESS"
    assert "consensus_timestamp" in out
    assert "charged_tx_fee_hbar" in out
    assert "transfers" in out
    assert "hashscan_url" in out


def test_tx_fee_usd_computed_from_price(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=TX_FIXTURE):
        with patch.object(mod, "fetch_hbar_price", return_value=0.10):
            mod.main(["tx", "0.0.1234-1234567890-000000000"])

    out = json.loads(capsys.readouterr().out)
    # fee_tinybar=10000 → 0.0001 HBAR × $0.10 = $0.00001
    assert "charged_tx_fee_usd" in out
    assert out["charged_tx_fee_usd"] == pytest.approx(0.00001, rel=1e-3)


def test_tx_memo_base64_decoded(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    memo_b64 = base64.b64encode(b"test memo content").decode()
    fixture = {
        "transactions": [{**TX_FIXTURE["transactions"][0], "memo_base64": memo_b64}]
    }

    with patch.object(mod, "_http_get", return_value=fixture):
        with patch.object(mod, "fetch_hbar_price", return_value=None):
            mod.main(["tx", "0.0.1234-1234567890-000000000"])

    out = json.loads(capsys.readouterr().out)
    assert out["memo"] == "test memo content"


def test_tx_at_form_normalized_in_http_call(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    seen_paths: list = []

    def fake_http(path, params=None, **kw):
        seen_paths.append(path)
        return TX_FIXTURE

    with patch.object(mod, "_http_get", side_effect=fake_http):
        with patch.object(mod, "fetch_hbar_price", return_value=None):
            mod.main(["tx", "0.0.1234@1234567890.000000000"])

    # The HTTP call should use the canonical dash form
    assert any("0.0.1234-1234567890-000000000" in p for p in seen_paths)
    assert not any("@" in p for p in seen_paths)


def test_tx_not_found_returns_exit_1(monkeypatch):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=None):
        assert mod.main(["tx", "0.0.1234-1234567890-000000000"]) == 1


def test_tx_transfers_sorted_by_absolute_amount(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=TX_FIXTURE):
        with patch.object(mod, "fetch_hbar_price", return_value=None):
            mod.main(["tx", "0.0.1234-1234567890-000000000"])

    out = json.loads(capsys.readouterr().out)
    amounts = [abs(t["amount_hbar"]) for t in out["transfers"]]
    assert amounts == sorted(amounts, reverse=True)


# ===========================================================================
# Group H — activity command
# ===========================================================================


def test_activity_returns_correct_count(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=ACTIVITY_FIXTURE):
        exit_code = mod.main(["activity", "0.0.1234"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["count"] == 5
    assert len(out["transactions"]) == 5


def test_activity_limit_passed_to_query(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    seen_params: list = []

    def fake_http(path, params=None, **kw):
        if params:
            seen_params.append(dict(params))
        return {"transactions": []}

    with patch.object(mod, "_http_get", side_effect=fake_http):
        mod.main(["activity", "0.0.1234", "--limit", "7"])

    assert any(p.get("limit") == "7" for p in seen_params)


def test_activity_account_id_passed_to_query(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    seen_params: list = []

    def fake_http(path, params=None, **kw):
        if params:
            seen_params.append(dict(params))
        return {"transactions": []}

    with patch.object(mod, "_http_get", side_effect=fake_http):
        mod.main(["activity", "0.0.4321"])

    assert any(p.get("account.id") == "0.0.4321" for p in seen_params)


# ===========================================================================
# Group I — nft command
# ===========================================================================


def test_nft_ipfs_uri_decoded_as_string(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=NFT_FIXTURE):
        exit_code = mod.main(["nft", "0.0.1234"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    coll_100 = next(c for c in out["collections"] if c["token_id"] == "0.0.100")
    nft_1 = next(n for n in coll_100["nfts"] if n["serial_number"] == 1)
    assert nft_1["metadata"] == _IPFS_URI.decode()


def test_nft_binary_metadata_shown_as_hex(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=NFT_FIXTURE):
        mod.main(["nft", "0.0.1234"])

    out = json.loads(capsys.readouterr().out)
    coll_100 = next(c for c in out["collections"] if c["token_id"] == "0.0.100")
    nft_2 = next(n for n in coll_100["nfts"] if n["serial_number"] == 2)
    assert nft_2["metadata"] == _BINARY_RAW.hex()


def test_nft_grouped_by_collection(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=NFT_FIXTURE):
        mod.main(["nft", "0.0.1234"])

    out = json.loads(capsys.readouterr().out)
    collection_ids = {c["token_id"] for c in out["collections"]}
    assert collection_ids == {"0.0.100", "0.0.200"}

    coll_100 = next(c for c in out["collections"] if c["token_id"] == "0.0.100")
    assert coll_100["count"] == 2


def test_nft_total_count_correct(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=NFT_FIXTURE):
        mod.main(["nft", "0.0.1234"])

    out = json.loads(capsys.readouterr().out)
    assert out["total_nfts"] == 3  # 2 from 0.0.100 + 1 from 0.0.200


def test_nft_limit_passed_to_query(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    seen_params: list = []

    def fake_http(path, params=None, **kw):
        if params:
            seen_params.append(dict(params))
        return {"nfts": []}

    with patch.object(mod, "_http_get", side_effect=fake_http):
        mod.main(["nft", "0.0.1234", "--limit", "20"])

    assert any(p.get("limit") == "20" for p in seen_params)


# ===========================================================================
# Group J — price command
# ===========================================================================


def test_price_hbar_calls_fetch_hbar_price(monkeypatch, capsys):
    mod = load_module()

    with patch.object(mod, "fetch_hbar_price", return_value=0.0923) as mock_fn:
        exit_code = mod.main(["price", "HBAR"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["symbol"] == "HBAR"
    assert out["price_usd"] == pytest.approx(0.0923)
    mock_fn.assert_called_once()


def test_price_hbar_case_insensitive(monkeypatch, capsys):
    mod = load_module()

    with patch.object(mod, "fetch_hbar_price", return_value=0.09):
        exit_code = mod.main(["price", "hbar"])

    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["price_usd"] == pytest.approx(0.09)


def test_price_unknown_token_id_returns_null(monkeypatch, capsys):
    mod = load_module()

    exit_code = mod.main(["price", "0.0.99999"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["price_usd"] is None
    assert "not in known token registry" in out["note"]


def test_price_unknown_symbol_returns_null(monkeypatch, capsys):
    mod = load_module()

    exit_code = mod.main(["price", "UNKNOWNSYMBOL"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["price_usd"] is None


# ===========================================================================
# Group K — fees command
# ===========================================================================


def test_fees_output_has_required_operations(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=EXCHANGE_RATE_FIXTURE):
        exit_code = mod.main(["fees"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0

    op_names = {op["operation"] for op in out["operations"]}
    for expected in [
        "CryptoTransfer (HBAR)",
        "TokenAssociate",
        "ConsensusSubmitMessage",
        "ContractCall",
        "TokenCreate",
    ]:
        assert expected in op_names, f"Missing operation: {expected}"


def test_fees_hbar_cost_computed_correctly(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    # cent_equivalent=12, hbar_equivalent=1
    # → 12 cents = 1 HBAR → 1 USD = 100/12 ≈ 8.333 HBAR
    with patch.object(mod, "_http_get", return_value=EXCHANGE_RATE_FIXTURE):
        exit_code = mod.main(["fees"])

    out = json.loads(capsys.readouterr().out)
    expected_hbar_per_usd = 100 / 12
    assert out["hbar_per_usd"] == pytest.approx(expected_hbar_per_usd, rel=1e-3)

    transfer_op = next(
        op for op in out["operations"] if op["operation"] == "CryptoTransfer (HBAR)"
    )
    assert transfer_op["cost_usd"] == 0.0001
    assert transfer_op["cost_hbar"] == pytest.approx(
        0.0001 * expected_hbar_per_usd, rel=1e-3
    )


def test_fees_usd_costs_present_when_exchange_rate_unavailable(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=None):
        exit_code = mod.main(["fees"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["exchange_rate_source"] == "unavailable"
    # USD costs must always be present
    assert all("cost_usd" in op for op in out["operations"])
    # HBAR costs absent when rate unavailable
    assert all("cost_hbar" not in op for op in out["operations"])


def test_fees_includes_schedule_metadata(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=None):
        mod.main(["fees"])

    out = json.loads(capsys.readouterr().out)
    assert "fee_schedule_version" in out
    assert "fee_schedule_url" in out
    assert "hedera" in out["fee_schedule_url"]


# ===========================================================================
# Group L — topic command
# ===========================================================================


def _topic_http(path, params=None, **kw):
    if "/messages" in path:
        return MESSAGES_FIXTURE
    if "/topics/" in path:
        return TOPIC_FIXTURE
    return None


def test_topic_metadata_fields(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", side_effect=_topic_http):
        exit_code = mod.main(["topic", "0.0.1001"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["topic_id"] == "0.0.1001"
    assert out["memo"] == "test topic"
    assert out["admin_key"] is True    # fixture has {"key": "abc"}
    assert out["submit_key"] is False  # fixture has None
    assert "created_timestamp" in out
    assert "hashscan_url" in out


def test_topic_messages_decoded_from_base64(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", side_effect=_topic_http):
        mod.main(["topic", "0.0.1001"])

    out = json.loads(capsys.readouterr().out)
    assert len(out["recent_messages"]) == 1
    msg = out["recent_messages"][0]
    assert msg["sequence_number"] == 42
    assert msg["message"] == "hello from HCS"


def test_topic_messages_limit_param_forwarded(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    seen_params: list = []

    def fake_http(path, params=None, **kw):
        if params:
            seen_params.append(dict(params))
        if "/messages" in path:
            return {"messages": []}
        return TOPIC_FIXTURE

    with patch.object(mod, "_http_get", side_effect=fake_http):
        mod.main(["topic", "0.0.1001", "--messages", "3"])

    assert any(p.get("limit") == "3" for p in seen_params)


def test_topic_not_found_returns_exit_1(monkeypatch):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=None):
        assert mod.main(["topic", "0.0.9999"]) == 1


def test_topic_running_hash_prefix_truncated(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", side_effect=_topic_http):
        mod.main(["topic", "0.0.1001"])

    out = json.loads(capsys.readouterr().out)
    msg = out["recent_messages"][0]
    full_hash = MESSAGES_FIXTURE["messages"][0]["running_hash"]
    assert msg["running_hash_prefix"] == full_hash[:16]


# ===========================================================================
# Group M — contract command
# ===========================================================================


def test_contract_required_fields_present(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=CONTRACT_FIXTURE):
        exit_code = mod.main(["contract", "0.0.5678"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["contract_id"] == "0.0.5678"
    assert out["evm_address"] == "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
    assert out["admin_key"] is True
    assert "bytecode_size_bytes" in out
    assert "balance_hbar" in out
    assert "hashscan_url" in out


def test_contract_bytecode_size_derived_from_hex_length(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    fixture = {**CONTRACT_FIXTURE, "bytecode": "0x" + "ab" * 100}

    with patch.object(mod, "_http_get", return_value=fixture):
        mod.main(["contract", "0.0.5678"])

    out = json.loads(capsys.readouterr().out)
    assert out["bytecode_size_bytes"] == 100


def test_contract_accepts_dotted_id(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=CONTRACT_FIXTURE):
        exit_code = mod.main(["contract", "0.0.5678"])

    assert exit_code == 0


def test_contract_accepts_evm_address(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=CONTRACT_FIXTURE):
        exit_code = mod.main(
            ["contract", "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"]
        )

    assert exit_code == 0


def test_contract_not_found_returns_exit_1(monkeypatch):
    mod = load_module()
    monkeypatch.delenv("HEDERA_MIRROR_URL", raising=False)

    with patch.object(mod, "_http_get", return_value=None):
        assert mod.main(["contract", "0.0.9999"]) == 1


# ===========================================================================
# Group N — HTTP retry behavior
# ===========================================================================


def test_http_retries_on_429_and_succeeds_on_third_attempt(monkeypatch):
    mod = load_module()

    attempt_count = [0]
    success_payload = b'{"result": "ok"}'

    class Fake429(urllib.error.HTTPError):
        def __init__(self):
            super().__init__(
                url="http://x", code=429, msg="Too Many Requests", hdrs=None, fp=None
            )

    class FakeResponse:
        def read(self):
            return success_payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fake_urlopen(req, timeout=None):
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise Fake429()
        return FakeResponse()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with patch.object(mod, "_mirror_base", return_value="http://test"):
            with patch("time.sleep"):  # avoid actual sleeping in tests
                result = mod._http_get("/api/v1/test")

    assert result == {"result": "ok"}
    assert attempt_count[0] == 3


def test_http_returns_none_on_404(monkeypatch):
    mod = load_module()

    class Fake404(urllib.error.HTTPError):
        def __init__(self):
            super().__init__(
                url="http://x", code=404, msg="Not Found", hdrs=None, fp=None
            )

    with patch("urllib.request.urlopen", side_effect=Fake404()):
        with patch.object(mod, "_mirror_base", return_value="http://test"):
            result = mod._http_get("/api/v1/missing")

    assert result is None
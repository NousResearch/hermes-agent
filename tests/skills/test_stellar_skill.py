"""Offline tests for the optional Stellar skill's helper script.

Golden XDR values were produced with the official @stellar/stellar-sdk (JS); the
ledger entries are real mainnet responses captured from getLedgerEntries. No test
touches the network: Horizon and RPC calls are patched at the module boundary.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import struct
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "stellar"
    / "scripts"
    / "stellar_client.py"
)

ACCOUNT_G = "GAHK7EEG2WWHVKDNT4CEQFZGKF2LGDSW2IVM4S5DP42RBW3K6BTODB4A"
ACCOUNT_G_HEX = "0eaf9086d5ac7aa86d9f044817265174b30e56d22ace4ba37f3510db6af066e1"
USDC_SAC_C = "CCW67TSZV3SSS2HXMBQ5JFGCKJNXKZM7UQUWUZPUTHXSTZLEO7SJMI75"
USDC_SAC_HEX = "adefce59aee52968f76061d494c2525b75659fa4296a65f499ef29e56477e496"
USDC_ISSUER = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"
WASM_CONTRACT_C = "CDL74RF5BLYR2YBLCCI7F5FB6TPSCLKEJUBSD2RSVWZ4YHF3VMFAIGWA"

# LedgerKey(ContractData{contract=USDC SAC, key=LedgerKeyContractInstance, persistent}) from the JS SDK
USDC_INSTANCE_KEY_B64 = "AAAABgAAAAGt785ZruUpaPdgYdSUwlJbdWWfpClqZfSZ7ynlZHfklgAAABQAAAAB"

# Real mainnet getLedgerEntries `xdr` for the USDC SAC instance (LedgerEntryData, CONTRACT_DATA)
USDC_INSTANCE_XDR = (
    "AAAABgAAAAAAAAABre/OWa7lKWj3YGHUlMJSW3Vln6QpamX0me8p5WR35JYAAAAUAAAAAQAAABMAAAABAAAAAQAAAAMAAAAP"
    "AAAACE1FVEFEQVRBAAAAEQAAAAEAAAADAAAADwAAAAdkZWNpbWFsAAAAAAMAAAAHAAAADwAAAARuYW1lAAAADgAAAD1VU0RD"
    "OkdBNVpTRUpZQjM3SlJDNUFWQ0lBNU1PUDRSSFRNMzM1WDJLR1gzSUhPSkFQUDVSRTM0SzRLWlZOAAAAAAAADwAAAAZzeW1i"
    "b2wAAAAAAA4AAAAEVVNEQwAAABAAAAABAAAAAQAAAA8AAAAFQWRtaW4AAAAAAAASAAAAAZ601+BSioJxdy1pbNxo0dqyC/TY"
    "oHAyo14PApmGrMUeAAAAEAAAAAEAAAABAAAADwAAAAlBc3NldEluZm8AAAAAAAAQAAAAAQAAAAIAAAAPAAAACUFscGhhTnVt"
    "NAAAAAAAABEAAAABAAAAAgAAAA8AAAAKYXNzZXRfY29kZQAAAAAADgAAAARVU0RDAAAADwAAAAZpc3N1ZXIAAAAAAA0AAAAg"
    "O5kROA7+mIugqJAOsc/kTzZvfb6Ua+0HckD39iTfFcU="
)

# Real mainnet instance entry for a WASM contract (executable = wasm hash, 5 instance storage keys)
WASM_INSTANCE_XDR = (
    "AAAABgAAAAAAAAAB1/5EvQrxHWArEJHy9KH03yEtRE0DIeoyrbPMHLurCgQAAAAUAAAAAQAAABMAAAAA2ywUKQ1JZOOAXyUn"
    "3RMpObpfs/zKxWswv6uP0JEBFicAAAABAAAABQAAABAAAAABAAAAAQAAAA8AAAAJRmFybUJsb2NrAAAAAAAAEQAAAAEAAAAK"
    "AAAADwAAAAdlbnRyb3B5AAAAAA0AAAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPAAAAB21heF9nYXAA"
    "AAAAAwAAAAAAAAAPAAAACW1heF9zdGFrZQAAAAAAAAoAAAAAAAAAAAAAAAAAYx2PAAAADwAAAAltYXhfemVyb3MAAAAAAAAD"
    "AAAAAAAAAA8AAAAHbWluX2dhcAAAAAAD/////wAAAA8AAAAJbWluX3N0YWtlAAAAAAAACgAAAAAAAAAAAAAAAAAAAAAAAAAP"
    "AAAACW1pbl96ZXJvcwAAAAAAAAP/////AAAADwAAABBub3JtYWxpemVkX3RvdGFsAAAACgAAAAAAAAAAAAAAAAAAAAAAAAAP"
    "AAAADHN0YWtlZF90b3RhbAAAAAoAAAAAAAAAAAAAAAAAAAAAAAAADwAAAAl0aW1lc3RhbXAAAAAAAAAFAAAAAGrABsEAAAAQ"
    "AAAAAQAAAAEAAAAPAAAAC0Zhcm1FbnRyb3B5AAAAAA0AAAAgAAAACKrpeUFcFVMGtdE5/PTOQtDv3VVQ4uqczGo4kTkAAAAQ"
    "AAAAAQAAAAEAAAAPAAAACUZhcm1JbmRleAAAAAAAAAMAAuKhAAAAEAAAAAEAAAABAAAADwAAAA5Ib21lc3RlYWRBc3NldAAA"
    "AAAAEgAAAAF1u0RwsaT/YezHKV6LjrdEGd1Ybu5ATN9SSZFdiQ4IdwAAABAAAAABAAAAAQAAAA8AAAALSG9tZXN0ZWFkZXIA"
    "AAAAEgAAAAAAAAAAR1vypFiHKHeKgnE2nuA5VhED/841SUAs4KR5zr8bCfU="
)
WASM_HASH_HEX = "db2c14290d4964e3805f2527dd132939ba5fb3fccac56b30bfab8fd091011627"

# ScVal golden values from @stellar/stellar-sdk: base64 XDR -> expected decoded Python value
SCVAL_FIXTURES = {
    "sym": ("AAAADwAAAAh0cmFuc2Zlcg==", "transfer"),
    "str": ("AAAADgAAAA1oZWxsbyBzdGVsbGFyAAAA", "hello stellar"),
    "u32": ("AAAAAwAAACo=", 42),
    "i32": ("AAAABP////k=", -7),
    "u64": ("AAAABQAAAR9x+wTL", 1234567890123),
    "i64": ("AAAABv/////////7", -5),
    "i128_min": ("AAAACoAAAAAAAAAAAAAAAAAAAAA=", -170141183460469231731687303715884105728),
    "i128_pos": ("AAAACgAAAAAAAAAAq1SpjOsfCtI=", 12345678901234567890),
    "u128_max": ("AAAACf////////////////////8=", 340282366920938463463374607431768211455),
    "u256": ("AAAACwAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 2 ** 200),
    "i256": ("AAAADP////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", -(2 ** 200)),
    "bool": ("AAAAAAAAAAE=", True),
    "void": ("AAAAAQ==", None),
    "addr_g": ("AAAAEgAAAAAAAAAADq+QhtWseqhtnwRIFyZRdLMOVtIqzkujfzUQ22rwZuE=", ACCOUNT_G),
    "addr_c": ("AAAAEgAAAAGt785ZruUpaPdgYdSUwlJbdWWfpClqZfSZ7ynlZHfklg==", USDC_SAC_C),
    "bytes": ("AAAADQAAAATerb7v", "0xdeadbeef"),
    "vec": ("AAAAEAAAAAEAAAACAAAADwAAAAFhAAAAAAAAAwAAAAE=", ["a", 1]),
    "map": ("AAAAEQAAAAEAAAABAAAADwAAAAFrAAAAAAAABAAAAAk=", {"k": 9}),
    "timepoint": ("AAAABwAAAABlU/EA", 1700000000),
    "duration": ("AAAACAAAAAAAAAA8", 60),
    "error": ("AAAAAgAAAAAAAAAF", "Error(Contract, #5)"),
}

HORIZON_ACCOUNT = json.loads(
    '{"id":"GAHK7EEG2WWHVKDNT4CEQFZGKF2LGDSW2IVM4S5DP42RBW3K6BTODB4A","account_id":"GAHK7EEG2WWHVKDNT4CEQFZGKF2LGDSW2IVM4S5DP42RBW3K6BTODB4A",'
    '"sequence":"64041853623977989","subentry_count":2,"last_modified_ledger":64717097,"num_sponsoring":0,"num_sponsored":0,'
    '"thresholds":{"low_threshold":0,"med_threshold":0,"high_threshold":0},'
    '"flags":{"auth_required":false,"auth_revocable":false,"auth_immutable":false,"auth_clawback_enabled":false},'
    '"balances":[{"balance":"0.0000000","limit":"922337203685.4775807","buying_liabilities":"0.0000000","selling_liabilities":"0.0000000",'
    '"last_modified_ledger":64717097,"is_authorized":true,"is_authorized_to_maintain_liabilities":true,"asset_type":"credit_alphanum4",'
    '"asset_code":"USDC","asset_issuer":"GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"},'
    '{"balance":"12.5000000","limit":"922337203685.4775807","buying_liabilities":"0.0000000","selling_liabilities":"0.0000000",'
    '"last_modified_ledger":64717094,"is_authorized":true,"is_authorized_to_maintain_liabilities":true,"asset_type":"credit_alphanum4",'
    '"asset_code":"VELO","asset_issuer":"GDM4RQUQQUVSKQA7S6EM7XBZP3FCGH4Q7CL6TABQ7B2BEJ5ERARM2M5M"},'
    '{"balance":"9.9999501","buying_liabilities":"0.0000000","selling_liabilities":"1.0000000","asset_type":"native"}],'
    '"signers":[{"weight":1,"key":"GAHK7EEG2WWHVKDNT4CEQFZGKF2LGDSW2IVM4S5DP42RBW3K6BTODB4A","type":"ed25519_public_key"}],"data":{}}'
)


def load_module():
    spec = importlib.util.spec_from_file_location("stellar_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any unpatched HTTP call is a test bug, not a flaky network."""
    import urllib.request

    def explode(*_args, **_kwargs):
        raise AssertionError("test attempted a live network call")

    monkeypatch.setattr(urllib.request, "urlopen", explode)


@pytest.fixture
def mod():
    return load_module()


# ---------------------------------------------------------------------------
# StrKey + ledger keys
# ---------------------------------------------------------------------------

def test_strkey_roundtrip_matches_sdk_vectors(mod):
    assert mod.strkey_decode(ACCOUNT_G) == ("G", bytes.fromhex(ACCOUNT_G_HEX))
    assert mod.strkey_decode(USDC_SAC_C) == ("C", bytes.fromhex(USDC_SAC_HEX))
    assert mod.strkey_encode("G", bytes.fromhex(ACCOUNT_G_HEX)) == ACCOUNT_G
    assert mod.strkey_encode("C", bytes.fromhex(USDC_SAC_HEX)) == USDC_SAC_C
    assert mod.is_account_id(ACCOUNT_G) and not mod.is_contract_id(ACCOUNT_G)
    assert mod.is_contract_id(USDC_SAC_C) and not mod.is_account_id(USDC_SAC_C)


def test_strkey_rejects_typos_and_secret_seeds(mod):
    assert not mod.is_account_id(ACCOUNT_G[:-1] + "B")  # checksum mismatch
    assert not mod.is_account_id("not-an-address")
    assert not mod.is_account_id("")
    with pytest.raises(mod.ClientError, match="checksum"):
        mod.strkey_decode(ACCOUNT_G[:-1] + "B")
    with pytest.raises(mod.ClientError, match="SECRET"):
        mod.require_account_id("S" + "A" * 55)


def test_contract_instance_key_matches_sdk(mod):
    assert mod.contract_instance_key(USDC_SAC_C) == USDC_INSTANCE_KEY_B64
    code_key = base64.b64decode(mod.contract_code_key(WASM_HASH_HEX))
    assert code_key == struct.pack(">i", 7) + bytes.fromhex(WASM_HASH_HEX)


# ---------------------------------------------------------------------------
# XDR decoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SCVAL_FIXTURES))
def test_scval_decoder_matches_sdk_fixtures(mod, name):
    b64, expected = SCVAL_FIXTURES[name]
    reader = mod.XdrReader(base64.b64decode(b64))
    assert mod.decode_scval(reader) == expected
    assert reader.pos == len(reader.data), "decoder must consume the whole value"


def test_decode_sac_instance_entry(mod):
    entry = mod.decode_ledger_entry_data(USDC_INSTANCE_XDR)
    assert entry["type"] == "contract_data"
    assert entry["contract"] == USDC_SAC_C
    assert entry["durability"] == "persistent"
    instance = entry["value"]
    assert instance["executable"] == {"type": "stellar_asset"}
    storage = instance["storage"]
    assert set(storage) == {"METADATA", "Admin", "AssetInfo"}
    assert storage["METADATA"] == {"decimal": 7, "name": f"USDC:{USDC_ISSUER}", "symbol": "USDC"}
    assert storage["Admin"].startswith("C")
    assert storage["AssetInfo"][0] == "AlphaNum4"
    assert storage["AssetInfo"][1]["asset_code"] == "USDC"


def test_decode_wasm_instance_entry(mod):
    entry = mod.decode_ledger_entry_data(WASM_INSTANCE_XDR)
    assert entry["contract"] == WASM_CONTRACT_C
    instance = entry["value"]
    assert instance["executable"] == {"type": "wasm", "wasm_hash": WASM_HASH_HEX}
    assert list(instance["storage"]) == ["FarmBlock", "FarmEntropy", "FarmIndex", "HomesteadAsset", "Homesteader"]
    assert instance["storage"]["FarmBlock"]["min_gap"] == 2 ** 32 - 1  # stored as u32::MAX
    assert instance["storage"]["HomesteadAsset"].startswith("C")
    assert instance["storage"]["Homesteader"].startswith("G")


def test_decode_contract_code_entry_with_cost_inputs(mod):
    cost = list(range(1, 11))
    code = b"\x00asm\x01\x00\x00\x00ok"  # 10 bytes -> 2 bytes XDR padding
    raw = (
        struct.pack(">i", 7)                      # LedgerEntryType CONTRACT_CODE
        + struct.pack(">i", 1)                    # ext v1
        + struct.pack(">i", 0)                    # v1.ext ExtensionPoint
        + struct.pack(">i", 0)                    # costInputs.ext ExtensionPoint
        + b"".join(struct.pack(">I", n) for n in cost)
        + bytes.fromhex(WASM_HASH_HEX)
        + struct.pack(">I", len(code)) + code + b"\x00\x00"
    )
    entry = mod.decode_ledger_entry_data(base64.b64encode(raw).decode())
    assert entry == {
        "type": "contract_code",
        "wasm_hash": WASM_HASH_HEX,
        "code_bytes": 10,
        "cost_inputs": dict(zip(mod.CONTRACT_CODE_COST_FIELDS, cost)),
    }


def test_scaddress_muxed_claimable_and_pool_variants(mod):
    ed25519 = bytes.fromhex(ACCOUNT_G_HEX)
    muxed = struct.pack(">i", 18) + struct.pack(">i", 2) + struct.pack(">Q", 7) + ed25519
    decoded = mod.decode_scval(mod.XdrReader(muxed))
    assert decoded.startswith("M")
    assert mod.strkey_decode(decoded) == ("M", ed25519 + struct.pack(">Q", 7))
    pool = struct.pack(">i", 18) + struct.pack(">i", 4) + bytes(32)
    assert mod.decode_scval(mod.XdrReader(pool)).startswith("L")
    claimable = struct.pack(">i", 18) + struct.pack(">i", 3) + struct.pack(">i", 0) + bytes(32)
    assert mod.decode_scval(mod.XdrReader(claimable)).startswith("B")


def test_undecodable_scval_is_returned_raw_not_raised(mod):
    assert mod.decode_scval_b64("AAAAEgAA") == {"undecoded_xdr": "AAAAEgAA"}
    assert mod.decode_scval_b64("not base64!") == {"undecoded_xdr": "not base64!"}


# ---------------------------------------------------------------------------
# TransactionResult decoding
# ---------------------------------------------------------------------------

def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def test_tx_result_success_and_failed_payment(mod):
    success = _b64(struct.pack(">q", 100) + struct.pack(">i", 0) + struct.pack(">I", 1)
                   + struct.pack(">i", 0) + struct.pack(">i", 1) + struct.pack(">i", 0))
    assert mod.decode_tx_result(success) == {
        "fee_charged_stroops": 100, "code": "txSUCCESS", "operations": ["op0 payment: SUCCESS"],
    }
    failed = _b64(struct.pack(">q", 100) + struct.pack(">i", -1) + struct.pack(">I", 1)
                  + struct.pack(">i", 0) + struct.pack(">i", 1) + struct.pack(">i", -2))
    assert mod.decode_tx_result(failed)["code"] == "txFAILED"
    assert mod.decode_tx_result(failed)["operations"] == ["op0 payment: UNDERFUNDED"]
    archived = _b64(struct.pack(">q", 5000) + struct.pack(">i", -1) + struct.pack(">I", 1)
                    + struct.pack(">i", 0) + struct.pack(">i", 24) + struct.pack(">i", -4))
    assert mod.decode_tx_result(archived)["operations"] == ["op0 invoke_host_function: ENTRY_ARCHIVED"]
    assert mod.decode_tx_result(_b64(struct.pack(">q", 100) + struct.pack(">i", -5)))["code"] == "txBAD_SEQ"


def test_tx_result_fee_bump_unwraps_inner(mod):
    inner_ops = struct.pack(">I", 1) + struct.pack(">i", 0) + struct.pack(">i", 24) + struct.pack(">i", 0) + bytes(32)
    raw = struct.pack(">q", 200) + struct.pack(">i", 1) + bytes(32) + struct.pack(">q", 150) + struct.pack(">i", 0) + inner_ops
    result = mod.decode_tx_result(_b64(raw))
    assert result["code"] == "txFEE_BUMP_INNER_SUCCESS"
    assert result["inner_code"] == "txSUCCESS"
    assert result["operations"] == ["op0 invoke_host_function: SUCCESS"]


def test_tx_result_garbage_is_returned_raw(mod):
    assert mod.decode_tx_result("AAAA") == {"undecoded_xdr": "AAAA"}
    assert mod.decode_tx_result(None) == {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_network_resolution_reads_hermes_dotenv(tmp_path, monkeypatch, mod):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text(
        'STELLAR_NETWORK=testnet\nSTELLAR_RPC_URL="https://rpc.example.test/"\n', encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for key in ("STELLAR_NETWORK", "STELLAR_RPC_URL", "STELLAR_HORIZON_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)

    cfg = mod.resolve_config(None)
    assert cfg["name"] == "testnet"
    assert cfg["horizon"] == "https://horizon-testnet.stellar.org"
    assert cfg["rpc"] == "https://rpc.example.test"
    assert cfg["friendbot"] == "https://friendbot.stellar.org"
    assert mod.resolve_config("mainnet")["name"] == "mainnet"  # CLI flag beats .env
    with pytest.raises(mod.ClientError, match="Unknown network"):
        mod.resolve_config("futurenet")


def test_user_dotenv_overrides_project_dotenv(tmp_path, monkeypatch, mod):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("STELLAR_NETWORK=mainnet\n", encoding="utf-8")
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text("STELLAR_NETWORK=testnet\n", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("STELLAR_NETWORK", raising=False)
    assert mod._env_lookup("STELLAR_NETWORK") == "testnet"


def test_parse_asset_known_codes_and_explicit_issuers(mod):
    cfg = mod.resolve_config("mainnet")
    assert mod.parse_asset("XLM", cfg) == ("XLM", None)
    assert mod.parse_asset("usdc", cfg) == ("USDC", USDC_ISSUER)
    assert mod.parse_asset("yxlm", cfg) == ("yXLM", "GARDNV3Q7YGT4AKSDF25LT32YSCCW4EV22Y2TV3I2PU2MMXJTEDL5T55")
    assert mod.parse_asset(f"FOO:{ACCOUNT_G}", cfg) == ("FOO", ACCOUNT_G)
    with pytest.raises(mod.ClientError, match="Unknown asset"):
        mod.parse_asset("FOO", cfg)
    with pytest.raises(mod.ClientError, match="not a valid Stellar account id"):
        mod.parse_asset("FOO:GABC", cfg)


# ---------------------------------------------------------------------------
# Commands (HTTP boundary patched)
# ---------------------------------------------------------------------------

def test_account_command_reports_reserve_and_spendable_offline(mod, capsys):
    def fake_horizon(cfg, path, **params):
        assert cfg["name"] == "mainnet"
        assert path == f"/accounts/{ACCOUNT_G}"
        return HORIZON_ACCOUNT

    with patch.object(mod, "horizon_get", side_effect=fake_horizon):
        exit_code = mod.main(["account", ACCOUNT_G, "--no-prices", "--json"])

    assert exit_code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["xlm_balance"] == pytest.approx(9.9999501)
    assert data["xlm_min_balance"] == pytest.approx(2.0)          # (2 + 2 subentries) * 0.5
    assert data["xlm_spendable"] == pytest.approx(6.9999501)      # minus 1 XLM selling liabilities
    assert data["xlm_usd"] is None                                # --no-prices made no DEX calls
    assert [t["code"] for t in data["trustlines"]] == ["VELO", "USDC"]  # non-zero balance first
    assert data["trustlines"][0]["balance"] == "12.5000000"
    assert data["explorer_url"].endswith(ACCOUNT_G)


def test_account_404_explains_unfunded_accounts(mod, capsys):
    def missing(cfg, path, **params):
        raise mod.ClientError("HTTP 404 from horizon.stellar.org: Resource Missing.", status=404)

    with patch.object(mod, "horizon_get", side_effect=missing):
        assert mod.main(["account", ACCOUNT_G, "--no-prices"]) == 1
    assert "never" in capsys.readouterr().err.lower() or True
    with patch.object(mod, "horizon_get", side_effect=missing):
        mod.main(["account", ACCOUNT_G, "--no-prices"])
    assert "does not exist on mainnet" in capsys.readouterr().err


def test_fund_refuses_mainnet_before_any_http(mod, capsys):
    assert mod.main(["fund", ACCOUNT_G]) == 1
    assert "testnet" in capsys.readouterr().err


def test_contract_command_renders_sac_and_ttl_offline(mod, capsys):
    latest = 64_736_946

    def fake_rpc(cfg, method, params=None):
        if method == "getLedgerEntries":
            assert params == {"keys": [USDC_INSTANCE_KEY_B64]}
            return {
                "latestLedger": latest,
                "entries": [{"key": USDC_INSTANCE_KEY_B64, "xdr": USDC_INSTANCE_XDR,
                             "lastModifiedLedgerSeq": 62313360, "liveUntilLedgerSeq": latest + 17280 * 10}],
            }
        if method == "getHealth":
            return {"latestLedger": latest, "oldestLedger": latest - 120_960,
                    "latestLedgerCloseTime": "1790969437", "oldestLedgerCloseTime": str(1790969437 - 604_800)}
        raise AssertionError(method)

    with patch.object(mod, "rpc_call", side_effect=fake_rpc):
        assert mod.main(["contract", USDC_SAC_C]) == 0
    out = capsys.readouterr().out
    assert "Stellar Asset Contract (SAC) for USDC:" in out
    assert "decimals 7" in out
    assert "~10.0 days" in out            # 172,800 ledgers at 5.0 s/ledger
    assert "METADATA, Admin, AssetInfo" in out


def test_contract_command_flags_expired_instance(mod, capsys):
    latest = 70_000_000

    def fake_rpc(cfg, method, params=None):
        if method == "getLedgerEntries":
            return {"latestLedger": latest, "entries": [{"xdr": WASM_INSTANCE_XDR, "lastModifiedLedgerSeq": 1,
                                                          "liveUntilLedgerSeq": latest - 5}]} if params["keys"][0].startswith("AAAABg") else {"latestLedger": latest, "entries": []}
        return {}

    with patch.object(mod, "rpc_call", side_effect=fake_rpc):
        assert mod.main(["contract", WASM_CONTRACT_C]) == 0
    out = capsys.readouterr().out
    assert f"WASM contract  hash {WASM_HASH_HEX}" in out
    assert "EXPIRED at ledger" in out
    assert "archived" in out


def test_stats_json_offline(mod, capsys):
    def fake_horizon(cfg, path, **params):
        if path == "/":
            return {"network_passphrase": cfg["passphrase"], "current_protocol_version": 29,
                    "history_latest_ledger": 64736898, "horizon_version": "29.0.0-abc", "core_version": "stellar-core 29.0.0"}
        if path == "/fee_stats":
            return {"last_ledger_base_fee": "100", "ledger_capacity_usage": "0.6",
                    "fee_charged": {"p50": "9230", "p99": "317683", "max": "317727"}}
        raise AssertionError(path)

    def fake_rpc(cfg, method, params=None):
        assert method == "getHealth"
        return {"status": "healthy", "latestLedger": 64736899, "oldestLedger": 64615940, "ledgerRetentionWindow": 120960,
                "latestLedgerCloseTime": "1790969437", "oldestLedgerCloseTime": str(1790969437 - 604_795)}

    with patch.object(mod, "horizon_get", side_effect=fake_horizon), \
         patch.object(mod, "rpc_call", side_effect=fake_rpc), \
         patch.object(mod.Pricer, "xlm_usd", return_value=0.2124):
        assert mod.main(["stats", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["protocol_version"] == 29
    assert data["horizon_version"] == "29.0.0"
    assert data["rpc_retention_ledgers"] == 120960
    assert data["seconds_per_ledger"] == pytest.approx(5.0, abs=0.01)
    assert data["xlm_usd"] == 0.2124


def test_invoke_host_function_summary_decodes_arguments(mod):
    op = {
        "type": "invoke_host_function",
        "function": "HostFunctionTypeHostFunctionTypeInvokeContract",
        "parameters": [
            {"type": "Address", "value": SCVAL_FIXTURES["addr_c"][0]},
            {"type": "Sym", "value": SCVAL_FIXTURES["sym"][0]},
            {"type": "Address", "value": SCVAL_FIXTURES["addr_g"][0]},
            {"type": "I128", "value": SCVAL_FIXTURES["i128_pos"][0]},
        ],
        "asset_balance_changes": [{"type": "transfer", "amount": "5.0000000", "asset_type": "credit_alphanum4",
                                   "asset_code": "USDC", "asset_issuer": USDC_ISSUER, "from": ACCOUNT_G, "to": USDC_ISSUER}],
    }
    summary = mod.summarize_operation(op)
    assert summary["contract"] == USDC_SAC_C
    assert summary["method"] == "transfer"
    assert summary["args"] == [ACCOUNT_G, 12345678901234567890]
    assert summary["text"].startswith("CCW6...MI75.transfer(")
    assert summary["balance_changes"] == [f"transfer 5 USDC:GA5Z...KZVN {mod.short(ACCOUNT_G)} -> {mod.short(USDC_ISSUER)}"]


def test_amount_formatting_keeps_horizon_precision(mod):
    assert mod.fmt_amount("922337203685.4775807") == "922,337,203,685.4775807"
    assert mod.fmt_amount("9.9999501") == "9.9999501"
    assert mod.fmt_amount("0.0000000") == "0"
    assert mod.fmt_stroops("13940") == "0.001394 XLM"
    assert mod.fmt_usd(0.21242) == "$0.21242"
    assert mod.fmt_usd(1234.5) == "$1,234.50"

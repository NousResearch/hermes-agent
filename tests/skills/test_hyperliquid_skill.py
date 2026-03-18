from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "hyperliquid"
    / "scripts"
    / "hyperliquid_client.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("hyperliquid_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_perp_markets_extracts_change_and_volume():
    mod = load_module()

    payload = [
        {
            "universe": [
                {"name": "BTC", "szDecimals": 5, "maxLeverage": 50},
                {"name": "ETH", "szDecimals": 4, "maxLeverage": 25, "isDelisted": True},
            ]
        },
        [
            {
                "markPx": "100000",
                "prevDayPx": "95000",
                "funding": "0.0001",
                "openInterest": "123456789",
                "dayNtlVlm": "999999999",
            },
            {
                "markPx": "2500",
                "prevDayPx": "2600",
                "funding": "-0.0002",
                "openInterest": "20000000",
                "dayNtlVlm": "11111111",
            },
        ],
    ]

    rows = mod._normalize_perp_markets(payload)

    assert len(rows) == 2
    assert rows[0]["coin"] == "BTC"
    assert round(rows[0]["change_pct"], 2) == 5.26
    assert rows[0]["day_ntl_vlm"] == "999999999"
    assert rows[1]["is_delisted"] is True


def test_normalize_dexs_includes_first_perp_dex_placeholder():
    mod = load_module()

    rows = mod._normalize_dexs(
        [
            None,
            {
                "name": "test",
                "fullName": "test dex",
                "deployer": "0x1234567890abcdef1234567890abcdef12345678",
                "assetToStreamingOiCap": [["COIN", "100"]],
            },
        ]
    )

    assert rows[0]["label"] == "first-perp-dex"
    assert rows[1]["label"] == "test"
    assert rows[1]["asset_caps"] == 1


def test_main_markets_json_prints_normalized_payload(capsys):
    mod = load_module()

    payload = [
        {"universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 50}]},
        [{"markPx": "101000", "prevDayPx": "100000", "dayNtlVlm": "10"}],
    ]

    with patch.object(mod, "_post_info", return_value=payload):
        exit_code = mod.main(["markets", "--limit", "1", "--json"])

    stdout = capsys.readouterr().out
    rendered = json.loads(stdout)

    assert exit_code == 0
    assert rendered["count"] == 1
    assert rendered["markets"][0]["coin"] == "BTC"
    assert round(rendered["markets"][0]["change_pct"], 2) == 1.0


def test_main_candles_json_limits_rows(capsys):
    mod = load_module()

    payload = [
        {"t": 1000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10", "n": 3},
        {"t": 2000, "o": "1.5", "h": "2.5", "l": "1.4", "c": "2.0", "v": "20", "n": 5},
        {"t": 3000, "o": "2.0", "h": "2.2", "l": "1.8", "c": "2.1", "v": "15", "n": 4},
    ]

    with patch.object(mod, "_post_info", return_value=payload):
        exit_code = mod.main(["candles", "BTC", "--limit", "2", "--json"])

    stdout = capsys.readouterr().out
    rendered = json.loads(stdout)

    assert exit_code == 0
    assert rendered["count"] == 3
    assert len(rendered["candles"]) == 2
    assert rendered["summary"]["open"] == "1"
    assert rendered["summary"]["close"] == "2.1"


def test_main_review_json_builds_market_context_and_findings(capsys):
    mod = load_module()

    def fake_post_info(payload):
        payload_type = payload["type"]
        if payload_type == "userFillsByTime":
            return [
                {"fill": {"coin": "BTC", "dir": "Close Long", "px": "110000", "sz": "0.1", "closedPnl": "120", "fee": "5", "feeToken": "USDC", "time": 4000}},
                {"fill": {"coin": "BTC", "dir": "Open Long", "px": "100000", "sz": "0.1", "closedPnl": "0", "fee": "1", "feeToken": "USDC", "time": 3000}},
                {"fill": {"coin": "ETH", "dir": "Close Short", "px": "2200", "sz": "1", "closedPnl": "-80", "fee": "4", "feeToken": "USDC", "time": 2000}},
                {"fill": {"coin": "ETH", "dir": "Open Short", "px": "2000", "sz": "1", "closedPnl": "0", "fee": "1", "feeToken": "USDC", "time": 1000}},
            ]
        if payload_type == "candleSnapshot" and payload["req"]["coin"] == "BTC":
            return [
                {"t": 1000, "o": "100000", "h": "111000", "l": "99000", "c": "110000", "v": "10", "n": 3},
            ]
        if payload_type == "candleSnapshot" and payload["req"]["coin"] == "ETH":
            return [
                {"t": 1000, "o": "2000", "h": "2210", "l": "1990", "c": "2200", "v": "50", "n": 10},
            ]
        if payload_type == "fundingHistory" and payload["coin"] == "BTC":
            return [{"coin": "BTC", "fundingRate": "0.0001", "premium": "0.0002", "time": 1000}]
        if payload_type == "fundingHistory" and payload["coin"] == "ETH":
            return [{"coin": "ETH", "fundingRate": "0.0002", "premium": "0.0003", "time": 1000}]
        raise AssertionError(f"Unexpected payload: {payload}")

    with patch.object(mod, "_post_info", side_effect=fake_post_info):
        exit_code = mod.main(["review", "0xabc", "--hours", "72", "--json"])

    stdout = capsys.readouterr().out
    rendered = json.loads(stdout)

    assert exit_code == 0
    assert rendered["summary"]["fill_count"] == 4
    assert rendered["summary"]["realized_pnl"] == 40.0
    assert rendered["summary"]["total_fees"] == 11.0
    assert rendered["summary"]["net_after_fees"] == 29.0
    assert len(rendered["coin_reviews"]) == 2
    eth_review = next(item for item in rendered["coin_reviews"] if item["coin"] == "ETH")
    assert round(eth_review["market_context"]["price_change_pct"], 2) == 10.0
    assert eth_review["market_context"]["average_funding_rate"] == 0.0002
    assert any("ETH" in finding and "rising market" in finding for finding in rendered["findings"])


def test_main_review_json_respects_coin_filter(capsys):
    mod = load_module()

    def fake_post_info(payload):
        if payload["type"] == "userFillsByTime":
            return [
                {"fill": {"coin": "BTC", "dir": "Close Long", "px": "110000", "sz": "0.1", "closedPnl": "120", "fee": "5", "feeToken": "USDC", "time": 4000}},
                {"fill": {"coin": "ETH", "dir": "Close Short", "px": "2200", "sz": "1", "closedPnl": "-80", "fee": "4", "feeToken": "USDC", "time": 2000}},
            ]
        if payload["type"] == "candleSnapshot":
            return [{"t": 1000, "o": "100000", "h": "111000", "l": "99000", "c": "110000", "v": "10", "n": 3}]
        if payload["type"] == "fundingHistory":
            return [{"coin": "BTC", "fundingRate": "0.0001", "premium": "0.0002", "time": 1000}]
        raise AssertionError(f"Unexpected payload: {payload}")

    with patch.object(mod, "_post_info", side_effect=fake_post_info):
        exit_code = mod.main(["review", "0xabc", "--coin", "BTC", "--json"])

    stdout = capsys.readouterr().out
    rendered = json.loads(stdout)

    assert exit_code == 0
    assert rendered["summary"]["fill_count"] == 1
    assert rendered["summary"]["unique_coins"] == 1
    assert rendered["coin_reviews"][0]["coin"] == "BTC"


def test_resolve_user_uses_env_fallback(monkeypatch):
    mod = load_module()
    monkeypatch.setenv("HYPERLIQUID_USER_ADDRESS", "0xenv123")

    assert mod._resolve_user("") == "0xenv123"
    assert mod._resolve_user(None) == "0xenv123"
    assert mod._resolve_user("0xcli456") == "0xcli456"


def test_resolve_user_errors_when_missing(monkeypatch, tmp_path):
    mod = load_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HYPERLIQUID_USER_ADDRESS", raising=False)

    try:
        mod._resolve_user("")
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected SystemExit when no user is provided")

    assert "HYPERLIQUID_USER_ADDRESS" in message


def test_main_state_json_uses_env_fallback(monkeypatch, capsys):
    mod = load_module()
    monkeypatch.setenv("HYPERLIQUID_USER_ADDRESS", "0xenv999")

    with patch.object(
        mod,
        "_post_info",
        return_value={"marginSummary": {"accountValue": "123"}, "assetPositions": [], "withdrawable": "50"},
    ) as mock_post:
        exit_code = mod.main(["state", "--json"])

    stdout = capsys.readouterr().out
    rendered = json.loads(stdout)

    assert exit_code == 0
    assert rendered["user"] == "0xenv999"
    assert mock_post.call_args[0][0]["user"] == "0xenv999"


def test_env_lookup_reads_hermes_dotenv(tmp_path, monkeypatch):
    mod = load_module()
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True)
    (hermes_home / ".env").write_text(
        "HYPERLIQUID_USER_ADDRESS=0xdotenv123\nHYPERLIQUID_API_URL=https://api.hyperliquid-testnet.xyz\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HYPERLIQUID_USER_ADDRESS", raising=False)
    monkeypatch.delenv("HYPERLIQUID_API_URL", raising=False)

    assert mod._env_lookup("HYPERLIQUID_USER_ADDRESS") == "0xdotenv123"
    assert mod._resolve_user("") == "0xdotenv123"
    assert mod._info_url() == "https://api.hyperliquid-testnet.xyz/info"


def test_user_dotenv_overrides_project_dotenv(tmp_path, monkeypatch):
    mod = load_module()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".env").write_text("HYPERLIQUID_USER_ADDRESS=0xproject\n", encoding="utf-8")

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text("HYPERLIQUID_USER_ADDRESS=0xuserhome\n", encoding="utf-8")

    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HYPERLIQUID_USER_ADDRESS", raising=False)

    assert mod._env_lookup("HYPERLIQUID_USER_ADDRESS") == "0xuserhome"


def test_main_export_json_writes_expected_contract(tmp_path, capsys):
    mod = load_module()
    output_path = tmp_path / "exports" / "btc-1h.json"

    def fake_post_info(payload):
        if payload["type"] == "candleSnapshot":
            return [
                {"t": 1000, "o": "100", "h": "110", "l": "95", "c": "108", "v": "50", "n": 4},
                {"t": 2000, "o": "108", "h": "115", "l": "107", "c": "112", "v": "60", "n": 5},
            ]
        if payload["type"] == "fundingHistory":
            return [
                {"coin": "BTC", "fundingRate": "0.0001", "premium": "0.0002", "time": 1500},
                {"coin": "BTC", "fundingRate": "0.0003", "premium": "0.0004", "time": 2000},
            ]
        raise AssertionError(f"Unexpected payload: {payload}")

    with patch.object(mod, "_post_info", side_effect=fake_post_info):
        exit_code = mod.main(
            [
                "export",
                "BTC",
                "--interval",
                "1h",
                "--hours",
                "24",
                "--end-time-ms",
                "5000",
                "--output",
                str(output_path),
                "--json",
            ]
        )

    stdout = capsys.readouterr().out
    rendered = json.loads(stdout)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert rendered["output_path"] == str(output_path)
    assert saved["schema_version"] == "hyperliquid-market-export-v1"
    assert saved["source"]["coin"] == "BTC"
    assert saved["window"]["start_time_ms"] == 5000 - 24 * 60 * 60 * 1000
    assert saved["window"]["end_time_ms"] == 5000
    assert saved["summary"]["candle_count"] == 2
    assert saved["summary"]["funding_count"] == 2
    assert round(saved["summary"]["price_change_pct"], 2) == 12.0
    assert saved["summary"]["average_funding_rate"] == 0.0002
    assert len(saved["candles"]) == 2
    assert len(saved["funding_history"]) == 2


def test_main_export_json_skips_funding_for_spot(tmp_path, capsys):
    mod = load_module()
    output_path = tmp_path / "purr-usdc.json"

    def fake_post_info(payload):
        if payload["type"] == "candleSnapshot":
            return [{"t": 1000, "o": "1", "h": "1.2", "l": "0.9", "c": "1.1", "v": "100", "n": 10}]
        raise AssertionError(f"Unexpected payload: {payload}")

    with patch.object(mod, "_post_info", side_effect=fake_post_info):
        exit_code = mod.main(
            [
                "export",
                "PURR/USDC",
                "--end-time-ms",
                "5000",
                "--output",
                str(output_path),
                "--json",
            ]
        )

    stdout = capsys.readouterr().out
    rendered = json.loads(stdout)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert rendered["summary"]["funding_count"] == 0
    assert saved["source"]["market_type"] == "spot"
    assert saved["funding_history"] == []

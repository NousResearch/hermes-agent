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





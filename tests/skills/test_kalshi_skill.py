from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "research"
    / "kalshi"
    / "scripts"
    / "kalshi.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("kalshi_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mock_response(payload: dict) -> MagicMock:
    """Build a context-manager mock that mimics urlopen()."""
    body = json.dumps(payload).encode("utf-8")
    fake = MagicMock()
    fake.__enter__.return_value.read.return_value = body
    return fake


def test_status_command_prints_exchange_state(capsys):
    mod = load_module()
    payload = {"exchange_active": True, "trading_active": True}

    with patch("urllib.request.urlopen", return_value=_mock_response(payload)):
        exit_code = mod.main(["status"])

    assert exit_code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == payload


def test_events_command_passes_filters_and_pagination(capsys):
    mod = load_module()
    payload = {"events": [{"event_ticker": "KX-A"}], "cursor": "next-page"}

    with patch("urllib.request.urlopen", return_value=_mock_response(payload)) as fake:
        exit_code = mod.main([
            "events",
            "--status", "open",
            "--series-ticker", "KXNBAGAME",
            "--limit", "5",
            "--cursor", "abc",
            "--with-markets",
        ])

    assert exit_code == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["cursor"] == "next-page"

    request_obj = fake.call_args.args[0]
    url = request_obj.full_url
    assert "/events?" in url
    assert "status=open" in url
    assert "series_ticker=KXNBAGAME" in url
    assert "cursor=abc" in url
    assert "with_nested_markets=true" in url
    assert "limit=5" in url


def test_orderbook_command_uses_path_param(capsys):
    mod = load_module()
    payload = {"orderbook_fp": {"yes_dollars": [["0.64", "10"]],
                                "no_dollars": [["0.36", "12"]]}}

    with patch("urllib.request.urlopen", return_value=_mock_response(payload)) as fake:
        exit_code = mod.main(["orderbook", "KX-FOO-BAR", "--depth", "5"])

    assert exit_code == 0
    request_obj = fake.call_args.args[0]
    assert "/markets/KX-FOO-BAR/orderbook" in request_obj.full_url
    assert "depth=5" in request_obj.full_url


def test_candles_command_computes_window_and_path(capsys):
    mod = load_module()
    payload = {"candlesticks": [{"end_period_ts": 1780337400}]}
    fixed_now = 1780337400

    with patch("urllib.request.urlopen", return_value=_mock_response(payload)) as fake, \
         patch("time.time", return_value=fixed_now):
        exit_code = mod.main([
            "candles",
            "KXNBAGAME",
            "KXNBAGAME-26JUN03NYKSAS-SAS",
            "--hours", "2",
            "--interval", "60",
        ])

    assert exit_code == 0
    request_obj = fake.call_args.args[0]
    url = request_obj.full_url
    assert "/series/KXNBAGAME/markets/KXNBAGAME-26JUN03NYKSAS-SAS/candlesticks" in url
    assert f"start_ts={fixed_now - 7200}" in url
    assert f"end_ts={fixed_now}" in url
    assert "period_interval=60" in url


def test_recent_trades_command_passes_ticker_and_cursor(capsys):
    mod = load_module()
    payload = {"trades": [{"trade_id": "t1", "yes_price_dollars": "0.64"}],
               "cursor": ""}

    with patch("urllib.request.urlopen", return_value=_mock_response(payload)) as fake:
        exit_code = mod.main([
            "recent-trades",
            "KX-FOO-BAR",
            "--limit", "3",
            "--cursor", "page2",
        ])

    assert exit_code == 0
    url = fake.call_args.args[0].full_url
    assert "/markets/trades?" in url
    assert "ticker=KX-FOO-BAR" in url
    assert "limit=3" in url
    assert "cursor=page2" in url


def test_http_error_returns_exit_code_2(capsys):
    """4xx response should bubble up as a non-zero exit, not a traceback."""
    import urllib.error

    mod = load_module()
    err = urllib.error.HTTPError(
        url="https://api.elections.kalshi.com/trade-api/v2/markets/MISSING",
        code=404,
        msg="Not Found",
        hdrs=None,
        fp=io.BytesIO(b'{"error":"not found"}'),
    )

    with patch("urllib.request.urlopen", side_effect=err):
        exit_code = mod.main(["market", "MISSING"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "404" in captured.err


def test_kalshi_error_is_runtime_error_subclass():
    """Allows downstream code to catch as RuntimeError without importing the module."""
    mod = load_module()
    assert issubclass(mod.KalshiError, RuntimeError)


@pytest.mark.parametrize("subcommand", [
    "status", "events", "event", "markets", "market",
    "orderbook", "recent-trades", "candles", "series",
])
def test_help_text_advertises_every_subcommand(subcommand):
    """Catch silent removal of a subcommand."""
    mod = load_module()
    parser = mod.build_parser()
    help_text = parser.format_help()
    assert subcommand in help_text

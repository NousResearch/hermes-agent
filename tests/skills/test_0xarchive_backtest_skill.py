"""Offline optional-skill packaging and trader contracts; no engine or API required."""

from contextlib import nullcontext
from email.message import Message
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import urllib.error
import urllib.parse

import pytest


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "optional-skills" / "finance" / "0xarchive-backtest"
IDENTIFIER = "official/finance/0xarchive-backtest"


@pytest.fixture
def trader(monkeypatch):
    spec = importlib.util.spec_from_file_location("quant_trader_under_test", SKILL / "scripts/trader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.urllib.request, "urlopen", Mock(side_effect=AssertionError("network forbidden")))
    monkeypatch.delenv("OXARCHIVE_API_KEY", raising=False)
    return module


def test_optional_bundle_loads_without_credentials(tmp_path, monkeypatch):
    from tools.skills_hub_official import OptionalSkillSource
    from tools import skills_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("OXARCHIVE_API_KEY", raising=False)
    source = OptionalSkillSource()
    source._optional_dir = ROOT / "optional-skills"
    monkeypatch.setattr(source, "_fetch_from_live_repo", Mock(side_effect=AssertionError("network forbidden")))
    matches = [m for m in source.list_local() if m.identifier == IDENTIFIER]
    assert len(matches) == 1
    assert "quant" in matches[0].tags
    bundle = source.fetch(IDENTIFIER)
    assert bundle is not None
    expected = {"SKILL.md", "LICENSE", "scripts/trader.py", "templates/mirror.py"}
    expected.update(f"references/{name}.md" for name in (
        "ports", "strategy-api", "data", "traders", "exploration", "ideas", "audit", "gauntlet"
    ))
    assert {name.replace("\\", "/") for name in bundle.files} == expected
    assert len(bundle.files) == len(expected)
    destination = tmp_path / "skills" / bundle.name
    for name, content in bundle.files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
    monkeypatch.setattr(skills_tool, "load_env", dict)
    capture = Mock(return_value={"missing_names": ["OXARCHIVE_API_KEY"],
                                 "setup_skipped": True, "gateway_setup_hint": None})
    monkeypatch.setattr(skills_tool, "_capture_required_environment_variables", capture)
    skills_tool.clear_skills_cache()
    try:
        loaded = json.loads(skills_tool.skill_view(bundle.name))
        assert loaded["success"], loaded
        assert loaded["skill_dir"] == str(destination)
        assert loaded["setup_needed"]
        assert loaded["required_environment_variables"][0]["name"] == "OXARCHIVE_API_KEY"
        for name in sorted(expected - {"SKILL.md", "LICENSE"}):
            result = json.loads(skills_tool.skill_view(bundle.name, file_path=name))
            assert result["success"], result
    finally:
        skills_tool.clear_skills_cache()


@pytest.mark.parametrize("value, expected", [
    ("1970-01-01", 0), ("1970-01-01T01:00:00+01:00", 0),
    ("1970-01-01T00:00:01Z", 1000), ("123", 123), (456, 456),
])
def test_timestamp_inputs_are_utc(trader, value, expected):
    assert trader.ms(value) == expected


def test_fetch_without_key_never_contacts_api(trader, tmp_path):
    with pytest.raises(SystemExit, match="OXARCHIVE_API_KEY"):
        trader.main(["fetch", "0xTEST", "--start", "2026-01-01", "--end", "2026-01-02",
                     "--out", str(tmp_path / "absent.json")])
    trader.urllib.request.urlopen.assert_not_called()
    assert not (tmp_path / "absent.json").exists()


def row(ts, position=1, symbol="BTC"):
    return {"timestamp": ts, "symbol": symbol, "end_position": str(position),
            "start_position": "0", "price": "100", "size": "1", "side": "B",
            "direction": "Open Long", "closed_pnl": "2", "fee": "0.1", "crossed": False}


@pytest.mark.parametrize("venue, prefix", [
    ("hyperliquid", "/v1/hyperliquid/wallets/0xTEST/positions/changes"),
    ("hip3", "/v1/hyperliquid/hip3/wallets/0xTEST/positions/changes"),
    ("lighter", "/v1/lighter/accounts/0xTEST/positions/changes"),
])
def test_fetch_pages_and_splits_windows(trader, tmp_path, monkeypatch, capsys, venue, prefix):
    monkeypatch.setenv("OXARCHIVE_API_KEY", "dummy-offline-key")
    start = trader.ms("2026-01-01")
    get = Mock(side_effect=[
        {"data": [row(start)], "meta": {"has_more": True, "next_cursor": "page2", "notice": "partial"}},
        {"data": [row(start + 1000, 0)], "meta": {"has_more": False}},
        {"data": [row(start + 30 * trader.DAY, 2, "ETH")], "meta": {}},
    ])
    monkeypatch.setattr(trader, "get", get)
    data = tmp_path / "data"
    (data / "BTC").mkdir(parents=True)
    out = tmp_path / "book.json"
    assert trader.main(["fetch", "0xTEST", "--venue", venue, "--start", str(start),
                        "--end", str(start + 31 * trader.DAY), "--data", str(data), "--out", str(out)]) == 0
    calls = get.call_args_list
    assert len(calls) == 3
    assert all(call.args[0] == prefix for call in calls)
    assert calls[0].args[1] == {"start": start, "end": start + 30 * trader.DAY, "limit": 5000}
    assert calls[1].args[1]["cursor"] == "page2"
    assert calls[2].args[1]["start"] == start + 30 * trader.DAY
    book = json.loads(out.read_text())
    assert len(book["changes"]) == 3
    assert book["in_dataset"] == ["BTC"]
    assert book["notices"] == ["partial"]
    assert book["changes"][0]["maker"] is True
    printed = capsys.readouterr()
    assert "dummy-offline-key" not in printed.out + printed.err


def test_fetch_reconstructs_gross_in_timestamp_order(trader, tmp_path, monkeypatch):
    """A closed BTC position must not overlap a later ETH position just because pages descend."""
    monkeypatch.setenv("OXARCHIVE_API_KEY", "dummy-offline-key")
    start = trader.ms("2026-01-01")
    changes = [row(start + 2000, 1, "ETH"), row(start + 1000, 0), row(start, 1)]
    monkeypatch.setattr(trader, "get", Mock(return_value={"data": changes, "meta": {}}))
    output = tmp_path / "book.json"
    assert trader.main(["fetch", "0xTEST", "--start", str(start),
                        "--end", str(start + trader.DAY), "--out", str(output)]) == 0
    book = json.loads(output.read_text())
    assert book["max_gross_usd"] == 100
    assert [change["ts"] for change in book["changes"]] == [start, start + 1000, start + 2000]


def test_mirror_rejects_book_without_positive_scale(tmp_path, monkeypatch):
    engine = ModuleType("oxarchive_backtest")
    engine.__dict__.update(Series=dict, Strategy=Mock(),
                           x=SimpleNamespace(cap=Mock(), rebalance=Mock()))
    monkeypatch.setitem(sys.modules, "oxarchive_backtest", engine)
    path = tmp_path / "empty-book.json"
    path.write_text(json.dumps({"changes": [], "max_gross_usd": 0, "in_dataset": []}))
    spec = importlib.util.spec_from_file_location("quant_mirror_under_test", SKILL / "templates/mirror.py")
    assert spec is not None and spec.loader is not None
    mirror = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mirror)
    with pytest.raises(ValueError, match="positive.*position value"):
        mirror.strategy(path=str(path))


@pytest.mark.parametrize("kind", ["json", "text", "transport"])
def test_get_redacts_key_from_errors(trader, monkeypatch, kind):
    key = "dummy-secret-do-not-print"
    if kind == "transport":
        error = urllib.error.URLError(f"proxy refused {key}")
    else:
        body = json.dumps({"error": f"invalid key {key}"}) if kind == "json" else f"invalid key {key}"
        error = urllib.error.HTTPError("https://example.invalid", 403, "Forbidden", Message(),
                                       io.BytesIO(body.encode()))
    monkeypatch.setattr(trader.urllib.request, "urlopen", Mock(side_effect=error))
    with pytest.raises(SystemExit) as caught:
        trader.get("/v1/hyperliquid/wallets/0xTEST/positions/changes", {}, key)
    assert key not in str(caught.value)
    assert "[redacted]" in str(caught.value)


def test_get_auth_headers_and_retry(trader, monkeypatch):
    error = urllib.error.HTTPError("https://example.invalid", 429, "rate limit", {"Retry-After": "0"},
                                   io.BytesIO(b'{"error_code":"rate_limited"}'))
    response = io.BytesIO(b'{"data": [], "meta": {}}')
    opener = Mock(side_effect=[error, response])
    monkeypatch.setattr(trader.urllib.request, "urlopen", opener)
    sleep = Mock()
    monkeypatch.setattr(trader.time, "sleep", sleep)
    assert trader.get("/test", {"start": 123}, "dummy-offline-key") == {"data": [], "meta": {}}
    request = opener.call_args.args[0]
    assert request.get_header("X-api-key") == "dummy-offline-key"
    assert request.get_header("0xarchive-version") == trader.VERSION
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query) == {"start": ["123"]}
    assert opener.call_args.kwargs["timeout"] == 60
    sleep.assert_called_once_with(0.0)


def test_insufficient_credits_does_not_retry(trader, monkeypatch):
    error = urllib.error.HTTPError("https://example.invalid", 429, "credits", {}, io.BytesIO(
        b'{"error_code":"insufficient_credits","error":"no credits","request_id":"req-offline"}'
    ))
    trader.urllib.request.urlopen.side_effect = error
    sleep = Mock()
    monkeypatch.setattr(trader.time, "sleep", sleep)
    with pytest.raises(SystemExit, match="insufficient_credits.*req-offline"):
        trader.get("/test", {}, "dummy-offline-key")
    assert trader.urllib.request.urlopen.call_count == 1
    sleep.assert_not_called()


def test_profile_saved_book_needs_neither_key_nor_engine(trader, tmp_path, capsys):
    changes = []
    for ts, pos in [(0, 1), (trader.HOUR, -1), (3 * trader.HOUR, 0)]:
        changes.append({"ts": ts, "symbol": "BTC", "end_position": pos, "price": 100,
                        "closed_pnl": 2, "fee": 0.1, "maker": True})
    book = {"address": "0xTEST", "start": 0, "end": 4 * trader.HOUR,
            "max_gross_usd": 100, "changes": changes, "in_dataset": []}
    path = tmp_path / "book.json"
    path.write_text(json.dumps(book))
    assert trader.main(["profile", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["round_trips"] == 2
    assert result["hold_hours_median"] == 1.5
    assert result["long_share_of_held_time"] == 0.333
    assert result["not_in_dataset"] == ["BTC"]
    assert result["realized_pnl"] == 6
    assert result["fees"] == 0.3
    assert result["maker_share"] == 1
    trader.urllib.request.urlopen.assert_not_called()


def report(fills, returns=(0.01, 0.02, 0.03)):
    return SimpleNamespace(fills=[SimpleNamespace(ts=t, market=m, size=s) for t, m, s in fills],
                           daily_returns=list(returns), attribution={"BTC": {"pnl": 10}},
                           stats={"sharpe": 1.2}, result_hash="offline-test-hash")


def test_compare_distinguishes_same_opposite_flat_and_extra(trader):
    h = trader.HOUR
    mirror = report([(0, "BTC", 1), (4 * h, "BTC", -1)])
    candidate = report([(0, "BTC", 1), (2 * h, "BTC", -2), (3 * h, "BTC", 1), (0, "ETH", 1)])
    result = trader.compare(mirror, candidate, 0, 4 * h)
    assert result["trader_held_market_hours"] == 3
    assert result["same_side"] == 0.333
    assert result["other_side"] == 0.333
    assert result["flat_while_trader_held"] == 0.333
    assert result["candidate_only_market_hours"] == 3
    assert result["daily_return_correlation"] == 1


def test_compare_empty_reports_has_no_spurious_scores(trader):
    empty = report([], ())
    result = trader.compare(empty, empty, 0, trader.HOUR)
    assert result["same_side"] is None
    assert result["daily_return_correlation"] is None


def test_replication_records_both_runs_and_passes_parameters(trader, tmp_path, monkeypatch, capsys):
    engine = ModuleType("oxarchive_backtest")
    engine.Config = Mock(return_value="config")
    engine.load_strategy = Mock(side_effect=["mirror-module", "candidate-module"])
    engine.strategy_of = Mock(side_effect=["mirror-strategy", "candidate-strategy"])
    engine.backtest = Mock(side_effect=[report([]), report([])])
    engine.Sealed = type("Sealed", (Exception,), {})
    engine.MissingData = type("MissingData", (Exception,), {})
    engine.study = SimpleNamespace(running=Mock(side_effect=lambda *args: nullcontext()))
    monkeypatch.setitem(sys.modules, "oxarchive_backtest", engine)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("# inert fixture; mocked loader never executes it\n")
    book = tmp_path / "trader.json"
    book.write_text("{}")
    assert trader.main(["replicate", str(book), str(candidate), "--start", "2026-01-01",
                        "--end", "2026-01-02", "-p", "n=3", "-c", "starting_cash=1000"]) == 0
    assert engine.study.running.call_count == 2
    assert engine.study.running.call_args_list[1].args == (str(candidate), candidate.read_text())
    assert engine.strategy_of.call_args_list[1].args == ("candidate-module", {"n": 3})
    engine.Config.assert_called_once_with(starting_cash=1000)
    assert engine.load_strategy.call_args_list[0].args[0] == SKILL / "templates/mirror.py"
    assert all(call.args[3] == "data/hyperliquid" for call in engine.backtest.call_args_list)
    output = json.loads(capsys.readouterr().out)
    assert output["trader_result_hash"] == output["candidate_result_hash"] == "offline-test-hash"

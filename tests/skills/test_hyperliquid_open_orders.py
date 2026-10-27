"""Regression for #42295: real CLI against synthetic loopback /info fixtures.

No exchange access, real wallet, signing or trades. Only stdlib HTTP is used.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


SCRIPT = (Path(__file__).resolve().parents[2] / "optional-skills" / "blockchain"
          / "hyperliquid" / "scripts" / "hyperliquid_client.py")
USER = "0x0000000000000000000000000000000000000000"


@pytest.fixture
def info_cli(tmp_path):
    requests = []
    responses = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            body = json.dumps(responses.get(payload["type"], {})).encode()
            self.send_response(200 if self.path == "/info" and payload["type"] in responses else 400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Allowlist OS locations only; never inherit credentials or proxy settings.
    env = {key: value for key, value in os.environ.items()
           if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT"}}
    for key in ("HOME", "USERPROFILE", "HERMES_HOME", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP"):
        location = tmp_path / key.lower()
        location.mkdir()
        env[key] = str(location)
    env.update(HYPERLIQUID_API_URL=f"http://127.0.0.1:{server.server_port}",
               HYPERLIQUID_USER_ADDRESS=USER, NO_PROXY="127.0.0.1", PYTHONUTF8="1")

    def run(*args):
        result = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=tmp_path,
                                env=env, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        return result.stdout

    try:
        yield run, responses, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def order(oid, coin, **fields):
    payload = dict(coin=coin, oid=oid, side="A", limitPx="100", sz="1", origSz="1",
                   timestamp=oid, orderType="Limit", isTrigger=False, triggerPx="0",
                   triggerCondition="N/A", reduceOnly=False, isPositionTpsl=False,
                   children=[])
    return {**payload, **fields}


@pytest.mark.parametrize("dex", ["", "xyz"])
def test_open_orders_cli_keeps_position_pnl_separate_from_recursive_tpsl(info_cli, dex):
    run, responses, requests = info_cli
    coin = f"{dex}:BTC" if dex else "BTC"
    stop = order(3, coin, orderType="Stop Market", isTrigger=True, triggerPx="90",
                 triggerCondition="Price below 90", reduceOnly=True, isPositionTpsl=True)
    take = order(2, coin, orderType="Take Profit Market", isTrigger=True, triggerPx="120",
                 triggerCondition="Price above 120", reduceOnly=True, isPositionTpsl=True,
                 children=[stop])
    responses.update(frontendOpenOrders=[order(1, coin, children=[take]), order(4, "ETH")],
                     clearinghouseState={"assetPositions": [{"position": {
                         "coin": coin, "szi": "2", "entryPx": "99", "positionValue": "220",
                         "unrealizedPnl": "22.25"}}]})
    flags = ["--dex", dex] if dex else []
    data = json.loads(run("open-orders", USER, *flags, "--limit", "0", "--json"))
    assert requests == [("/info", {"type": kind, "user": USER, "dex": dex})
                        for kind in ("frontendOpenOrders", "clearinghouseState")]
    assert data["user"] == USER and data["dex"] == dex
    assert data["count"] == 4
    rows = {row["oid"]: row for row in data["orders"]}
    assert set(rows) == {1, 2, 3, 4}
    assert rows[2]["parent_oid"] == 1 and rows[3]["parent_oid"] == 2
    assert rows[1]["parent_oid"] is None
    assert rows[2]["order_type"] == "Take Profit Market"
    assert rows[3]["trigger_px"] == "90"
    assert rows[2]["trigger_condition"] == "Price above 120"
    assert rows[2]["is_trigger"] is True
    assert rows[2]["reduce_only"] is True and rows[2]["is_position_tpsl"] is True
    assert rows[4]["coin"] == "ETH"  # Order without a position must not gain a PnL.
    assert all(not any("pnl" in key.lower() for key in row) for row in rows.values())
    assert len(data["positions"]) == 1
    assert data["positions"][0]["unrealized_pnl"] == "22.25"
    assert data["positions"][0]["size"] == "2"
    assert data["positions"][0]["entry_px"] == "99"
    table = run("open-orders", *flags, "--limit", "0")  # Env-default address.
    for label in ("Take Profit Market", "Stop Market", "Trigger", "Reduce Only",
                  "Position TP/SL", "Parent", "uPnL", "22.25", "Price below 90"):
        assert label in table
    limited = json.loads(run("open-orders", *flags, "--limit", "1", "--json"))
    assert limited["count"] == 4 and len(limited["orders"]) == 1
    assert limited["positions"] == data["positions"]
    assert len(json.loads(run("open-orders", *flags, "--limit", "-1", "--json"))["orders"]) == 4


@pytest.mark.parametrize("zero", [0, "0"])
def test_open_orders_cli_preserves_zero_values_and_empty_results(info_cli, zero):
    run, responses, requests = info_cli
    responses.update(frontendOpenOrders=[order(0, "BTC", sz=zero, limitPx=zero, triggerPx=zero)],
                     clearinghouseState={"assetPositions": [{"position": {
                         "coin": "BTC", "szi": zero, "entryPx": zero, "unrealizedPnl": zero}}]})
    data = json.loads(run("open-orders", "--json"))
    assert data["orders"][0]["size"] == zero
    assert data["orders"][0]["limit_px"] == zero
    assert data["orders"][0]["trigger_px"] == zero
    assert data["orders"][0]["reduce_only"] is False
    assert data["positions"][0]["unrealized_pnl"] == zero
    table = run("open-orders")
    position_line = next(line for line in table.splitlines() if line.startswith("BTC"))
    assert position_line.split()[1:5] == ["0", "0", "-", "0"]
    order_line = [line for line in table.splitlines() if line.startswith("BTC")][1]
    assert order_line.split()[1:9] == ["0", "-", "A", "0", "0", "Limit", "0", "N/A"]
    responses["frontendOpenOrders"] = []
    no_orders = json.loads(run("open-orders", "--json"))
    assert no_orders["orders"] == [] and no_orders["count"] == 0
    assert no_orders["positions"] == data["positions"]
    responses["clearinghouseState"] = {"assetPositions": []}
    empty = json.loads(run("open-orders", "--json"))
    assert empty["positions"] == [] and empty["orders"] == []
    table = run("open-orders")
    assert "Positions: 0" in table and "Open orders returned: 0 of 0" in table
    assert "(no data)" in table
    assert all(path == "/info" and body["dex"] == "" for path, body in requests)


def test_historical_orders_cli_stays_historical(info_cli):
    run, responses, requests = info_cli
    responses["historicalOrders"] = [
        {"order": order(7, "BTC"), "status": "filled", "statusTimestamp": 1000},
        {"order": order(8, "ETH"), "status": "canceled", "statusTimestamp": 2000},
    ]
    data = json.loads(run("orders", "--limit", "1", "--json"))
    assert requests == [("/info", {"type": "historicalOrders", "user": USER})]
    assert data["count"] == 2 and len(data["orders"]) == 1
    assert data["orders"][0]["oid"] == 8
    assert data["orders"][0]["status"] == "canceled"
    assert "positions" not in data
    assert "canceled" in run("orders", "--limit", "1")

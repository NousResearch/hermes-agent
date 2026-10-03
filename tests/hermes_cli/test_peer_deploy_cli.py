"""``hermes peer deploy`` — the health gate's CLI caller.

Covers: the real parser admits ``peer deploy <target>`` with wait/json flags;
Tier-1 freshness + Tier-2 writer-identity filtering over a /health/detailed
payload (the #130708 stale-feishu shape); the poll loop (unhealthy-then-
healthy exits 0, never-healthy exits 1, unreachable exits 1, unknown peer
exits 2).
"""

from __future__ import annotations

import argparse
import datetime
import json
import urllib.error
from argparse import Namespace
from types import SimpleNamespace

from hermes_cli import peer_deploy as gate
from hermes_cli.subcommands import peer as peer_cmd


def _fresh_updated_at() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _detailed(platforms, *, state="running", pid=100, start=1700000000.0):
    return {
        "status": "ok",
        "gateway_state": state,
        "platforms": platforms,
        "pid": pid,
        "start_time": start,
        "updated_at": _fresh_updated_at(),
    }


def _130708_platforms():
    live = {"writer_pid": 100, "writer_start_time": 1700000000.0}
    return {
        "api_server": {**live, "state": "connected"},
        "google_chat": {**live, "state": "connected"},
        "feishu": {"writer_pid": 99, "writer_start_time": 1699000000.0, "state": "retrying"},
    }


def test_parser_admits_peer_deploy():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers()
    peer_cmd.build_peer_parser(subs)
    args = parser.parse_args(["peer", "deploy", "spark", "--wait-seconds", "10", "--json"])
    assert args.peer_action == "deploy"
    assert args.target == "spark"
    assert args.wait_seconds == "10"
    assert args.json is True
    assert args.func is peer_cmd.cmd_peer


def test_remote_gate_healthy_on_130708_payload():
    result = gate.evaluate_remote_gate(_detailed(_130708_platforms()))
    assert result["healthy"] is True
    assert result["tier1_fresh"] is True
    assert result["tier2_healthy"] is True
    assert result["all_owned"] == ["api_server", "google_chat"]
    assert result["missing"] == []


def test_remote_gate_fails_stale_heartbeat():
    payload = _detailed(_130708_platforms())
    payload["updated_at"] = "2020-01-01T00:00:00Z"
    result = gate.evaluate_remote_gate(payload)
    assert result["healthy"] is False
    assert result["tier1_fresh"] is False


def test_remote_gate_fails_stopped_state():
    result = gate.evaluate_remote_gate(_detailed(_130708_platforms(), state="stopped"))
    assert result["healthy"] is False


def _deploy_args(**over):
    base = {"target": "spark", "wait_seconds": 0, "json": False}
    base.update(over)
    return Namespace(**base)


def _resolve_ok(monkeypatch):
    monkeypatch.setattr(
        peer_cmd, "_resolve_peer_target",
        lambda target: ("spark", None, {"url": "http://spark.lan:8377"}, "k"))
    monkeypatch.setattr(peer_cmd, "_base_url", lambda peer, profile: "http://spark.lan:8377")


def test_deploy_poll_unhealthy_then_healthy_exits_0(monkeypatch, capsys):
    _resolve_ok(monkeypatch)
    bad = _detailed({"api_server": {
        "writer_pid": 100, "writer_start_time": 1700000000.0, "state": "retrying"}})
    good = _detailed({"api_server": {
        "writer_pid": 100, "writer_start_time": 1700000000.0, "state": "connected"}})
    calls = {"n": 0}

    def _fake_request(url, key, **kw):
        assert url == "http://spark.lan:8377/health/detailed"
        calls["n"] += 1
        return good if calls["n"] >= 3 else bad

    monkeypatch.setattr(peer_cmd, "_request", _fake_request)
    status = peer_cmd._peer_deploy(_deploy_args(wait_seconds=60))
    assert status == 0
    assert calls["n"] == 3
    assert "healthy" in capsys.readouterr().out


def test_deploy_never_healthy_exits_1(monkeypatch, capsys):
    _resolve_ok(monkeypatch)
    bad = _detailed({"api_server": {
        "writer_pid": 100, "writer_start_time": 1700000000.0, "state": "retrying"}})
    monkeypatch.setattr(peer_cmd, "_request", lambda *a, **k: bad)
    status = peer_cmd._peer_deploy(_deploy_args(wait_seconds=0))
    assert status == 1
    err = capsys.readouterr().err
    assert "api_server" in err and "Roll back" in err


def test_deploy_unreachable_exits_1(monkeypatch, capsys):
    _resolve_ok(monkeypatch)

    def _boom(*a, **k):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(peer_cmd, "_request", _boom)
    status = peer_cmd._peer_deploy(_deploy_args(wait_seconds=0))
    assert status == 1
    assert "never answered" in capsys.readouterr().err


def test_deploy_unknown_peer_exits_1(monkeypatch, capsys):
    # Same convention as cmd_peer: an unregistered peer is a peer error (1),
    # not a usage error — only a malformed target is exit 2.
    monkeypatch.setattr(
        peer_cmd, "_resolve_peer_target",
        lambda target: (_ for _ in ()).throw(LookupError("No peer named 'ghost'.")))
    assert peer_cmd._peer_deploy(_deploy_args()) == 1


def test_deploy_negative_wait_exits_2(capsys):
    assert peer_cmd._peer_deploy(_deploy_args(wait_seconds=-5)) == 2

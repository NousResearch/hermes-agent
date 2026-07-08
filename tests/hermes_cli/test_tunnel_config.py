import pytest
from hermes_cli.tunnel_config import resolve_tunnel_config, parse_origin


def _stub_config(monkeypatch, tunnel_cfg):
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"tunnel": tunnel_cfg} if tunnel_cfg else {})


def test_defaults_when_nothing_set(monkeypatch):
    _stub_config(monkeypatch, None)
    for v in (
        "HERMES_TUNNEL_ZONE", "HERMES_TUNNEL_NAME", "HERMES_TUNNEL_CREDS",
        "HERMES_TUNNEL_METRICS_PORT", "HERMES_TUNNEL_IDLE_TIMEOUT",
        "HERMES_TUNNEL_DRAIN_SECONDS", "HERMES_TUNNEL_POLL_INTERVAL", "HERMES_TUNNEL_ADMIN",
    ):
        monkeypatch.delenv(v, raising=False)
    c = resolve_tunnel_config()
    assert c["zone"] == ""
    assert c["idle_timeout_seconds"] == 1800
    assert c["drain_seconds"] == 15
    assert c["poll_interval_seconds"] == 5
    assert c["metrics_port"] == 0
    assert c["admin"] == []
    assert c["routes"] == []


def test_env_overrides_config(monkeypatch):
    _stub_config(monkeypatch, {"zone": "config.example", "tunnel_name": "cfg-name",
                               "idle_timeout_seconds": 600})
    monkeypatch.setenv("HERMES_TUNNEL_ZONE", "noit2.com")
    monkeypatch.setenv("HERMES_TUNNEL_NAME", "env-name")
    monkeypatch.setenv("HERMES_TUNNEL_IDLE_TIMEOUT", "1800")
    c = resolve_tunnel_config()
    assert c["zone"] == "noit2.com"
    assert c["tunnel_name"] == "env-name"
    assert c["idle_timeout_seconds"] == 1800


def test_empty_env_falls_back_to_config(monkeypatch):
    _stub_config(monkeypatch, {"zone": "noit2.com", "tunnel_name": "cfg-name"})
    monkeypatch.setenv("HERMES_TUNNEL_ZONE", "")   # empty -> treated as unset
    c = resolve_tunnel_config()
    assert c["zone"] == "noit2.com"


def test_admin_env_csv(monkeypatch):
    _stub_config(monkeypatch, {"admin": ["a"]})
    monkeypatch.setenv("HERMES_TUNNEL_ADMIN", "alice,bob")
    c = resolve_tunnel_config()
    assert c["admin"] == ["alice", "bob"]


def test_parse_origin():
    assert parse_origin("alice=127.0.0.1:3000") == {
        "subdomain": "alice", "host": "127.0.0.1", "port": 3000}


def test_parse_origin_rejects_bad():
    with pytest.raises(ValueError):
        parse_origin("no-port")
    with pytest.raises(ValueError):
        parse_origin("alice=not-a-port")


def test_cli_origins_override_config_routes(monkeypatch):
    _stub_config(monkeypatch, {"zone": "noit2.com",
                               "routes": [{"subdomain": "alice", "host": "127.0.0.1", "port": 3000}]})
    c = resolve_tunnel_config(cli_origins=["alice=127.0.0.1:9000", "alice-api=127.0.0.1:8080"])
    sub_to_port = {r["subdomain"]: r["port"] for r in c["routes"]}
    assert sub_to_port == {"alice": 9000, "alice-api": 8080}
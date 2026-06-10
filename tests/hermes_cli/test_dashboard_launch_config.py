from __future__ import annotations

import argparse
import sys


def _ns(**kw):
    defaults = dict(
        port=9119,
        host="127.0.0.1",
        no_open=False,
        insecure=False,
        stop=False,
        status=False,
        skip_build=True,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_dashboard_launch_uses_env_overrides_when_flags_not_explicit(
    monkeypatch, tmp_path
):
    from hermes_cli import main as mod
    import hermes_cli.config as config_mod

    monkeypatch.setenv("HERMES_DASHBOARD_HOST", "0.0.0.0")
    monkeypatch.setenv("HERMES_DASHBOARD_PORT", "9120")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("ok", encoding="utf-8")
    monkeypatch.setenv("HERMES_WEB_DIST", str(dist))
    monkeypatch.setattr(mod, "_sync_bundled_skills_quietly", lambda: None)

    start_calls: dict = {}

    def fake_start_server(**kwargs):
        start_calls.update(kwargs)

    fake_ws = type("FakeWS", (), {"start_server": staticmethod(fake_start_server)})

    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {"dashboard": {"host": "127.0.0.9", "port": 9999}},
    )
    fake_plugins = type("FakePlugins", (), {"discover_plugins": staticmethod(lambda: None)})
    monkeypatch.setitem(sys.modules, "hermes_cli.web_server", fake_ws)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", fake_plugins)

    mod.cmd_dashboard(_ns())

    assert start_calls["host"] == "0.0.0.0"
    assert start_calls["port"] == 9120


def test_dashboard_launch_explicit_flags_override_config_and_env(monkeypatch, tmp_path):
    from hermes_cli import main as mod
    import hermes_cli.config as config_mod

    monkeypatch.setenv("HERMES_DASHBOARD_HOST", "0.0.0.0")
    monkeypatch.setenv("HERMES_DASHBOARD_PORT", "9120")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("ok", encoding="utf-8")
    monkeypatch.setenv("HERMES_WEB_DIST", str(dist))
    monkeypatch.setattr(mod, "_sync_bundled_skills_quietly", lambda: None)

    start_calls: dict = {}

    def fake_start_server(**kwargs):
        start_calls.update(kwargs)

    fake_ws = type("FakeWS", (), {"start_server": staticmethod(fake_start_server)})

    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {"dashboard": {"host": "10.0.0.5", "port": 9555}},
    )
    fake_plugins = type("FakePlugins", (), {"discover_plugins": staticmethod(lambda: None)})
    monkeypatch.setitem(sys.modules, "hermes_cli.web_server", fake_ws)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", fake_plugins)

    mod.cmd_dashboard(
        _ns(
            host="127.0.0.1",
            port=9119,
            host_explicit=True,
            port_explicit=True,
        )
    )

    assert start_calls["host"] == "127.0.0.1"
    assert start_calls["port"] == 9119


def test_load_dashboard_network_config_reads_config_and_env(monkeypatch):
    import hermes_cli.web_server as ws

    monkeypatch.setattr(
        ws,
        "load_config",
        lambda: {
            "dashboard": {
                "cors_origins": ["https://dash.example", "https://dash.example"],
                "allowed_hosts": ["tailscale-node.ts.net"],
            }
        },
    )
    monkeypatch.delenv("HERMES_DASHBOARD_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("HERMES_DASHBOARD_ALLOWED_HOSTS", raising=False)

    cors_origins, allowed_hosts = ws._load_dashboard_network_config()
    assert cors_origins == ("https://dash.example",)
    assert allowed_hosts == ("tailscale-node.ts.net",)

    monkeypatch.setenv(
        "HERMES_DASHBOARD_CORS_ORIGINS",
        "https://env.example, https://dash.example",
    )
    monkeypatch.setenv(
        "HERMES_DASHBOARD_ALLOWED_HOSTS",
        "env-node.ts.net, tailscale-node.ts.net",
    )

    cors_origins, allowed_hosts = ws._load_dashboard_network_config()
    assert cors_origins == ("https://env.example", "https://dash.example")
    assert allowed_hosts == ("env-node.ts.net", "tailscale-node.ts.net")

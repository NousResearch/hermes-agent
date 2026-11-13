"""Behaviour contract for ``dashboard.trusted_hosts`` — operator-declared Host aliases.

A dashboard served behind Tailscale Serve, MagicDNS or a reverse proxy is reached
under hostnames that are neither the bind address nor ``dashboard.public_url``.
The Host gate (GHSA-ppp5-vxwm-4cf7) must accept exactly the declared aliases —
never a wildcard, suffix, or malformed entry — and must still fail closed on
everything else.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_repo = str(Path(__file__).resolve().parents[1])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _write_dashboard_config(home: Path, dashboard: dict) -> None:
    import yaml
    (home / "config.yaml").write_text(
        yaml.safe_dump({"dashboard": dashboard}), encoding="utf-8"
    )


class TestConfiguredTrustedHosts:
    def test_aliases_are_normalized_and_exact(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_dashboard_config(
            tmp_path,
            {"trusted_hosts": ["m5-studio.tailnet.ts.net:443", "HERMES.local", "hermes.lan"]},
        )
        from hermes_cli.web_server_boundary import _configured_trusted_hosts, _is_accepted_host

        hosts = _configured_trusted_hosts()
        assert hosts == frozenset({"m5-studio.tailnet.ts.net", "hermes.local", "hermes.lan"})
        # Declared hosts are accepted (with or without a port suffix)...
        assert _is_accepted_host("m5-studio.tailnet.ts.net:9119", "127.0.0.1", hosts)
        # ...and never weaken the rebinding check for lookalikes.
        assert not _is_accepted_host("evil.m5-studio.tailnet.ts.net", "127.0.0.1", hosts)

    def test_malformed_entries_fail_closed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_dashboard_config(
            tmp_path,
            {"trusted_hosts": ["http://m5-studio.local", "m5 studio", 42], "public_url": ""},
        )
        from hermes_cli.web_server_boundary import _dashboard_public_hosts

        assert _dashboard_public_hosts() == frozenset()

    def test_public_url_and_aliases_share_one_trust_set(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_dashboard_config(
            tmp_path,
            {
                "public_url": "https://m5-studio.snow-pinecone.ts.net",
                "trusted_hosts": ["m5-studio"],
            },
        )
        from hermes_cli.web_server_boundary import _dashboard_public_hosts

        assert _dashboard_public_hosts() == frozenset(
            {"m5-studio.snow-pinecone.ts.net", "m5-studio"}
        )

    def test_declared_alias_engages_the_auth_gate(self, tmp_path, monkeypatch):
        """A loopback backend served under a non-loopback alias is as exposed as one
        with a non-loopback public_url, so the sign-in gate must engage."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_dashboard_config(tmp_path, {"trusted_hosts": ["m5-studio"]})
        from hermes_cli.web_server import should_require_dashboard_auth

        assert should_require_dashboard_auth("127.0.0.1")
        # A loopback-only alias does not engage the gate.
        _write_dashboard_config(tmp_path, {"trusted_hosts": ["localhost"]})
        assert not should_require_dashboard_auth("127.0.0.1")


class TestMiddlewareBehaviour:
    def test_declared_alias_accepted_unknown_host_rejected_with_remedy(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_dashboard_config(tmp_path, {"trusted_hosts": ["m5-studio"]})
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app, _dashboard_public_hosts

        app.state.bound_host = "127.0.0.1"
        app.state.trusted_public_hosts = _dashboard_public_hosts()
        try:
            client = TestClient(app)
            assert client.get("/api/health", headers={"Host": "m5-studio"}).status_code != 400
            resp = client.get("/api/health", headers={"Host": "evil.example"})
            assert resp.status_code == 400
            # The rejection names the offending Host so the operator can diagnose it.
            assert "evil.example" in resp.json()["detail"]
            assert "trusted_hosts" in resp.json()["detail"]
        finally:
            del app.state.bound_host
            del app.state.trusted_public_hosts

    def test_without_trusted_hosts_the_gate_stays_closed(self, tmp_path, monkeypatch):
        """Back-compat: no aliases declared → only the bound host is accepted."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        app.state.bound_host = "127.0.0.1"
        app.state.trusted_public_hosts = frozenset()
        try:
            client = TestClient(app)
            assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 400
        finally:
            del app.state.bound_host
            del app.state.trusted_public_hosts


class TestWebSocketGate:
    def test_ws_upgrade_follows_the_same_alias_set(self):
        from hermes_cli.web_server_boundary import ws_host_origin_reason

        state = SimpleNamespace(
            bound_host="127.0.0.1",
            trusted_public_hosts=frozenset({"m5-studio"}),
        )

        class _Upgrade:
            def __init__(self, headers):
                self.headers = headers

        assert ws_host_origin_reason(
            _Upgrade({"host": "m5-studio", "origin": "https://m5-studio"}), state
        ) is None
        assert ws_host_origin_reason(
            _Upgrade({"host": "evil.m5-studio", "origin": ""}), state
        ) is not None

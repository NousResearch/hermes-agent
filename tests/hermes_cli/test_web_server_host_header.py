"""Tests for GHSA-ppp5-vxwm-4cf7 — Host-header validation.

DNS rebinding defence: a victim browser that has the dashboard open
could be tricked into fetching from an attacker-controlled hostname
that TTL-flips to 127.0.0.1. Same-origin / CORS checks won't help —
the browser now treats the attacker origin as same-origin. Validating
the Host header at the application layer rejects the attack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_repo = str(Path(__file__).resolve().parents[1])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


class TestHostHeaderValidator:
    """Unit test the _is_accepted_host helper directly — cheaper and
    more thorough than spinning up the full FastAPI app."""

    def test_loopback_bind_accepts_loopback_names(self):
        from hermes_cli.web_server import _is_accepted_host

        for bound in ("127.0.0.1", "localhost", "::1"):
            for host_header in (
                "127.0.0.1", "127.0.0.1:9119",
                "localhost", "localhost:9119",
                "[::1]", "[::1]:9119",
            ):
                assert _is_accepted_host(host_header, bound), (
                    f"bound={bound} must accept host={host_header}"
                )

    def test_loopback_bind_rejects_attacker_hostnames(self):
        """The core rebinding defence: attacker-controlled hosts that
        TTL-flip to 127.0.0.1 must be rejected."""
        from hermes_cli.web_server import _is_accepted_host

        for bound in ("127.0.0.1", "localhost"):
            for attacker in (
                "evil.example",
                "evil.example:9119",
                "rebind.attacker.test:80",
                "localhost.attacker.test",  # subdomain trick
                "127.0.0.1.evil.test",  # lookalike IP prefix
                "",  # missing Host
            ):
                assert not _is_accepted_host(attacker, bound), (
                    f"bound={bound} must reject attacker host={attacker!r}"
                )

    def test_zero_zero_bind_accepts_anything_without_allowlist(self):
        """0.0.0.0 without an allowlist preserves the historical explicit
        all-interfaces opt-in behavior for operators who pass --insecure."""
        from hermes_cli.web_server import _is_accepted_host

        for host in ("10.0.0.5", "evil.example", "my-server.corp.net"):
            assert _is_accepted_host(host, "0.0.0.0")
            assert _is_accepted_host(host + ":9119", "0.0.0.0")

    def test_zero_zero_bind_can_be_restricted_by_allowed_hosts(self):
        """LAN binds can still protect the app layer when operators provide
        explicit allowed Host values."""
        from hermes_cli.web_server import _is_accepted_host

        allowed_hosts = {
            "hermes-dashboard.transformers.lan",
            "10.0.0.196",
            "localhost",
        }

        assert _is_accepted_host(
            "hermes-dashboard.transformers.lan:9443",
            "0.0.0.0",
            allowed_hosts=allowed_hosts,
        )
        assert _is_accepted_host(
            "10.0.0.196:9443",
            "0.0.0.0",
            allowed_hosts=allowed_hosts,
        )
        assert not _is_accepted_host(
            "evil.example:9443",
            "0.0.0.0",
            allowed_hosts=allowed_hosts,
        )
        assert _is_accepted_host(
            "localhost:9119",
            "127.0.0.1",
            allowed_hosts={"hermes-dashboard.transformers.lan"},
        )
        assert _is_accepted_host(
            "hermes-dashboard.transformers.lan:9443",
            "127.0.0.1",
            allowed_hosts={"hermes-dashboard.transformers.lan"},
        )

    def test_explicit_non_loopback_bind_requires_exact_match(self):
        """If the operator bound to a specific non-loopback hostname,
        the Host header must match exactly."""
        from hermes_cli.web_server import _is_accepted_host

        assert _is_accepted_host("my-server.corp.net", "my-server.corp.net")
        assert _is_accepted_host("my-server.corp.net:9119", "my-server.corp.net")
        # Different host — reject
        assert not _is_accepted_host("evil.example", "my-server.corp.net")
        # Loopback — reject (we bound to a specific non-loopback name)
        assert not _is_accepted_host("localhost", "my-server.corp.net")

    def test_case_insensitive_comparison(self):
        """Host headers are case-insensitive per RFC — accept variations."""
        from hermes_cli.web_server import _is_accepted_host

        assert _is_accepted_host("LOCALHOST", "127.0.0.1")
        assert _is_accepted_host("LocalHost:9119", "127.0.0.1")


class TestHostHeaderMiddleware:
    """End-to-end test via the FastAPI app — verify the middleware
    rejects bad Host headers with 400."""

    def test_rebinding_request_rejected(self):
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        # Simulate start_server having set the bound_host
        app.state.bound_host = "127.0.0.1"
        try:
            client = TestClient(app)
            # The TestClient sends Host: testserver by default — which is
            # NOT a loopback alias, so the middleware must reject it.
            resp = client.get(
                "/api/status",
                headers={"Host": "evil.example"},
            )
            assert resp.status_code == 400
            assert "Invalid Host header" in resp.json()["detail"]
        finally:
            # Clean up so other tests don't inherit the bound_host
            if hasattr(app.state, "bound_host"):
                del app.state.bound_host

    def test_legit_loopback_request_accepted(self):
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        app.state.bound_host = "127.0.0.1"
        try:
            client = TestClient(app)
            # /api/status is in _PUBLIC_API_PATHS — passes auth — so the
            # only thing that can reject is the host header middleware
            resp = client.get(
                "/api/status",
                headers={"Host": "localhost:9119"},
            )
            # Either 200 (endpoint served) or some other non-400 —
            # just not the host-rejection 400
            assert resp.status_code != 400 or (
                "Invalid Host header" not in resp.json().get("detail", "")
            )
        finally:
            if hasattr(app.state, "bound_host"):
                del app.state.bound_host

    def test_no_bound_host_skips_validation(self):
        """If app.state.bound_host isn't set (e.g. running under test
        infra without calling start_server), middleware must pass through
        rather than crash."""
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        # Make sure bound_host isn't set
        if hasattr(app.state, "bound_host"):
            del app.state.bound_host

        client = TestClient(app)
        resp = client.get("/api/status")
        # Should get through to the status endpoint, not a 400
        assert resp.status_code != 400


class TestDashboardServerStartup:
    def test_start_server_passes_tls_files_to_uvicorn(self, tmp_path, monkeypatch):
        import uvicorn
        from hermes_cli.web_server import app, start_server

        cert = tmp_path / "dashboard.crt"
        key = tmp_path / "dashboard.key"
        cert.write_text("test cert")
        key.write_text("test key")
        calls = []

        def fake_run(*args, **kwargs):
            calls.append((args, kwargs))

        monkeypatch.setattr(uvicorn, "run", fake_run)

        start_server(
            host="0.0.0.0",
            port=9443,
            open_browser=False,
            allow_public=True,
            tls_cert=str(cert),
            tls_key=str(key),
            allowed_hosts={"hermes-dashboard.transformers.lan", "10.0.0.196"},
        )

        assert calls == [
            (
                (app,),
                {
                    "host": "0.0.0.0",
                    "port": 9443,
                    "log_level": "warning",
                    "proxy_headers": False,
                    "ssl_certfile": str(cert),
                    "ssl_keyfile": str(key),
                },
            )
        ]
        assert app.state.allowed_hosts == {
            "hermes-dashboard.transformers.lan",
            "10.0.0.196",
        }

    def test_start_server_rejects_partial_tls_configuration(self, tmp_path):
        from hermes_cli.web_server import start_server

        cert = tmp_path / "dashboard.crt"
        cert.write_text("test cert")

        with pytest.raises(SystemExit, match="both --tls-cert and --tls-key"):
            start_server(
                host="127.0.0.1",
                port=9443,
                open_browser=False,
                tls_cert=str(cert),
            )

    def test_tls_dashboard_uses_wss_sidecar_url(self, monkeypatch):
        from hermes_cli.web_server import _build_sidecar_url, app

        monkeypatch.setattr(app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(app.state, "bound_port", 9443, raising=False)
        monkeypatch.setattr(app.state, "dashboard_tls_enabled", True, raising=False)

        url = _build_sidecar_url("abc-123")
        assert url is not None
        assert url.startswith("wss://127.0.0.1:9443/api/pub?")

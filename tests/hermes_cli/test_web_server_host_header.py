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

    def test_zero_zero_bind_accepts_anything(self):
        """0.0.0.0 means operator explicitly opted into all-interfaces
        (requires --insecure). No Host-layer defence is possible — rely
        on operator network controls."""
        from hermes_cli.web_server import _is_accepted_host

        for host in ("10.0.0.5", "evil.example", "my-server.corp.net"):
            assert _is_accepted_host(host, "0.0.0.0")
            assert _is_accepted_host(host + ":9119", "0.0.0.0")

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


class TestHostHeaderAdditionalHosts:
    """HERMES_DASHBOARD_ADDITIONAL_HOSTS lets operators add proxy hostnames
    (e.g. tailnet MagicDNS names) to the accept list on a loopback bind,
    without resorting to --insecure / all-interfaces. The DNS-rebinding
    defence must still hold for any host NOT in the list."""

    def test_loopback_bind_accepts_listed_host(self, monkeypatch):
        from hermes_cli.web_server import _is_accepted_host

        monkeypatch.setenv(
            "HERMES_DASHBOARD_ADDITIONAL_HOSTS", "magic.tailnet.ts.net"
        )
        for bound in ("127.0.0.1", "localhost", "::1"):
            assert _is_accepted_host("magic.tailnet.ts.net", bound)
            assert _is_accepted_host("magic.tailnet.ts.net:9119", bound)

    def test_loopback_bind_still_rejects_attacker_hosts_with_env_set(
        self, monkeypatch
    ):
        """Core safety property: adding one trusted host MUST NOT widen
        the accept list to any other attacker-controlled hostname that
        TTL-flips to 127.0.0.1."""
        from hermes_cli.web_server import _is_accepted_host

        monkeypatch.setenv(
            "HERMES_DASHBOARD_ADDITIONAL_HOSTS", "magic.tailnet.ts.net"
        )
        for attacker in (
            "evil.example",
            "evil.example:9119",
            "magic.tailnet.ts.net.attacker.test",  # suffix attack
            "attacker.magic.tailnet.ts.net",       # subdomain ≠ listed exact
        ):
            assert not _is_accepted_host(attacker, "127.0.0.1"), (
                f"attacker host {attacker!r} must still be rejected"
            )

    def test_case_insensitive_match_against_env_entries(self, monkeypatch):
        """Env entries are normalized to lowercase; Host headers are
        compared case-insensitively per RFC 7230."""
        from hermes_cli.web_server import _is_accepted_host

        monkeypatch.setenv(
            "HERMES_DASHBOARD_ADDITIONAL_HOSTS", "Magic.Tailnet.TS.NET"
        )
        assert _is_accepted_host("magic.tailnet.ts.net", "127.0.0.1")
        assert _is_accepted_host("MAGIC.TAILNET.TS.NET", "127.0.0.1")

    def test_comma_separated_list_accepts_each_entry(self, monkeypatch):
        from hermes_cli.web_server import _is_accepted_host

        monkeypatch.setenv(
            "HERMES_DASHBOARD_ADDITIONAL_HOSTS",
            "a.example, b.example ,  c.example",
        )
        assert _is_accepted_host("a.example", "127.0.0.1")
        assert _is_accepted_host("b.example", "127.0.0.1")
        assert _is_accepted_host("c.example", "127.0.0.1")
        assert not _is_accepted_host("d.example", "127.0.0.1")

    def test_empty_env_var_is_no_op(self, monkeypatch):
        """An unset or empty env var must not change existing behaviour."""
        from hermes_cli.web_server import _is_accepted_host

        monkeypatch.delenv("HERMES_DASHBOARD_ADDITIONAL_HOSTS", raising=False)
        assert _is_accepted_host("localhost", "127.0.0.1")
        assert not _is_accepted_host("evil.example", "127.0.0.1")

        monkeypatch.setenv("HERMES_DASHBOARD_ADDITIONAL_HOSTS", "   ,  , ")
        assert _is_accepted_host("localhost", "127.0.0.1")
        assert not _is_accepted_host("evil.example", "127.0.0.1")

    def test_additional_hosts_not_consulted_on_non_loopback_bind(
        self, monkeypatch
    ):
        """When the operator bound to a specific non-loopback hostname,
        the validator requires exact bound-host match — the additional
        hosts list is a loopback-bind-only convenience and must not
        weaken explicit non-loopback binds."""
        from hermes_cli.web_server import _is_accepted_host

        monkeypatch.setenv(
            "HERMES_DASHBOARD_ADDITIONAL_HOSTS", "magic.tailnet.ts.net"
        )
        assert not _is_accepted_host(
            "magic.tailnet.ts.net", "my-server.corp.net"
        )
        assert _is_accepted_host("my-server.corp.net", "my-server.corp.net")

    def test_zero_zero_bind_still_accepts_anything(self, monkeypatch):
        """0.0.0.0 (--insecure) accepts everything regardless of
        HERMES_DASHBOARD_ADDITIONAL_HOSTS — the env var is meaningful
        only on loopback binds."""
        from hermes_cli.web_server import _is_accepted_host

        monkeypatch.setenv(
            "HERMES_DASHBOARD_ADDITIONAL_HOSTS", "magic.tailnet.ts.net"
        )
        assert _is_accepted_host("anything.example", "0.0.0.0")
        assert _is_accepted_host("magic.tailnet.ts.net", "0.0.0.0")


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

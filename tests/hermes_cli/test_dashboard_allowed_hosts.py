"""Tests for dashboard reverse-proxy Host allowlisting."""

from hermes_cli.web_server import _is_accepted_host


def test_dashboard_allowed_hosts_accepts_explicit_host(monkeypatch):
    monkeypatch.setenv("HERMES_DASHBOARD_ALLOWED_HOSTS", "dashboard.tailnet.example")

    assert _is_accepted_host("dashboard.tailnet.example", "127.0.0.1") is True
    assert _is_accepted_host("dashboard.tailnet.example:443", "127.0.0.1") is True


def test_dashboard_allowed_hosts_keeps_unlisted_host_blocked(monkeypatch):
    monkeypatch.setenv("HERMES_DASHBOARD_ALLOWED_HOSTS", "dashboard.tailnet.example")

    assert _is_accepted_host("evil.example", "127.0.0.1") is False


def test_dashboard_allowed_hosts_does_not_enable_wildcards(monkeypatch):
    monkeypatch.setenv(
        "HERMES_DASHBOARD_ALLOWED_HOSTS",
        "*, *.example.test, .example.org",
    )

    assert _is_accepted_host("evil.example", "127.0.0.1") is False
    assert _is_accepted_host("dashboard.example.test", "127.0.0.1") is False
    assert _is_accepted_host("sub.example.org", "127.0.0.1") is False


def test_dashboard_allowed_hosts_trims_comma_separated_hosts(monkeypatch):
    monkeypatch.setenv(
        "HERMES_DASHBOARD_ALLOWED_HOSTS",
        " dashboard.tailnet.example , alt.example:8443,\tMIXED.Example ",
    )

    assert _is_accepted_host("dashboard.tailnet.example", "127.0.0.1") is True
    assert _is_accepted_host("alt.example:443", "127.0.0.1") is True
    assert _is_accepted_host("mixed.example", "127.0.0.1") is True
    assert _is_accepted_host("missing.example", "127.0.0.1") is False

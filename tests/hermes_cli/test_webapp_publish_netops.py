"""Unit tests for the Desktop setup flow's network helpers (no real network)."""
import subprocess
from types import SimpleNamespace

from hermes_cli.webapp_publish import netops


# ── public_ip ────────────────────────────────────────────────────────────────

def test_public_ip_first_valid_answer_wins():
    calls = []

    def fetch(url):
        calls.append(url)
        return "203.0.113.7\n"

    assert netops.public_ip(fetch=fetch) == "203.0.113.7"
    assert calls == [netops.IP_ECHO_SERVICES[0]]


def test_public_ip_falls_through_invalid_and_failing_services():
    answers = iter(["not-an-ip", OSError("down"), "198.51.100.4"])

    def fetch(url):
        value = next(answers)
        if isinstance(value, Exception):
            raise value
        return value

    assert netops.public_ip(fetch=fetch) == "198.51.100.4"


def test_public_ip_none_when_all_fail():
    def fetch(url):
        raise OSError("no network")

    assert netops.public_ip(fetch=fetch) is None


def test_public_ip_rejects_ipv6_and_garbage():
    assert netops.public_ip(fetch=lambda url: "::1") is None
    assert netops.public_ip(fetch=lambda url: "999.1.1.1") is None


# ── dig parsing / resolve / verify ───────────────────────────────────────────

def _fake_run(stdout="", returncode=0):
    def run(cmd, **kwargs):
        return SimpleNamespace(stdout=stdout, returncode=returncode)
    return run


def test_dig_a_parses_only_ipv4_answers(monkeypatch):
    monkeypatch.setattr(netops.shutil, "which", lambda name: "/usr/bin/dig")
    stdout = "desk.example.com.cdn.cloudflare.net.\n203.0.113.7\n198.51.100.4\n"
    answers = netops._dig_a("desk.example.com", "1.1.1.1", run=_fake_run(stdout))
    assert answers == ["203.0.113.7", "198.51.100.4"]


def test_dig_a_without_dig_returns_empty(monkeypatch):
    monkeypatch.setattr(netops.shutil, "which", lambda name: None)
    assert netops._dig_a("desk.example.com", None) == []


def test_verify_a_record_matches_any_single_resolver(monkeypatch):
    monkeypatch.setattr(netops.shutil, "which", lambda name: "/usr/bin/dig")

    def run(cmd, **kwargs):
        # Only the 8.8.8.8 query has propagated.
        if "@8.8.8.8" in cmd:
            return SimpleNamespace(stdout="203.0.113.7\n", returncode=0)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(netops, "_system_a", lambda host: [])
    ok, seen = netops.verify_a_record("desk.example.com", "203.0.113.7", run=run)
    assert ok is True
    assert seen == {"8.8.8.8": ["203.0.113.7"]}


def test_verify_a_record_mismatch_reports_all_sources(monkeypatch):
    monkeypatch.setattr(netops.shutil, "which", lambda name: "/usr/bin/dig")
    monkeypatch.setattr(netops, "_system_a", lambda host: ["192.0.2.99"])
    ok, seen = netops.verify_a_record(
        "desk.example.com", "203.0.113.7", run=_fake_run("192.0.2.1\n"))
    assert ok is False
    assert seen["1.1.1.1"] == ["192.0.2.1"]
    assert seen["system"] == ["192.0.2.99"]


def test_verify_a_record_invalid_expected_ip():
    ok, seen = netops.verify_a_record("desk.example.com", "not-an-ip")
    assert ok is False and seen == {}


def test_resolve_a_includes_system_resolver(monkeypatch):
    monkeypatch.setattr(netops.shutil, "which", lambda name: None)
    monkeypatch.setattr(netops, "_system_a", lambda host: ["203.0.113.7"])
    assert netops.resolve_a("desk.example.com") == {"system": ["203.0.113.7"]}


# ── hostname validation ──────────────────────────────────────────────────────

def test_valid_hostname_accepts_real_domains():
    assert netops.valid_hostname("desk.example.com")
    assert netops.valid_hostname("a-b.example.co.uk")
    assert netops.valid_hostname("DESK.Example.COM")


def test_valid_hostname_rejects_malformed():
    for bad in ("", "localhost", "desk", "http://desk.example.com", "desk.example.com/",
                "*.example.com", "desk.example.com.", "-desk.example.com",
                "desk..example.com", "x" * 250 + ".com"):
        assert not netops.valid_hostname(bad), bad


def test_is_local_name_single_label_only():
    assert netops.is_local_name("hermes-desktop.local")
    assert netops.is_local_name("desk.local")
    assert not netops.is_local_name("a.b.local")
    assert not netops.is_local_name("desk.example.com")
    assert not netops.is_local_name("hermes-desktop.local.")
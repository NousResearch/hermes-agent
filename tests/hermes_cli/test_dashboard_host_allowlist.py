"""Tests for dashboard hostname allowlist in _is_accepted_host."""
import pytest
from hermes_cli.web_server import _is_accepted_host


def test_loopback_host_always_accepted():
    assert _is_accepted_host("127.0.0.1:10000", "127.0.0.1") is True
    assert _is_accepted_host("localhost:10000", "127.0.0.1") is True


def test_unknown_host_rejected():
    assert _is_accepted_host("evil.example.com:443", "127.0.0.1") is False


def test_0_0_0_0_bind_accepts_any_host():
    assert _is_accepted_host("anything.example.com:443", "0.0.0.0") is True

"""Tests for the configurable desktop PTY keep-alive TTL.

The desktop chat runs in a keep-alive PTY child of `hermes serve`; the
reaper kills the child after it has been detached for the registry TTL.
The TTL was hardcoded at 30 minutes; it is now configurable via
``serve.pty_ttl_minutes`` (default 30, preserving the historical
behavior).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestPtyKeepaliveTtl:

    def test_default_ttl_is_30_minutes(self):
        """No config → the historical 30-minute TTL."""
        from hermes_cli import web_server

        with patch("hermes_cli.web_server.load_config", return_value={}):
            assert web_server._pty_keepalive_ttl_seconds() == 30 * 60

    def test_custom_ttl_minutes(self):
        from hermes_cli import web_server

        cfg = {"serve": {"pty_ttl_minutes": 5}}
        with patch("hermes_cli.web_server.load_config", return_value=cfg):
            assert web_server._pty_keepalive_ttl_seconds() == 5 * 60

    def test_zero_or_negative_falls_back_to_default(self):
        from hermes_cli import web_server

        for bad in (0, -10):
            cfg = {"serve": {"pty_ttl_minutes": bad}}
            with patch("hermes_cli.web_server.load_config", return_value=cfg):
                assert web_server._pty_keepalive_ttl_seconds() == 30 * 60

    def test_non_numeric_falls_back_to_default(self):
        from hermes_cli import web_server

        cfg = {"serve": {"pty_ttl_minutes": "soon"}}
        with patch("hermes_cli.web_server.load_config", return_value=cfg):
            assert web_server._pty_keepalive_ttl_seconds() == 30 * 60

    def test_config_error_falls_back_to_default(self):
        from hermes_cli import web_server

        with patch(
            "hermes_cli.web_server.load_config", side_effect=RuntimeError("boom")
        ):
            assert web_server._pty_keepalive_ttl_seconds() == 30 * 60

    def test_registry_uses_resolved_ttl(self):
        """The PTY_REGISTRY is built with the resolved TTL (default)."""
        from hermes_cli import web_server

        assert web_server.PTY_REGISTRY._ttl == 30 * 60

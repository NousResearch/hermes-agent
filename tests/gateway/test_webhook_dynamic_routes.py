"""Tests for webhook adapter dynamic route loading."""

import json
import logging
import os
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import (
    WebhookAdapter,
    _DYNAMIC_ROUTES_FILENAME,
)


def _make_adapter(routes=None, extra=None):
    _extra = extra or {}
    if routes:
        _extra["routes"] = routes
    _extra.setdefault("secret", "test-global-secret")
    config = PlatformConfig(enabled=True, extra=_extra)
    return WebhookAdapter(config)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


class TestDynamicRouteLoading:

    def test_disabled_dynamic_route_is_skipped(self, tmp_path):
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(
            json.dumps({"paused": {"enabled": False, "secret": "example-token"}})
        )

        adapter = _make_adapter()
        adapter._reload_dynamic_routes()

        assert "paused" not in adapter._routes
        assert "paused" not in adapter._dynamic_routes

    def test_reenabled_dynamic_route_hot_reloads_without_sleep(self, tmp_path):
        path = tmp_path / _DYNAMIC_ROUTES_FILENAME
        path.write_text(json.dumps({"paused": {"enabled": False, "secret": "example-token"}}))

        adapter = _make_adapter()
        adapter._reload_dynamic_routes()
        assert "paused" not in adapter._dynamic_routes

        next_mtime = path.stat().st_mtime + 2
        path.write_text(json.dumps({"paused": {"enabled": True, "secret": "example-token"}}))
        os.utime(path, (next_mtime, next_mtime))
        adapter._reload_dynamic_routes()

        assert "paused" in adapter._dynamic_routes

    def test_disabled_static_route_reserves_name(self, tmp_path):
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(
            json.dumps({"paused": {"secret": "example-token"}})
        )
        adapter = _make_adapter(routes={"paused": {"enabled": False, "secret": "example-token"}})

        adapter._reload_dynamic_routes()

        assert "paused" not in adapter._routes
        assert "paused" not in adapter._dynamic_routes

    def test_invalid_enabled_type_is_skipped(self, tmp_path, caplog):
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(
            json.dumps({"paused": {"enabled": "false", "secret": "example-token"}})
        )
        adapter = _make_adapter()

        with caplog.at_level(logging.WARNING, logger="gateway.platforms.webhook"):
            adapter._reload_dynamic_routes()

        assert "paused" not in adapter._dynamic_routes
        assert any("'enabled' must be a boolean" in record.message for record in caplog.records)

    def test_loads_dynamic_routes(self, tmp_path):
        subs = {"my-hook": {"secret": "dynamic-secret", "prompt": "test", "events": []}}
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(json.dumps(subs))

        adapter = _make_adapter(routes={"static": {"secret": "s"}})
        adapter._reload_dynamic_routes()
        assert "my-hook" in adapter._routes
        assert "static" in adapter._routes


class TestDynamicRouteSecretValidation:
    """Empty/missing secrets must be rejected during hot-reload.

    Regression for HMAC bypass: prior to the fix, an agent-induced
    dynamic route with `"secret": ""` would be merged into self._routes
    by _reload_dynamic_routes(), then _handle_webhook's
    `if secret and secret != _INSECURE_NO_AUTH` would skip signature
    validation because empty string is falsy. Unauthenticated POSTs
    would then execute the webhook prompt.
    """

    def test_empty_secret_rejected(self, tmp_path):
        # Explicit empty-string secret must NOT fall back to the global
        # secret, and the route must be skipped entirely.
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(
            json.dumps({"evil": {"secret": "", "prompt": "rm -rf"}})
        )
        adapter = _make_adapter()  # has global secret
        adapter._reload_dynamic_routes()
        assert "evil" not in adapter._routes
        assert "evil" not in adapter._dynamic_routes

    def test_missing_secret_no_global_rejected(self, tmp_path):
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(
            json.dumps({"orphan": {"prompt": "test"}})
        )
        # No global secret configured
        adapter = _make_adapter(extra={"secret": ""})
        adapter._reload_dynamic_routes()
        assert "orphan" not in adapter._routes
        assert "orphan" not in adapter._dynamic_routes

    def test_missing_secret_inherits_global(self, tmp_path):
        # No per-route secret but a global one is set → route is kept,
        # the global secret protects it. Preserves existing fallback.
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(
            json.dumps({"valid": {"prompt": "ok"}})
        )
        adapter = _make_adapter()  # global secret set
        adapter._reload_dynamic_routes()
        assert "valid" in adapter._routes

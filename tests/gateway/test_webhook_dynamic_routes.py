"""Tests for webhook adapter dynamic route loading."""

import json
import os
import pytest
from pathlib import Path

from gateway.config import PlatformConfig
from gateway.platforms.webhook import (
    WebhookAdapter,
    _DYNAMIC_ROUTES_FILENAME,
    _INSECURE_NO_AUTH,
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
    def test_no_dynamic_file(self):
        adapter = _make_adapter(routes={"static": {"secret": "s"}})
        adapter._reload_dynamic_routes()
        assert "static" in adapter._routes
        assert len(adapter._dynamic_routes) == 0

    def test_loads_dynamic_routes(self, tmp_path):
        subs = {"my-hook": {"secret": "dynamic-secret", "prompt": "test", "events": []}}
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(json.dumps(subs))

        adapter = _make_adapter(routes={"static": {"secret": "s"}})
        adapter._reload_dynamic_routes()
        assert "my-hook" in adapter._routes
        assert "static" in adapter._routes

    def test_static_takes_precedence(self, tmp_path):
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(
            json.dumps({"conflict": {"secret": "dynamic", "prompt": "dyn"}})
        )
        adapter = _make_adapter(routes={"conflict": {"secret": "static", "prompt": "stat"}})
        adapter._reload_dynamic_routes()
        assert adapter._routes["conflict"]["secret"] == "static"

    def test_mtime_gated(self, tmp_path):
        import time
        path = tmp_path / _DYNAMIC_ROUTES_FILENAME
        path.write_text(json.dumps({"v1": {"secret": "s"}}))

        adapter = _make_adapter()
        adapter._reload_dynamic_routes()
        assert "v1" in adapter._dynamic_routes

        # Same mtime — no reload
        adapter._dynamic_routes["injected"] = True
        adapter._reload_dynamic_routes()
        assert "injected" in adapter._dynamic_routes

        # New write — reloads
        time.sleep(0.05)
        path.write_text(json.dumps({"v2": {"secret": "s"}}))
        adapter._reload_dynamic_routes()
        assert "v2" in adapter._dynamic_routes
        assert "v1" not in adapter._dynamic_routes

    def test_skips_dynamic_route_without_resolved_secret(self, tmp_path):
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(
            json.dumps({"hot-nosig": {"prompt": "test"}})
        )

        adapter = _make_adapter(extra={"secret": ""})
        adapter._reload_dynamic_routes()

        assert "hot-nosig" not in adapter._routes
        assert adapter._dynamic_routes == {}

    def test_skips_dynamic_route_with_empty_secret(self, tmp_path):
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(
            json.dumps({"hot-empty": {"secret": "", "prompt": "test"}})
        )

        adapter = _make_adapter(extra={"secret": ""})
        adapter._reload_dynamic_routes()

        assert "hot-empty" not in adapter._routes
        assert adapter._dynamic_routes == {}

    def test_skips_dynamic_route_with_insecure_no_auth(self, tmp_path):
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(
            json.dumps(
                {"hot-insecure": {"secret": _INSECURE_NO_AUTH, "prompt": "test"}}
            )
        )

        adapter = _make_adapter(extra={"host": "127.0.0.1"})
        adapter._reload_dynamic_routes()

        assert "hot-insecure" not in adapter._routes
        assert adapter._dynamic_routes == {}

    def test_skips_dynamic_route_that_is_not_mapping(self, tmp_path):
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(
            json.dumps({"hot-bad": "not-a-route"})
        )

        adapter = _make_adapter()
        adapter._reload_dynamic_routes()

        assert "hot-bad" not in adapter._routes
        assert adapter._dynamic_routes == {}

    def test_file_removal_clears(self, tmp_path):
        path = tmp_path / _DYNAMIC_ROUTES_FILENAME
        path.write_text(json.dumps({"temp": {"secret": "s"}}))
        adapter = _make_adapter()
        adapter._reload_dynamic_routes()
        assert "temp" in adapter._dynamic_routes

        path.unlink()
        adapter._reload_dynamic_routes()
        assert len(adapter._dynamic_routes) == 0

    def test_corrupted_file(self, tmp_path):
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text("not json")
        adapter = _make_adapter(routes={"static": {"secret": "s"}})
        adapter._reload_dynamic_routes()
        assert "static" in adapter._routes
        assert len(adapter._dynamic_routes) == 0

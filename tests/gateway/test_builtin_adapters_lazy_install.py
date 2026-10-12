"""Regression tests for the builtin adapters' lazy aiohttp self-heal (#136990).

The ``webhook`` and ``api_server`` gateway adapters bind ``AIOHTTP_AVAILABLE = False`` at import
time when aiohttp is missing, and ``_instantiate_builtin_adapter`` used to skip them permanently.
A gateway that boots onto a PM generation without the ``messaging`` extra (Docker image with
``extras: []``) therefore lost both listeners until someone repaired the environment by hand,
while the plugin adapters self-healed through ``pm.ensure_import``.

``ensure_webhook_requirements`` / ``ensure_api_server_requirements`` mirror the plugin adapters'
``check_telegram_requirements`` (#79812's active/passive split): at gateway start they get one
``pm.ensure_import("messaging")`` attempt and rebind the module-level symbols. The tests drive the
real transition — fake ``pm``/``aiohttp`` modules plus a flipped availability flag — so a symbol
the rebind misses fails here instead of in production (#85272 lesson).
"""

import sys
import types

import gateway.platforms.api_server as api_server_mod
import gateway.platforms.webhook as webhook_mod
from gateway.builtin_adapters import _instantiate_builtin_adapter
from gateway.config import Platform, PlatformConfig


def _fake_aiohttp(monkeypatch):
    """Seed sys.modules with a fake ``aiohttp`` whose ``web`` exposes the symbols the rebind and
    the middleware builder dereference at definition time."""
    fake_web = types.SimpleNamespace(
        Application=object, Response=object, Request=object, AppRunner=object,
        TCPSite=object, json_response=lambda *a, **k: None,
        HTTPRequestEntityTooLarge=type("HTTPRequestEntityTooLarge", (Exception,), {}),
        middleware=lambda fn: fn,
    )
    fake_aiohttp = types.ModuleType("aiohttp")
    fake_aiohttp.web = fake_web
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    return fake_web


def _fake_pm(monkeypatch, calls, error=None):
    fake_pm = types.ModuleType("pm")

    def _ensure_import(extra):
        calls.append(extra)
        if error is not None:
            raise error

    fake_pm.ensure_import = _ensure_import
    monkeypatch.setitem(sys.modules, "pm", fake_pm)


class TestWebhookEnsure:
    def test_installs_messaging_extra_and_rebinds_web(self, monkeypatch):
        monkeypatch.setattr(webhook_mod, "AIOHTTP_AVAILABLE", False)
        monkeypatch.setattr(webhook_mod, "web", None)
        calls = []
        _fake_pm(monkeypatch, calls)
        fake_web = _fake_aiohttp(monkeypatch)

        assert webhook_mod.ensure_webhook_requirements() is True
        assert calls == ["messaging"]
        assert webhook_mod.AIOHTTP_AVAILABLE is True
        assert webhook_mod.web is fake_web

    def test_install_failure_keeps_adapter_disabled(self, monkeypatch):
        monkeypatch.setattr(webhook_mod, "AIOHTTP_AVAILABLE", False)
        monkeypatch.setattr(webhook_mod, "web", None)
        calls = []
        _fake_pm(monkeypatch, calls, error=RuntimeError("venv", "no network"))
        _fake_aiohttp(monkeypatch)

        assert webhook_mod.ensure_webhook_requirements() is False
        assert calls == ["messaging"]
        assert webhook_mod.AIOHTTP_AVAILABLE is False
        assert webhook_mod.web is None

    def test_noop_when_already_available(self, monkeypatch):
        calls = []
        _fake_pm(monkeypatch, calls)

        assert webhook_mod.ensure_webhook_requirements() is True
        assert calls == []  # passive fast path: never installs when aiohttp is already bound


class TestApiServerEnsure:
    def test_rebinds_web_and_rebuilds_middlewares(self, monkeypatch):
        # A module imported bare: flag off, placeholders still None after the import-time build.
        monkeypatch.setattr(api_server_mod, "AIOHTTP_AVAILABLE", False)
        monkeypatch.setattr(api_server_mod, "web", None)
        monkeypatch.setattr(api_server_mod, "cors_middleware", None)
        monkeypatch.setattr(api_server_mod, "body_limit_middleware", None)
        monkeypatch.setattr(api_server_mod, "security_headers_middleware", None)
        calls = []
        _fake_pm(monkeypatch, calls)
        fake_web = _fake_aiohttp(monkeypatch)

        assert api_server_mod.ensure_api_server_requirements() is True
        assert calls == ["messaging"]
        assert api_server_mod.AIOHTTP_AVAILABLE is True
        assert api_server_mod.web is fake_web
        # The middleware stack must come back too — start() filters None out, so a lazy boot
        # that heals would otherwise silently run without CORS/security headers.
        assert callable(api_server_mod.cors_middleware)
        assert callable(api_server_mod.body_limit_middleware)
        assert callable(api_server_mod.security_headers_middleware)


class TestInstantiateSelfHeal:
    def test_probe_failure_triggers_installer_before_skipping(self, monkeypatch):
        monkeypatch.setattr(webhook_mod, "AIOHTTP_AVAILABLE", False)
        monkeypatch.setattr(webhook_mod, "web", None)
        calls = []
        _fake_pm(monkeypatch, calls)
        _fake_aiohttp(monkeypatch)

        adapter = _instantiate_builtin_adapter(Platform.WEBHOOK, PlatformConfig())

        assert calls == ["messaging"]
        assert adapter is not None
        assert webhook_mod.AIOHTTP_AVAILABLE is True

    def test_unhealable_adapter_still_skips_with_warning(self, monkeypatch, caplog):
        monkeypatch.setattr(webhook_mod, "AIOHTTP_AVAILABLE", False)
        monkeypatch.setattr(webhook_mod, "web", None)
        calls = []
        # ensure_import "succeeds" but the restart-needed path is exactly this shape: the raise
        # happens after the sync, so the adapter must stay disabled rather than half-boot.
        _fake_pm(monkeypatch, calls, error=RuntimeError("venv", "aiohttp installed; restart Hermes"))
        _fake_aiohttp(monkeypatch)

        adapter = _instantiate_builtin_adapter(Platform.WEBHOOK, PlatformConfig())

        assert adapter is None
        assert webhook_mod.AIOHTTP_AVAILABLE is False

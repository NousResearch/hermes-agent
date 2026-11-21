"""The Hermes Desktop (browser) row in the gateway setup wizards.

Wizard-only row: listed by both `hermes setup` (gateway checklist) and
`hermes gateway setup` (platform loop) via _all_platforms, dispatched to the
shared publishing flow, status read from webapp.* — and never a registry
entry, so it cannot leak into delivery targets or the Channels catalog.
"""
from types import SimpleNamespace

import pytest

from hermes_cli.gateway_setup_wizard import (
    _all_platforms,
    _configure_platform,
    _platform_status,
)


def _desktop_row():
    return next(p for p in _all_platforms() if p["key"] == "desktop")


def test_desktop_listed_in_platform_menu():
    row = _desktop_row()
    assert row["label"] == "Hermes Desktop (browser)"
    assert row["emoji"] == "🖥️"
    assert row.get("_registry_entry") is None  # wizard-only, not a channel


def test_status_follows_webapp_enabled(monkeypatch):
    monkeypatch.setattr("hermes_cli.webapp_access.gateway_home", lambda: "/tmp/fake-home")
    monkeypatch.setattr("gateway.webapp_listener.webapp_settings",
                        lambda home: SimpleNamespace(enabled=True))
    assert _platform_status(_desktop_row()) == "configured"
    monkeypatch.setattr("gateway.webapp_listener.webapp_settings",
                        lambda home: SimpleNamespace(enabled=False))
    assert _platform_status(_desktop_row()) == "not configured"


def test_status_probe_failure_is_not_configured(monkeypatch):
    def boom(home):
        raise OSError("unreadable")
    monkeypatch.setattr("gateway.webapp_listener.webapp_settings", boom)
    assert _platform_status(_desktop_row()) == "not configured"


def test_configure_dispatches_to_publishing_flow(monkeypatch):
    called = []
    monkeypatch.setattr("hermes_cli.webapp_publish.flow.interactive_setup",
                        lambda: called.append(True))
    _configure_platform(_desktop_row())
    assert called == [True]


def test_desktop_never_a_delivery_target_or_channel_card():
    from gateway.platform_registry import platform_registry
    from hermes_cli.platforms import get_all_platforms
    from hermes_cli.web_server_messaging import _messaging_platform_catalog
    _all_platforms()  # ensure discovery ran
    assert platform_registry.get("desktop") is None
    assert "desktop" not in get_all_platforms()
    assert "desktop" not in [e["id"] for e in _messaging_platform_catalog()]


def test_configured_desktop_counts_as_setup_progress():
    from hermes_cli.setup_platforms import _is_progress
    assert _is_progress("configured") is True
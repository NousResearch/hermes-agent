"""Tests for the CLI strangler boundary."""

from __future__ import annotations

import types

import nous_cli.legacy
import nous_cli.main


def test_main_routes_through_legacy_seam(monkeypatch):
    calls = []

    monkeypatch.setattr(
        nous_cli.main,
        "dispatch_legacy",
        lambda: calls.append("legacy") or 17,
    )

    assert nous_cli.main.main() == 17
    assert calls == ["legacy"]


def test_legacy_seam_imports_legacy_cli_lazily(monkeypatch):
    calls = []
    fake_module = types.SimpleNamespace(main=lambda: calls.append("main") or 23)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "hermes_cli.main":
            return fake_module
        return real_import(name, globals, locals, fromlist, level)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    assert nous_cli.legacy.dispatch_legacy() == 23
    assert calls == ["main"]

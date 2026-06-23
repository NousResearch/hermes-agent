"""Regression tests for #51099 — Honcho missing-dependency availability.

When ``memory.provider`` is set to ``honcho`` but the ``honcho-ai`` package
is not installed in the runtime environment, Hermes used to log the provider
as registered/activated and then fail every background session init, because
``is_available()`` only inspected config and never checked that the package
was importable (the SDK is imported lazily at session-init time).

Activation is gated on ``is_available()`` (see ``agent/agent_init.py``), so a
config-only check let an unusable provider activate. These tests pin the
corrected behavior: a missing package means not available.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

from plugins.memory.honcho import HonchoMemoryProvider


def _configured_cfg() -> SimpleNamespace:
    """A config that, on its own, would make Honcho look enabled."""
    return SimpleNamespace(enabled=True, api_key=None, base_url="http://127.0.0.1:8000")


def test_is_available_false_when_honcho_package_missing(monkeypatch):
    """Configured but ``honcho`` package absent -> not available."""
    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: _configured_cfg(),
    )
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "honcho" else real_find_spec(name, *a, **k),
    )
    assert HonchoMemoryProvider().is_available() is False


def test_is_available_true_when_configured_and_package_present(monkeypatch):
    """Configured and ``honcho`` package importable -> available."""
    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        lambda: _configured_cfg(),
    )
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: SimpleNamespace() if name == "honcho" else real_find_spec(name, *a, **k),
    )
    assert HonchoMemoryProvider().is_available() is True

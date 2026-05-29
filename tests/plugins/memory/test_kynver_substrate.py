"""Kynver substrate detection — default-on when configured and healthy."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.kynver.substrate import (
    kynver_explicitly_disabled,
    resolve_memory_provider_name,
    substrate_active,
)


def test_explicit_disable_env():
    assert kynver_explicitly_disabled(env={"KYNVER_DISABLED": "1"}) is True
    assert kynver_explicitly_disabled(env={}) is False


def test_resolve_memory_provider_defaults_to_kynver_when_healthy():
    with (
        patch("plugins.memory.kynver.substrate.agentos_available", return_value=True),
        patch("plugins.memory.kynver.substrate.probe_agentos_health", return_value=True),
    ):
        assert resolve_memory_provider_name({}) == "kynver"


def test_resolve_memory_provider_respects_explicit_non_kynver():
    assert resolve_memory_provider_name({"provider": "honcho"}) == "honcho"


def test_substrate_inactive_when_health_probe_fails():
    client = MagicMock()
    with (
        patch("plugins.memory.kynver.substrate.agentos_available", return_value=True),
        patch("plugins.memory.kynver.substrate.probe_agentos_health", return_value=False),
    ):
        assert substrate_active(client=client) is False

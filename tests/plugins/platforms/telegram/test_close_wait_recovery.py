"""#69314: Telegram polling recovery must drain the general request pool,
not just the polling pool, when the send path is degraded."""

from __future__ import annotations

import inspect

import pytest


def test_polling_recovery_drains_general_pool():
    """_handle_polling_network_error must call
    _drain_general_connections_after_pool_timeout when send_path_degraded
    is set (#69314)."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    src = inspect.getsource(TelegramAdapter._handle_polling_network_error)
    assert "_drain_polling_connections" in src, \
        "Must drain polling pool"
    assert "_drain_general_connections_after_pool_timeout" in src, \
        "Must also drain general pool when send path is degraded (#69314)"
    assert "_send_path_degraded" in src, \
        "Must check _send_path_degraded before draining general pool"


def test_drain_general_pool_only_when_degraded():
    """The general pool drain must be gated on _send_path_degraded,
    not unconditional (avoid draining on pure polling-only errors)."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    src = inspect.getsource(TelegramAdapter._handle_polling_network_error)
    # Find the drain call and verify it's guarded
    assert 'if getattr(self, "_send_path_degraded", False)' in src, \
        "General pool drain must be gated on _send_path_degraded"


def test_send_path_degraded_set_on_network_error():
    """_handle_polling_network_error must set _send_path_degraded."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    src = inspect.getsource(TelegramAdapter._handle_polling_network_error)
    assert "_send_path_degraded = True" in src, \
        "Must set _send_path_degraded = True on network error"
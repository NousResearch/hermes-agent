"""Phase 4.5 Step 2 hook-timeout/config ownership gates."""

from __future__ import annotations

import logging

import pytest

import plugin_runtime.dispatch as dispatch


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, dispatch._HOOK_CALLBACK_TIMEOUT_SECS),
        (0, 0.0),
        (-1, dispatch._HOOK_CALLBACK_TIMEOUT_SECS),
        (dispatch._MAX_HOOK_CALLBACK_TIMEOUT_SECS + 1, dispatch._MAX_HOOK_CALLBACK_TIMEOUT_SECS),
        ("not-a-number", dispatch._HOOK_CALLBACK_TIMEOUT_SECS),
    ],
)
def test_hook_timeout_resolution_preserves_bounds(monkeypatch, raw, expected):
    monkeypatch.setattr(dispatch, "read_hook_callback_timeout_seconds", lambda: raw)

    assert dispatch._resolve_hook_callback_timeout() == expected


def test_hook_timeout_resolution_reads_bridge_once(monkeypatch):
    calls = []

    def read_timeout():
        calls.append(1)
        return 0.25

    monkeypatch.setattr(dispatch, "read_hook_callback_timeout_seconds", read_timeout)

    assert dispatch._resolve_hook_callback_timeout() == 0.25
    assert calls == [1]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("bad", "plugins.hook_callback_timeout is not a number; using default 30s"),
        (-1, "plugins.hook_callback_timeout=-1 is negative; using default 30s"),
        (601, "plugins.hook_callback_timeout=601 exceeds max 600s; clamping"),
    ],
)
def test_hook_timeout_warning_wording_is_preserved(monkeypatch, caplog, raw, message):
    monkeypatch.setattr(dispatch, "read_hook_callback_timeout_seconds", lambda: raw)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        dispatch._resolve_hook_callback_timeout()

    assert message in caplog.text


def test_cli_facade_reexports_runtime_timeout_resolver():
    import hermes_cli.plugins as plugins

    assert plugins._resolve_hook_callback_timeout is dispatch._resolve_hook_callback_timeout

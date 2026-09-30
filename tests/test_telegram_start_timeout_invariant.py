"""Regression guard: the start_polling wall bound must exceed its HTTP budget.

Incident (hub, 2026-08-28). `_UPDATER_START_TIMEOUT` was a hardcoded 30.0 while
the getUpdates request budget is `connect_timeout` (10.0) + `read_timeout`
(20.0) = exactly 30.0. Every polling reconnect raced its own HTTP layer and the
outer wall deadline always won: 151 consecutive failed start attempts, every
single one pinned at 30.0s, while a hand-run getUpdates long-poll against the
same token from the same host returned HTTP 200 in 30.2s. Telegram polling
looked permanently dead on a healthy network and inbound messages stopped.

The bug is a TIE, not a too-small number: the outer bound must strictly exceed
the inner budget, or the inner layer's real error is masked by an
indistinguishable outer TimeoutError that the reconnect ladder cannot classify.

These tests assert the invariant across default AND tuned configurations, so
raising `HERMES_TELEGRAM_HTTP_READ_TIMEOUT` can never silently re-create it.
"""

import importlib.util
import os
from pathlib import Path

import pytest

ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "plugins/platforms/telegram/adapter.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "_tg_adapter_timeout_probe", ADAPTER
    )
    assert spec is not None and spec.loader is not None, ADAPTER
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _inner_budget(connect: float, read: float) -> float:
    return connect + read


def test_default_bound_strictly_exceeds_default_http_budget():
    """The exact production tie that broke polling on 2026-08-28."""
    mod = _load()
    inner = _inner_budget(10.0, 20.0)
    assert mod._UPDATER_START_TIMEOUT > inner, (
        f"start bound {mod._UPDATER_START_TIMEOUT} does not exceed the inner "
        f"HTTP budget {inner}; a tie means the wall deadline always fires "
        "first and reconnect can never succeed"
    )


@pytest.mark.parametrize(
    "connect,read",
    [
        (10.0, 20.0),   # defaults — the incident configuration
        (10.0, 30.0),   # someone raises read_timeout
        (20.0, 40.0),   # both raised
        (5.0, 10.0),    # both lowered
        (1.0, 2.0),     # aggressive tuning
    ],
)
def test_bound_exceeds_budget_under_tuning(monkeypatch, connect, read):
    """Tuning either knob must never re-create the tie."""
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", str(connect))
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", str(read))
    mod = _load()
    bound = mod._updater_start_timeout()
    inner = _inner_budget(connect, read)
    assert bound > inner, (
        f"connect={connect} read={read}: bound {bound} <= inner budget {inner}"
    )


def test_bound_is_derived_not_hardcoded(monkeypatch):
    """A large read_timeout must move the bound, proving it is derived."""
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", "10.0")
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", "120.0")
    mod = _load()
    bound = mod._updater_start_timeout()
    assert bound > 130.0, (
        f"bound {bound} did not track a 120s read_timeout — it is still "
        "effectively hardcoded, so the tie can return"
    )


def test_bound_never_drops_below_the_historical_floor(monkeypatch):
    """Lowering the knobs must not make the bound uselessly short."""
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", "0.1")
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", "0.1")
    mod = _load()
    assert mod._updater_start_timeout() >= 30.0


def test_garbage_env_falls_back_to_defaults(monkeypatch):
    """A malformed override must not crash import or collapse the bound."""
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", "not-a-number")
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", "")
    mod = _load()
    bound = mod._updater_start_timeout()
    assert bound > _inner_budget(10.0, 20.0)

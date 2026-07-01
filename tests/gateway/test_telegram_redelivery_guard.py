"""Tier-1 unit tests for the Telegram boot-redelivery guard (scope B).

SPEC: ~/.hermes/plans/2026-07-01_telegram-redelivery-guard-SPEC.md
Covers the pure decision logic + HWM tracker (mock/tmp only). The LIVE-tier
tests (AC-8 companion rows, AC-9 edited_message, AC-11 side-effects, AC-13
checkpoint-fires) live in test_telegram_redelivery_live.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.telegram_redelivery import (
    TelegramHwmTracker,
    decide_redelivery,
    in_redelivery_scope,
    read_hwm,
)


# ── in_redelivery_scope: HWM-primary, time-cap fallback (INV-3, AC-4/AC-5) ──


def test_scope_hwm_primary_at_or_below_is_candidate():
    assert in_redelivery_scope(50, 100, seconds_since_boot=999_999) is True
    assert in_redelivery_scope(100, 100, seconds_since_boot=999_999) is True


def test_scope_hwm_primary_above_is_new_even_long_after_boot():
    # AC-4: update_id > HWM is new, even 30s+ post-boot; time cap does NOT apply.
    assert in_redelivery_scope(101, 100, seconds_since_boot=30) is False


def test_scope_no_hwm_uses_time_cap_only():
    # AC-5: no HWM -> only the 120s window gates.
    assert in_redelivery_scope(5, None, seconds_since_boot=30, no_hwm_window_secs=120) is True
    assert in_redelivery_scope(5, None, seconds_since_boot=200, no_hwm_window_secs=120) is False


def test_scope_none_update_id_never_in_scope():
    assert in_redelivery_scope(None, 100, seconds_since_boot=1) is False


# ── decide_redelivery: the §2 decision table (INV-1 fail-open) ──────────────


def _answerable(answered, present):
    return lambda _sid, _mid: (answered, present)


def test_decide_suppress_only_on_answered_present():
    # AC-1: in scope + positively present -> SUPPRESS (the one suppress case).
    assert decide_redelivery(
        in_scope=True, session_id="s", message_id="42", is_edited=False,
        answerable_fn=_answerable(True, True),
    ) is True


def test_decide_absent_processes():
    # AC-2: positively absent -> PROCESS.
    assert decide_redelivery(
        in_scope=True, session_id="s", message_id="42", is_edited=False,
        answerable_fn=_answerable(True, False),
    ) is False


def test_decide_unanswerable_processes_fail_open():
    # AC-3 (the critical one): authority can't answer -> PROCESS, never suppress.
    assert decide_redelivery(
        in_scope=True, session_id="s", message_id="42", is_edited=False,
        answerable_fn=_answerable(False, False),
    ) is False


def test_decide_lookup_raises_processes_fail_open():
    # AC-7: a raising authority -> PROCESS (fail open), not a crash/suppress.
    def _boom(_sid, _mid):
        raise RuntimeError("db down")
    assert decide_redelivery(
        in_scope=True, session_id="s", message_id="42", is_edited=False,
        answerable_fn=_boom,
    ) is False


def test_decide_out_of_scope_processes_without_query():
    # AC-4: out of scope -> PROCESS, and the answerable_fn is never called.
    called = {"n": 0}
    def _tracking(_sid, _mid):
        called["n"] += 1
        return (True, True)
    assert decide_redelivery(
        in_scope=False, session_id="s", message_id="42", is_edited=False,
        answerable_fn=_tracking,
    ) is False
    assert called["n"] == 0, "out-of-scope must short-circuit before any transcript query"


def test_decide_edited_message_processes_even_if_present():
    # AC-9: an edit carries the same message_id but must PROCESS, not suppress.
    assert decide_redelivery(
        in_scope=True, session_id="s", message_id="42", is_edited=True,
        answerable_fn=_answerable(True, True),
    ) is False


def test_decide_no_session_processes():
    assert decide_redelivery(
        in_scope=True, session_id=None, message_id="42", is_edited=False,
        answerable_fn=_answerable(True, True),
    ) is False


# ── TelegramHwmTracker: coalesced flush + fail-open read ────────────────────


def test_hwm_advances_in_memory_only_until_checkpoint(tmp_path):
    clock = {"t": 1000.0}
    tr = TelegramHwmTracker(tmp_path, "default", checkpoint_interval_secs=30,
                            clock=lambda: clock["t"])
    tr.observe_dispatch(5)
    tr.observe_dispatch(9)
    tr.observe_dispatch(3)  # lower — ignored
    assert tr.value == 9
    # within throttle window: no write yet
    assert tr.maybe_checkpoint() is False
    assert read_hwm(tmp_path, "default") is None  # nothing on disk


def test_hwm_checkpoint_fires_after_interval(tmp_path):
    clock = {"t": 1000.0}
    tr = TelegramHwmTracker(tmp_path, "default", checkpoint_interval_secs=30,
                            clock=lambda: clock["t"])
    tr.observe_dispatch(9)
    clock["t"] = 1031.0  # >30s later
    assert tr.maybe_checkpoint() is True
    assert read_hwm(tmp_path, "default") == 9


def test_hwm_flush_is_unconditional(tmp_path):
    tr = TelegramHwmTracker(tmp_path, "default", checkpoint_interval_secs=30,
                            clock=lambda: 1000.0)
    tr.observe_dispatch(7)
    assert tr.flush() is True  # shutdown flush ignores the throttle
    assert read_hwm(tmp_path, "default") == 7


def test_hwm_file_is_0600(tmp_path):
    tr = TelegramHwmTracker(tmp_path, "default", clock=lambda: 1000.0)
    tr.observe_dispatch(7)
    tr.flush()
    path = tmp_path / "state" / "telegram-last-dispatched-update-id.default.json"
    assert (path.stat().st_mode & 0o777) == 0o600


def test_read_hwm_corrupt_fails_open(tmp_path):
    # AC-5 / RC-3: a corrupt file reads as None (no crash, no suppress).
    d = tmp_path / "state"
    d.mkdir()
    (d / "telegram-last-dispatched-update-id.default.json").write_text("{not json")
    assert read_hwm(tmp_path, "default") is None


def test_read_hwm_partial_wrong_type_fails_open(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    (d / "telegram-last-dispatched-update-id.default.json").write_text(
        json.dumps({"update_id": "not-an-int"})
    )
    assert read_hwm(tmp_path, "default") is None
    # bool must not be accepted (bool is an int subclass)
    (d / "telegram-last-dispatched-update-id.default.json").write_text(
        json.dumps({"update_id": True})
    )
    assert read_hwm(tmp_path, "default") is None


def test_read_hwm_absent_fails_open(tmp_path):
    assert read_hwm(tmp_path, "default") is None

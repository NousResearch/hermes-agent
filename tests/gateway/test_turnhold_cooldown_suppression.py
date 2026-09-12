"""Regression tests: the turn-hold ``turnhold_deferred`` user notice is
suppressed while a compression-failure cooldown is already active for the
session (otherwise sustained traffic after a compression failure spams the
same notice every turn-hold expiry).

The suppression gate is fail-open by design: when the cooldown check cannot
be performed (no session db / no getter / raised), the notice is still sent —
suppression is an optimisation that must never hide a user-facing notice.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _install_fakes(monkeypatch, tmp_path):
    """The gateway modules import dotenv + run_agent at import time; fake
    both so the tests run in a hermes-cli environment without the runtime."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = object
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_runner(monkeypatch, tmp_path, *, session_db):
    """Build a bare GatewayRunner (no __init__) with the supplied session db."""
    from gateway.run import GatewayRunner
    _install_fakes(monkeypatch, tmp_path)
    runner = object.__new__(GatewayRunner)
    runner._session_db = session_db
    return runner


class _AttemptStub:
    """Bare-minimum stand-in for the hygiene turn-hold attempt object that
    run_turn._hmwa_hygiene_on_turn_hold expects."""
    def __init__(self):
        self.meta = {"attempt_id": "a1"}
        self.agent = SimpleNamespace(
            session_id="s1",
            _last_compaction_in_place=False,
        )
        self.commit_fence = SimpleNamespace(
            commit_watermark_fenced=False,
            is_cancelled=False,
        )
        self.wait_started = 0.0
        self.future = SimpleNamespace(
            exception=lambda: None,
        )


def _make_session_entry(sid: str = "s1"):
    """The function under test only reads .session_id and .session_key."""
    return SimpleNamespace(session_id=sid, session_key="k1")


def _noop(*_a, **_kw):
    """Sync stub for _hmwa_hygiene_stamp (called as a sync fn, not awaited)."""
    return None


async def _async_noop(*_a, **_kw):
    return None


def _make_notify(sent):
    """Build an async _hmwa_hygiene_notify mock that records into ``sent``.

    The real method signature is ``_hmwa_hygiene_notify(self, source, meta,
    message, what)`` — 4 positional args after self."""
    async def _notify(source, meta, message, what, **_kw):
        sent.append((message, what))
    return _notify


# ── Unit tests for _cooldown_is_active ──────────────────────────────

def test_cooldown_is_active_true_when_remaining(monkeypatch, tmp_path):
    """Positive remaining_seconds => True."""
    from gateway import run as gateway_run
    fake_db = MagicMock()
    fake_db.get_compression_failure_cooldown = lambda _sid: {"remaining_seconds": 42.0}
    runner = _make_runner(monkeypatch, tmp_path, session_db=SimpleNamespace(_db=fake_db))
    assert gateway_run._cooldown_is_active(runner, "s1") is True


def test_cooldown_is_active_false_when_expired(monkeypatch, tmp_path):
    """Zero/negative/empty/None payload => False."""
    from gateway import run as gateway_run
    for payload in (
        {"remaining_seconds": 0},
        {"remaining_seconds": -1.0},
        {},
        None,
    ):
        fake_db = MagicMock()
        fake_db.get_compression_failure_cooldown = lambda _sid, p=payload: p
        runner = _make_runner(monkeypatch, tmp_path, session_db=SimpleNamespace(_db=fake_db))
        assert gateway_run._cooldown_is_active(runner, "s1") is False, payload


def test_cooldown_is_active_fail_open_when_no_db(monkeypatch, tmp_path):
    """No session db at all => None (fail-open)."""
    from gateway import run as gateway_run
    runner = _make_runner(monkeypatch, tmp_path, session_db=None)
    assert gateway_run._cooldown_is_active(runner, "s1") is None


def test_cooldown_is_active_fail_open_when_getter_missing(monkeypatch, tmp_path):
    """Session db exists but the getter is missing => None (fail-open)."""
    from gateway import run as gateway_run
    fake_db = MagicMock(spec=[])  # no attributes
    runner = _make_runner(monkeypatch, tmp_path, session_db=SimpleNamespace(_db=fake_db))
    assert gateway_run._cooldown_is_active(runner, "s1") is None


def test_cooldown_is_active_fail_open_when_getter_raises(monkeypatch, tmp_path):
    """Getter raises => None (fail-open)."""
    from gateway import run as gateway_run
    fake_db = MagicMock()
    def _raise(_sid):
        raise RuntimeError("db locked")
    fake_db.get_compression_failure_cooldown = _raise
    runner = _make_runner(monkeypatch, tmp_path, session_db=SimpleNamespace(_db=fake_db))
    assert gateway_run._cooldown_is_active(runner, "s1") is None


# ── End-to-end: turnhold_deferred notice is gated by the cooldown ──

def test_turnhold_notice_suppressed_while_cooldown_active(monkeypatch, tmp_path):
    """An active cooldown suppresses the user-facing turnhold_deferred notice;
    the queried session id is the one passed to the db getter."""
    from gateway import run as gateway_run

    captured_sid = {}
    def _getter(sid):
        captured_sid["sid"] = sid
        return {"remaining_seconds": 30.0}

    _install_fakes(monkeypatch, tmp_path)
    fake_db = MagicMock(get_compression_failure_cooldown=_getter)
    runner = _make_runner(monkeypatch, tmp_path, session_db=SimpleNamespace(_db=fake_db))

    session_entry = _make_session_entry(sid="session-suppress-me")
    attempt = _AttemptStub()
    hs = SimpleNamespace(
        failure_cooldown_seconds=300.0,
        max_turn_hold_seconds=10.0,
    )

    sent: list = []
    runner._hmwa_hygiene_notify = _make_notify(sent)
    runner._hmwa_hygiene_stamp = _noop
    runner._hmwa_hygiene_defer_cleanup = _noop
    runner._hmwa_hygiene_cancel_or_adopt = _async_noop

    import asyncio
    try:
        asyncio.run(runner._hmwa_hygiene_on_turn_hold(
            attempt, hs, session_entry, session_entry.session_key, SimpleNamespace(),
        ))
    except Exception:
        pass
    assert captured_sid.get("sid") == "session-suppress-me"
    assert sent == [], f"expected zero notices, got {sent!r}"


def test_turnhold_notice_sent_when_cooldown_expired(monkeypatch, tmp_path):
    """No/expired cooldown => the turnhold_deferred notice is still sent once."""
    from gateway import run as gateway_run

    _install_fakes(monkeypatch, tmp_path)
    fake_db = MagicMock(get_compression_failure_cooldown=lambda _sid: {"remaining_seconds": 0})
    runner = _make_runner(monkeypatch, tmp_path, session_db=SimpleNamespace(_db=fake_db))

    session_entry = _make_session_entry(sid="session-send-me")
    attempt = _AttemptStub()
    hs = SimpleNamespace(
        failure_cooldown_seconds=300.0,
        max_turn_hold_seconds=10.0,
    )

    sent: list = []
    runner._hmwa_hygiene_notify = _make_notify(sent)
    runner._hmwa_hygiene_stamp = _noop
    runner._hmwa_hygiene_defer_cleanup = _noop
    runner._hmwa_hygiene_cancel_or_adopt = _async_noop

    import asyncio
    try:
        asyncio.run(runner._hmwa_hygiene_on_turn_hold(
            attempt, hs, session_entry, session_entry.session_key, SimpleNamespace(),
        ))
    except Exception as e:
        pass
    assert len(sent) == 1, f"expected one notice, got {sent!r}"
    text, why = sent[0]
    # The body passes the i18n key to t() which translates to the actual user-facing
    # message. We just check the call was made and the message is non-empty.
    assert text, "expected non-empty message text"
    assert why == "compression-turnhold notice"


def test_turnhold_notice_sent_when_cooldown_check_unavailable(monkeypatch, tmp_path):
    """No session db at all => fail-open: notice is still sent."""
    from gateway import run as gateway_run

    _install_fakes(monkeypatch, tmp_path)
    runner = _make_runner(monkeypatch, tmp_path, session_db=None)

    session_entry = _make_session_entry(sid="session-fail-open")
    attempt = _AttemptStub()
    hs = SimpleNamespace(
        failure_cooldown_seconds=300.0,
        max_turn_hold_seconds=10.0,
    )

    sent: list = []
    runner._hmwa_hygiene_notify = _make_notify(sent)
    runner._hmwa_hygiene_stamp = _noop
    runner._hmwa_hygiene_defer_cleanup = _noop
    runner._hmwa_hygiene_cancel_or_adopt = _async_noop

    import asyncio
    try:
        asyncio.run(runner._hmwa_hygiene_on_turn_hold(
            attempt, hs, session_entry, session_entry.session_key, SimpleNamespace(),
        ))
    except Exception as e:
        pass
    assert len(sent) == 1, f"expected fail-open notice, got {sent!r}"
    text, why = sent[0]
    # The body passes the i18n key to t() which translates to the actual user-facing
    # message. We just check the call was made and the message is non-empty.
    assert text, "expected non-empty message text"
    assert why == "compression-turnhold notice"

"""Tests for Honcho dialectic truncation behavior — issue #59469.

The bug: ``honcho_reasoning`` tool results were silently truncated to
``dialecticMaxChars`` (default 600 chars) — the auto-injection budget
wrongly applied to a tool the model explicitly called. The fix:
``dialectic_query()`` accepts ``apply_injection_cap: bool = True``; only
the auto-injection path applies the cap; the explicit ``honcho_reasoning``
tool path passes ``False``. When truncation does happen (auto-injection
path on a long result) we now append a visible
``[truncated, full result in logs]`` marker and log at WARN level.

Tests:
- Short result → no truncation regardless of cap setting.
- Long result with cap on (auto-injection) → truncated, marker visible,
  WARN logged, original length emitted in the warning.
- Long result with cap off (honcho_reasoning tool path) → returned
  untouched, no marker, no WARN.

See https://github.com/NousResearch/hermes-agent/issues/59469.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakePeerChat:
    """Stand-in for a Honcho peer whose ``.chat()`` returns a fixed string."""

    def __init__(self, response: str):
        self._response = response
        self.calls: list[dict] = []

    def chat(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self._response


class _FakePeerFactory:
    """``_get_or_create_peer`` replacement that returns the configured chat peer."""

    def __init__(self, peer_obj: _FakePeerChat):
        self._peer = peer_obj

    def __call__(self, *_args, **_kwargs):
        return self._peer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(*, dialectic_max_chars: int) -> HonchoSessionManager:
    """Build a HonchoSessionManager wired with the minimum viable config."""
    cfg = SimpleNamespace(
        write_frequency="turn",
        dialectic_reasoning_level="low",
        dialectic_dynamic=True,
        dialectic_max_chars=dialectic_max_chars,
        observation_mode="directional",
        user_observe_me=True,
        user_observe_others=True,
        ai_observe_me=True,
        ai_observe_others=False,  # simpler code path: target = self
        message_max_chars=25000,
        dialectic_max_input_chars=10000,
    )
    mgr = HonchoSessionManager(honcho=SimpleNamespace(), config=cfg)
    session = HonchoSession(
        key="s1",
        user_peer_id="u1",
        assistant_peer_id="a1",
        honcho_session_id="hs1",
    )
    mgr._cache[session.key] = session
    return mgr, session


def _wire_peer(mgr: HonchoSessionManager, response: str) -> _FakePeerChat:
    """Install a fake peer factory whose ``.chat()`` returns ``response``."""
    peer = _FakePeerChat(response)
    mgr._get_or_create_peer = _FakePeerFactory(peer)  # type: ignore[assignment]
    return peer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_short_result_is_not_truncated_when_cap_applied():
    """Below-cap result must pass through unmodified, no marker, no warn."""
    mgr, _session = _make_manager(dialectic_max_chars=600)
    _wire_peer(mgr, "x" * 100)  # 100 chars, well under 600

    result = mgr.dialectic_query("s1", "q", apply_injection_cap=True)

    assert result == "x" * 100
    assert "[truncated, full result in logs]" not in result


def test_long_result_is_truncated_with_marker_when_cap_applied():
    """Above-cap result on the auto-injection path must be visibly truncated.

    Regression test for #59469: previously truncated silently with a bare
    trailing " …". Now appends ``[truncated, full result in logs]`` so
    callers can detect data loss instead of inferring it from the missing
    tail.
    """
    mgr, _session = _make_manager(dialectic_max_chars=50)
    long_text = "lorem ipsum " * 50  # 600 chars
    _wire_peer(mgr, long_text)

    result = mgr.dialectic_query("s1", "q", apply_injection_cap=True)

    assert "[truncated, full result in logs]" in result
    # The " …" elision marker still appears just before the truncation marker
    assert " … [truncated, full result in logs]" in result
    # The truncated output is bounded — we don't keep the entire 600 chars
    assert len(result) < len(long_text)


def test_long_result_is_preserved_when_cap_not_applied():
    """honcho_reasoning tool path (apply_injection_cap=False) returns full text.

    The whole point of the fix: the explicit tool must not silently lose
    data to the auto-injection budget. Body must be returned untouched.
    """
    mgr, _session = _make_manager(dialectic_max_chars=50)
    long_text = "lorem ipsum " * 50  # 600 chars, far above the 50 cap
    _wire_peer(mgr, long_text)

    result = mgr.dialectic_query("s1", "q", apply_injection_cap=False)

    # Full text returned, untouched.
    assert result == long_text
    assert "[truncated, full result in logs]" not in result
    assert " …" not in result


def test_truncation_logs_warning_with_original_length(caplog):
    """When truncation happens, a WARN is emitted naming the original length.

    No more silent failure — operators must see when the cap clipped data
    so they can tune ``dialecticMaxChars`` or fix the over-long prompt.
    """
    mgr, _session = _make_manager(dialectic_max_chars=50)
    long_text = "x" * 500  # 500 chars
    _wire_peer(mgr, long_text)

    with caplog.at_level(logging.WARNING, logger="plugins.memory.honcho.session"):
        result = mgr.dialectic_query("s1", "q", apply_injection_cap=True)

    assert "[truncated, full result in logs]" in result

    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warns, "expected at least one WARN log when truncation occurred"
    msg = warns[0].getMessage()
    assert "truncated" in msg.lower()
    # The original 500-char length must be visible in the warning so the
    # operator can size the cap.
    assert "500" in msg
    # The cap (50) must also be visible.
    assert "50" in msg


def test_short_result_does_not_log_warning(caplog):
    """Below-cap results must not emit a truncation warning (no false positives)."""
    mgr, _session = _make_manager(dialectic_max_chars=600)
    _wire_peer(mgr, "x" * 100)

    with caplog.at_level(logging.WARNING, logger="plugins.memory.honcho.session"):
        mgr.dialectic_query("s1", "q", apply_injection_cap=True)

    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not warns, (
        "short result should not produce a truncation WARN; got: "
        f"{[r.getMessage() for r in warns]}"
    )


def test_long_result_with_cap_off_does_not_log_warning(caplog):
    """honcho_reasoning tool path must never emit a truncation warning."""
    mgr, _session = _make_manager(dialectic_max_chars=50)
    _wire_peer(mgr, "x" * 500)

    with caplog.at_level(logging.WARNING, logger="plugins.memory.honcho.session"):
        result = mgr.dialectic_query("s1", "q", apply_injection_cap=False)

    assert result == "x" * 500
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not warns, (
        "apply_injection_cap=False must not emit truncation WARNs; got: "
        f"{[r.getMessage() for r in warns]}"
    )


def test_default_behavior_keeps_backward_compatible_cap_path():
    """Default apply_injection_cap=True preserves prior auto-injection behavior.

    The existing auto-injection callers (``_run_dialectic_depth`` etc.)
    don't pass the new kwarg, so they must still get the cap applied —
    only the cap marker text changes from " …" to the more explicit
    " … [truncated, full result in logs]".
    """
    mgr, _session = _make_manager(dialectic_max_chars=50)
    _wire_peer(mgr, "x" * 500)

    # No kwarg → default True
    result = mgr.dialectic_query("s1", "q")

    assert "[truncated, full result in logs]" in result
    assert len(result) < 500
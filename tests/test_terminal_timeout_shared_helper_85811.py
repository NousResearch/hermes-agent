"""Tests for the shared TERMINAL_TIMEOUT resolution helper.

Regression for issue #85809 / review of #85811: TERMINAL_TIMEOUT was read
independently by tools/terminal_tool.py::_get_env_config AND
tools/process_registry.py's wait() path -- fixing only the first still
left the second producing a misleading "timed out after 0s" for
TERMINAL_TIMEOUT<=0. resolve_terminal_timeout_default() is the single
shared helper both now call, so the guard covers the whole class, not one
call site.
"""

from __future__ import annotations

from unittest.mock import patch

from utils import TERMINAL_TIMEOUT_DEFAULT_SECONDS, resolve_terminal_timeout_default


def test_default_is_180_seconds():
    assert TERMINAL_TIMEOUT_DEFAULT_SECONDS == 180


def test_zero_falls_back_to_default():
    with patch.dict("os.environ", {"TERMINAL_TIMEOUT": "0"}, clear=True):
        assert resolve_terminal_timeout_default() == 180


def test_negative_falls_back_to_default():
    with patch.dict("os.environ", {"TERMINAL_TIMEOUT": "-30"}, clear=True):
        assert resolve_terminal_timeout_default() == 180


def test_unparseable_string_falls_back_to_default():
    with patch.dict("os.environ", {"TERMINAL_TIMEOUT": "5m"}, clear=True):
        assert resolve_terminal_timeout_default() == 180


def test_positive_value_passes_through_unchanged():
    with patch.dict("os.environ", {"TERMINAL_TIMEOUT": "600"}, clear=True):
        assert resolve_terminal_timeout_default() == 600


def test_unset_uses_default():
    with patch.dict("os.environ", {}, clear=True):
        assert resolve_terminal_timeout_default() == 180


def test_process_registry_wait_actually_uses_the_shared_helpers_return_value():
    """Regression: process_registry.py's wait() must genuinely call
    resolve_terminal_timeout_default() and use its return value as the
    effective timeout -- a reversion to a standalone
    `int(os.getenv("TERMINAL_TIMEOUT", "180"))` would silently
    reintroduce the unguarded-second-reader bug (issue #85809 / review
    of #85811).

    Verified behaviorally: patches the shared helper to return a small,
    distinct value (2s) and confirms wait() on a session that never
    finishes gives up in roughly that time, not the real 180s default
    and not near-instantly (which is what TERMINAL_TIMEOUT=0 produced
    before this fix, on this exact call site)."""
    import time as _time
    from unittest.mock import patch as _patch

    import tools.process_registry as pr_mod

    registry = pr_mod.ProcessRegistry()
    session = pr_mod.ProcessSession(
        id="proc_shared_timeout_test",
        command="sleep 999",
        task_id="t1",
        started_at=_time.time(),
        exited=False,
        exit_code=None,
        output_buffer="",
    )
    registry._running[session.id] = session

    with _patch("utils.resolve_terminal_timeout_default", return_value=2):
        start = _time.monotonic()
        result = registry.wait(session.id, timeout=None)
        elapsed = _time.monotonic() - start

    assert result["status"] == "timeout", result
    assert 1.5 <= elapsed <= 4.0, (
        f"wait() took {elapsed:.2f}s -- expected ~2s (the patched shared "
        f"helper's return value), not near-instant (the TERMINAL_TIMEOUT=0 "
        f"bug this guards against) or the real 180s unpatched default"
    )

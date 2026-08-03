"""MoA one-shot model override must be restored on both success and failure.

These exercise the real ``GatewayRunner._restore_moa_one_shot`` helper that the
message-handling ``finally`` block calls, so they prove the production logic —
not a re-implementation of it. The bug being guarded: the restore used to live
in the ``try`` block, so a turn that raised skipped it and the MoA override
leaked permanently (every later message silently fanned out through MoA).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.run import GatewayRunner


def _make_runner():
    """Minimal GatewayRunner with only the fields _restore_moa_one_shot reads."""
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner._evict_cached_agent = MagicMock()
    return runner


def _make_event(moa_disable=False, moa_restore=None):
    event = SimpleNamespace()
    if moa_disable:
        event._moa_disable_after_turn = True
        event._moa_restore_override = moa_restore
    return event


def test_restore_reverts_to_previous_override():
    """A one-shot turn restores the prior per-session model override."""
    runner = _make_runner()
    key = "agent:main:telegram:dm:123"
    runner._session_model_overrides[key] = {"provider": "moa", "model": "default"}
    event = _make_event(
        moa_disable=True,
        moa_restore={"provider": "openrouter", "model": "gpt-4"},
    )

    runner._restore_moa_one_shot(event, key)

    assert runner._session_model_overrides[key] == {
        "provider": "openrouter",
        "model": "gpt-4",
    }
    runner._evict_cached_agent.assert_called_once_with(key)


def test_restore_none_clears_override():
    """If the user had no override before /moa, the MoA override is removed."""
    runner = _make_runner()
    key = "agent:main:discord:guild:456"
    runner._session_model_overrides[key] = {"provider": "moa", "model": "default"}
    event = _make_event(moa_disable=True, moa_restore=None)

    runner._restore_moa_one_shot(event, key)

    assert key not in runner._session_model_overrides
    runner._evict_cached_agent.assert_called_once_with(key)


def test_no_restore_for_non_one_shot_turn():
    """Normal (non-MoA) turns must not touch model overrides or evict agents."""
    runner = _make_runner()
    key = "agent:main:slack:channel:789"
    runner._session_model_overrides[key] = {"provider": "openrouter", "model": "gpt-4"}
    event = _make_event()  # no _moa_disable_after_turn

    runner._restore_moa_one_shot(event, key)

    assert runner._session_model_overrides[key] == {
        "provider": "openrouter",
        "model": "gpt-4",
    }
    runner._evict_cached_agent.assert_not_called()


def test_restore_runs_from_finally_even_when_turn_raises():
    """The whole point of the fix: a raising turn still reverts the override.

    Mirrors the real call site — the restore is invoked from a ``finally`` block,
    so it fires after an exception propagates out of the turn body.
    """
    runner = _make_runner()
    key = "agent:main:telegram:dm:999"
    runner._session_model_overrides[key] = {"provider": "moa", "model": "default"}
    event = _make_event(
        moa_disable=True,
        moa_restore={"provider": "openrouter", "model": "gpt-4"},
    )

    with __import__("pytest").raises(RuntimeError):
        try:
            raise RuntimeError("provider error mid-turn")
        finally:
            runner._restore_moa_one_shot(event, key)

    assert runner._session_model_overrides[key] == {
        "provider": "openrouter",
        "model": "gpt-4",
    }

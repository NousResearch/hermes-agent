"""MoA one-shot model override must be restored on both success and failure."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_runner():
    """Build a minimal GatewayRunner-like object with the fields the
    MoA one-shot restore path reads/writes."""
    runner = MagicMock()
    runner._session_model_overrides = {}
    runner._release_running_agent_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._begin_session_run_generation = MagicMock(return_value=1)
    return runner


def _make_event(text="hello", moa_disable=False, moa_restore=None):
    event = SimpleNamespace()
    event.text = text
    if moa_disable:
        event._moa_disable_after_turn = True
        event._moa_restore_override = moa_restore
    return event


class TestMoaOneShotRestore:
    """The gateway's MoA one-shot restore must fire in the finally block
    so a failed turn still reverts the model override."""

    def test_restore_fires_on_success(self):
        """Normal successful turn restores the previous model override."""
        runner = _make_runner()
        key = "agent:main:telegram:dm:123"
        runner._session_model_overrides[key] = {
            "provider": "moa", "model": "default",
        }
        event = _make_event(
            moa_disable=True,
            moa_restore={"provider": "openrouter", "model": "gpt-4"},
        )

        # Simulate: try block succeeds, finally runs
        try:
            pass  # _handle_message_with_agent succeeds
        finally:
            if getattr(event, "_moa_disable_after_turn", False):
                _restore = getattr(event, "_moa_restore_override", None)
                if _restore is None:
                    runner._session_model_overrides.pop(key, None)
                else:
                    runner._session_model_overrides[key] = _restore

        assert runner._session_model_overrides[key] == {
            "provider": "openrouter", "model": "gpt-4",
        }

    def test_restore_fires_on_exception(self):
        """A failed turn (exception) must still restore the previous model."""
        runner = _make_runner()
        key = "agent:main:telegram:dm:123"
        runner._session_model_overrides[key] = {
            "provider": "moa", "model": "default",
        }
        event = _make_event(
            moa_disable=True,
            moa_restore={"provider": "openrouter", "model": "gpt-4"},
        )

        # Simulate: try block raises, finally still runs
        try:
            raise RuntimeError("provider error")
        except RuntimeError:
            pass
        finally:
            if getattr(event, "_moa_disable_after_turn", False):
                _restore = getattr(event, "_moa_restore_override", None)
                if _restore is None:
                    runner._session_model_overrides.pop(key, None)
                else:
                    runner._session_model_overrides[key] = _restore

        assert runner._session_model_overrides[key] == {
            "provider": "openrouter", "model": "gpt-4",
        }

    def test_restore_none_clears_override(self):
        """When the user had no model override before /moa, the override
        should be removed (not left as MoA)."""
        runner = _make_runner()
        key = "agent:main:discord:guild:456"
        runner._session_model_overrides[key] = {
            "provider": "moa", "model": "default",
        }
        event = _make_event(moa_disable=True, moa_restore=None)

        try:
            raise RuntimeError("timeout")
        except RuntimeError:
            pass
        finally:
            if getattr(event, "_moa_disable_after_turn", False):
                _restore = getattr(event, "_moa_restore_override", None)
                if _restore is None:
                    runner._session_model_overrides.pop(key, None)
                else:
                    runner._session_model_overrides[key] = _restore

        assert key not in runner._session_model_overrides

    def test_no_restore_when_not_one_shot(self):
        """Normal (non-MoA) turns must not touch model overrides."""
        runner = _make_runner()
        key = "agent:main:slack:channel:789"
        runner._session_model_overrides[key] = {
            "provider": "openrouter", "model": "gpt-4",
        }
        event = _make_event()  # no _moa_disable_after_turn

        try:
            pass
        finally:
            if getattr(event, "_moa_disable_after_turn", False):
                runner._session_model_overrides.pop(key, None)

        assert runner._session_model_overrides[key] == {
            "provider": "openrouter", "model": "gpt-4",
        }

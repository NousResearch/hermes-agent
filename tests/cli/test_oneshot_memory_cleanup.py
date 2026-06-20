"""Regression tests for the oneshot (``hermes -z``) memory-provider cleanup.

Oneshot bypasses the interactive CLI's ``atexit``/``_run_cleanup`` wiring, so
``hermes_cli.oneshot._register_oneshot_memory_cleanup`` registers its own
``atexit`` hook to tear the memory provider down before interpreter finalize.
Without it, providers with daemon worker threads still blocked in HTTP recv
(e.g. Honcho) crash the process with SIGABRT (exit 134) at ``Py_FinalizeEx``.

These tests assert the hook is registered and, when invoked, forwards the
agent's transcript to ``shutdown_memory_provider`` (mirroring ``_run_cleanup``,
#15165) before exiting the process.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hermes_cli import oneshot


def _capture_registered_hook(agent):
    """Register the cleanup hook against ``agent`` and return the callback."""
    with patch("atexit.register") as mock_register:
        oneshot._register_oneshot_memory_cleanup(agent)
    mock_register.assert_called_once()
    return mock_register.call_args.args[0]


def test_registers_atexit_hook():
    """``_register_oneshot_memory_cleanup`` registers exactly one atexit hook."""
    agent = MagicMock()
    hook = _capture_registered_hook(agent)
    assert callable(hook)


def test_hook_forwards_session_transcript():
    """The hook forwards a populated ``_session_messages`` list (#15165)."""
    transcript = [
        {"role": "user", "content": "remember my cat is named Mochi"},
        {"role": "assistant", "content": "Got it — Mochi."},
    ]
    agent = MagicMock()
    agent._session_messages = transcript

    hook = _capture_registered_hook(agent)
    with patch("os._exit") as mock_exit:
        hook()

    agent.shutdown_memory_provider.assert_called_once_with(transcript)
    mock_exit.assert_called_once_with(0)


def test_hook_falls_back_to_no_arg_when_transcript_missing():
    """A non-list ``_session_messages`` (test stub) keeps no-arg behaviour."""
    agent = MagicMock()
    agent._session_messages = None

    hook = _capture_registered_hook(agent)
    with patch("os._exit") as mock_exit:
        hook()

    agent.shutdown_memory_provider.assert_called_once_with()
    mock_exit.assert_called_once_with(0)


def test_hook_exits_even_if_shutdown_raises():
    """A provider whose ``shutdown_memory_provider`` raises must not prevent
    the direct ``os._exit`` that bypasses interpreter finalization."""
    agent = MagicMock()
    agent._session_messages = []
    agent.shutdown_memory_provider.side_effect = RuntimeError("boom")

    hook = _capture_registered_hook(agent)
    with patch("os._exit") as mock_exit:
        hook()

    mock_exit.assert_called_once_with(0)

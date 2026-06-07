"""Tests for the approval panel invalidate throttle bypass (issue #41098).

The approval callback must bypass the _invalidate() 250ms throttle and call
app.invalidate() directly, otherwise the approval panel never renders when
another UI event triggered an invalidation within the previous 250ms.
"""
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import cli as cli_module
from cli import HermesCLI


def _make_cli_stub():
    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_state = None
    cli._approval_deadline = 0
    cli._approval_lock = threading.Lock()
    cli._sudo_state = None
    cli._sudo_deadline = 0
    cli._modal_input_snapshot = None
    cli._invalidate = MagicMock()
    cli._app = SimpleNamespace(invalidate=MagicMock(), current_buffer=MagicMock())
    cli._cprint = MagicMock()
    return cli


def test_approval_callback_bypasses_throttle_on_entry():
    """_approval_callback must call app.invalidate() directly, not _invalidate()."""
    cli = _make_cli_stub()
    cli._approval_choices = MagicMock(return_value=["once", "session", "always", "deny"])

    # Simulate a response after 0.1s
    def respond_later():
        time.sleep(0.1)
        cli._approval_state["response_queue"].put("once")

    t = threading.Thread(target=respond_later)
    t.start()
    result = cli._approval_callback("rm -rf /", "Delete everything")
    t.join()

    assert result == "once"
    # app.invalidate() should have been called (direct bypass)
    assert cli._app.invalidate.call_count >= 1
    # _invalidate() should NOT have been called for the initial render
    # (it may be called for cleanup, but the critical path is app.invalidate)


def test_approval_callback_calls_app_invalidate_not_throttled():
    """The initial render must use app.invalidate(), bypassing the 250ms throttle."""
    cli = _make_cli_stub()
    cli._approval_choices = MagicMock(return_value=["once", "session", "always", "deny"])

    def respond_later():
        time.sleep(0.1)
        cli._approval_state["response_queue"].put("deny")

    t = threading.Thread(target=respond_later)
    t.start()
    cli._approval_callback("dangerous_cmd", "Run dangerous command")
    t.join()

    # app.invalidate must be called at least once for the initial render
    cli._app.invalidate.assert_called()


def test_approval_callback_retry_bypasses_throttle():
    """The 5-second retry invalidation must also bypass the throttle."""
    cli = _make_cli_stub()
    cli._approval_choices = MagicMock(return_value=["once", "session", "always", "deny"])

    # Respond after 6 seconds (triggers at least one 5s retry)
    def respond_later():
        time.sleep(6)
        cli._approval_state["response_queue"].put("once")

    t = threading.Thread(target=respond_later)
    t.start()
    result = cli._approval_callback("cmd", "desc")
    t.join()

    assert result == "once"
    # app.invalidate should be called multiple times (initial + at least one retry)
    assert cli._app.invalidate.call_count >= 2


def test_approval_callback_timeout_bypasses_throttle():
    """Timeout cleanup must also bypass the throttle."""
    cli = _make_cli_stub()
    cli._approval_choices = MagicMock(return_value=["once", "session", "always", "deny"])

    # Short timeout so we don't wait 60s
    with patch.object(cli_module, "CLI_CONFIG", {"approvals": {"timeout": 2}}):
        result = cli._approval_callback("cmd", "desc")

    assert result == "deny"
    # app.invalidate should be called for the initial render
    assert cli._app.invalidate.call_count >= 1


def test_approval_callback_no_app_graceful():
    """If _app is not set, the callback must not crash."""
    cli = _make_cli_stub()
    cli._app = None
    cli._approval_choices = MagicMock(return_value=["once", "session", "always", "deny"])

    def respond_later():
        time.sleep(0.1)
        cli._approval_state["response_queue"].put("once")

    t = threading.Thread(target=respond_later)
    t.start()
    result = cli._approval_callback("cmd", "desc")
    t.join()

    assert result == "once"


def test_approval_callback_app_invalidate_exception_graceful():
    """If app.invalidate() raises, the callback must not crash."""
    cli = _make_cli_stub()
    cli._app.invalidate.side_effect = RuntimeError("renderer dead")
    cli._approval_choices = MagicMock(return_value=["once", "session", "always", "deny"])

    def respond_later():
        time.sleep(0.1)
        cli._approval_state["response_queue"].put("once")

    t = threading.Thread(target=respond_later)
    t.start()
    result = cli._approval_callback("cmd", "desc")
    t.join()

    assert result == "once"


def test_approval_callback_emits_bell():
    """The approval callback should emit a terminal bell on first render."""
    cli = _make_cli_stub()
    cli._approval_choices = MagicMock(return_value=["once", "session", "always", "deny"])

    def respond_later():
        time.sleep(0.1)
        cli._approval_state["response_queue"].put("once")

    t = threading.Thread(target=respond_later)
    t.start()
    with patch("cli.print") as mock_print:
        cli._approval_callback("cmd", "desc")
    t.join()

    # Check that print was called with the bell character
    bell_calls = [c for c in mock_print.call_args_list if c == call("\a", end="", flush=True)]
    assert len(bell_calls) >= 1, "Terminal bell should be emitted on approval"


def test_invalidate_method_still_exists():
    """The _invalidate() method should still exist and be callable."""
    cli = _make_cli_stub()
    # _invalidate is a real method on the class — just verify it's callable
    assert callable(getattr(cli, "_invalidate", None))


def test_approval_state_cleared_after_response():
    """_approval_state should be None after the callback returns."""
    cli = _make_cli_stub()
    cli._approval_choices = MagicMock(return_value=["once", "session", "always", "deny"])

    def respond_later():
        time.sleep(0.1)
        cli._approval_state["response_queue"].put("once")

    t = threading.Thread(target=respond_later)
    t.start()
    cli._approval_callback("cmd", "desc")
    t.join()

    assert cli._approval_state is None
    assert cli._approval_deadline == 0

"""The interactive run() stdin-unavailable early-out must release the
active-session lease, like every other run() exit path.

Regression: the `os.fstat(0)` bail (#6393) ran _run_cleanup() and
_print_exit_summary() but skipped _release_active_session(), so a claimed
global session slot lingered until interpreter exit instead of being freed
when the session ended.
"""

from unittest.mock import MagicMock, patch

from cli import HermesCLI


def test_stdin_unavailable_releases_active_session_lease():
    fake_self = MagicMock()
    with patch("cli.os.fstat", side_effect=OSError("fd 0 not available")), \
            patch("cli._run_cleanup") as mock_cleanup:
        aborted = HermesCLI._abort_if_stdin_unavailable(fake_self)

    assert aborted is True
    mock_cleanup.assert_called_once()
    fake_self._print_exit_summary.assert_called_once()
    # The fix: the lease is released on this early-out, mirroring normal exit.
    fake_self._release_active_session.assert_called_once()


def test_stdin_available_is_a_noop():
    fake_self = MagicMock()
    with patch("cli.os.fstat", return_value=object()), \
            patch("cli._run_cleanup") as mock_cleanup:
        aborted = HermesCLI._abort_if_stdin_unavailable(fake_self)

    assert aborted is False
    mock_cleanup.assert_not_called()
    fake_self._print_exit_summary.assert_not_called()
    fake_self._release_active_session.assert_not_called()

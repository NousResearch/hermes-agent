"""Regression tests for the stdin/tty guard in HermesCLI.run().

When hermes is launched via a pipe (e.g. `curl ... | bash` installer),
fd 0 is a valid file descriptor but not a tty. prompt_toolkit → asyncio →
kqueue then raises OSError: [Errno 22] Invalid argument when asked to
watch fd 0. Previously this produced a user-facing traceback.

These tests validate that:
  1. The pre-flight guard detects non-tty stdin and exits cleanly with a
     helpful message.
  2. The fallback exception handler matches EINVAL (errno 22) and
     EBADF (errno 9) in addition to the "is not registered" KeyError,
     printing a helpful message instead of re-raising.
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_cli(**kwargs):
    import importlib

    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), \
         patch.dict("os.environ", {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}, clear=False):
        import cli as _cli_mod
        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), \
             patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}):
            return _cli_mod.HermesCLI(**kwargs), _cli_mod


class TestStdinTTYGuard:
    """run() should refuse non-tty stdin up front with a helpful message."""

    def test_non_tty_stdin_exits_cleanly_without_traceback(self, capsys):
        cli, _ = _make_cli()
        # Short-circuit everything run() does before the tty check so the
        # test focuses on guard behavior and doesn't require a real session.
        cli._init_agent = MagicMock(return_value=True)
        cli.show_banner = MagicMock()
        cli._print_exit_summary = MagicMock()
        cli._ensure_runtime_credentials = MagicMock(return_value=True)

        # isatty(0) returns False → piped stdin case.
        with patch("os.isatty", return_value=False):
            # run() should return normally — NOT raise.
            cli.run()

        out = capsys.readouterr().out
        assert "stdin is not an interactive terminal" in out
        assert "curl" in out  # helpful hint about the curl|bash case
        # Ensure the exit-summary path was invoked (clean shutdown).
        cli._print_exit_summary.assert_called_once()

    def test_bad_stdin_fstat_exits_cleanly(self, capsys):
        cli, _ = _make_cli()
        cli._init_agent = MagicMock(return_value=True)
        cli.show_banner = MagicMock()
        cli._print_exit_summary = MagicMock()
        cli._ensure_runtime_credentials = MagicMock(return_value=True)

        # fstat(0) raises OSError → pre-flight guard triggers.
        real_fstat = os.fstat

        def fake_fstat(fd):
            if fd == 0:
                raise OSError(9, "Bad file descriptor")
            return real_fstat(fd)

        with patch("os.fstat", side_effect=fake_fstat):
            cli.run()

        out = capsys.readouterr().out
        assert "stdin is not an interactive terminal" in out
        cli._print_exit_summary.assert_called_once()


class TestSelectorErrorMatcher:
    """The fallback exception matcher must catch EINVAL / EBADF / KeyError."""

    def _matcher(self, err):
        """Replicate the matcher from cli.py:run() for unit-testing."""
        _err_str = str(err)
        _errno = getattr(err, "errno", None)
        return (
            "is not registered" in _err_str
            or "Bad file descriptor" in _err_str
            or "Invalid argument" in _err_str
            or _errno in (9, 22)
        )

    def test_matches_einval_from_kqueue(self):
        # The exact error produced by prompt_toolkit → asyncio → kqueue
        # when fd 0 is piped on macOS.
        err = OSError(22, "Invalid argument")
        assert self._matcher(err)

    def test_matches_ebadf(self):
        err = OSError(9, "Bad file descriptor")
        assert self._matcher(err)

    def test_matches_key_error_from_selector(self):
        err = KeyError("'0 is not registered'")
        assert self._matcher(err)

    def test_does_not_match_unrelated_oserror(self):
        err = OSError(2, "No such file or directory")
        assert not self._matcher(err)

    def test_does_not_match_unrelated_key_error(self):
        err = KeyError("'some_other_key'")
        assert not self._matcher(err)

"""Classic-CLI prompt_toolkit output selection: native Windows keeps prompt_toolkit's default
output unless PROMPT_TOOLKIT_NO_CPR opts in to the CPR-disabled output."""

from __future__ import annotations

import sys

import pytest

from cli import (
    _select_classic_cli_pt_output,
    _terminal_may_leak_cpr,
)

@pytest.fixture(autouse=True)
def _clear_cpr_env(monkeypatch):
    for var in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "PROMPT_TOOLKIT_NO_CPR"):
        monkeypatch.delenv(var, raising=False)

class TestClassicCliOutputSelection:

    @pytest.mark.platforms("windows")
    def test_windows_preserves_default_output_selection(self):
        assert _terminal_may_leak_cpr() is False
        assert _select_classic_cli_pt_output(sys.stdout) is None

    @pytest.mark.platforms("windows")
    def test_windows_honors_explicit_no_cpr(self, monkeypatch):
        monkeypatch.setenv("PROMPT_TOOLKIT_NO_CPR", "1")
        assert _terminal_may_leak_cpr() is True
        out = _select_classic_cli_pt_output(sys.stdout)
        # Build may return None if stdout is not a real tty in CI; if it
        # succeeds it must be CPR-disabled.
        assert out is None or out.enable_cpr is False

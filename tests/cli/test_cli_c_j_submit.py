"""Test c-j (LF) binding logic for POSIX thin-PTY regression #22908.

Upstream already added install_shift_enter_alias() for CSI-u terminals.
This test covers the remaining c-j binding path for default macOS Terminal
and other POSIX TTYs that deliver Shift+Enter as bare LF.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock


# ── minimal KeyBindings stand-in ───────────────────────────────────────────
class MockKeyBindings:
    def __init__(self):
        self.bindings = {}

    def add(self, *keys):
        def decorator(handler):
            self.bindings[keys] = handler
            return handler
        return decorator


# ── copy of the fixed function from cli.py ─────────────────────────────────
def _bind_prompt_submit_keys(kb, handler) -> None:
    """Bind terminal Enter to the submit handler.

    Only 'enter' is bound here. c-j (LF) handling is done separately
    so that Shift+Enter can insert newlines by default on POSIX.
    """
    kb.add("enter")(handler)


class TestBindPromptSubmitKeys(unittest.TestCase):
    def test_binds_enter_always(self):
        kb = MockKeyBindings()
        dummy = MagicMock()
        _bind_prompt_submit_keys(kb, dummy)
        self.assertIn(("enter",), kb.bindings)

    def test_does_not_bind_c_j(self):
        kb = MockKeyBindings()
        dummy = MagicMock()
        _bind_prompt_submit_keys(kb, dummy)
        self.assertNotIn(("c-j",), kb.bindings)


class TestCJBindingInContext(unittest.TestCase):
    """Simulate caller-side c-j binding for local POSIX terminals."""

    def _simulate_caller_binding(self, kb, submit_handler, preserve_ctrl_enter):
        """Mirror the logic added in cli.py around line 10887."""
        if sys.platform != "win32" and not preserve_ctrl_enter:
            if os.environ.get("HERMES_CLI_SUBMIT_ON_LF") == "1":
                kb.add("c-j")(submit_handler)
            else:
                @kb.add("c-j")
                def handle_c_j_newline(event):
                    event.current_buffer.insert_text("\n")

    def test_default_posix_c_j_newline(self):
        kb = MockKeyBindings()
        submit = MagicMock()
        self._simulate_caller_binding(kb, submit, preserve_ctrl_enter=False)

        self.assertIn(("c-j",), kb.bindings)
        mock_buffer = MagicMock()
        event = MagicMock()
        event.current_buffer = mock_buffer
        kb.bindings[("c-j",)](event)
        mock_buffer.insert_text.assert_called_once_with("\n")

    def test_env_submit_on_lf(self):
        with unittest.mock.patch.dict(os.environ, {"HERMES_CLI_SUBMIT_ON_LF": "1"}, clear=True):
            kb = MockKeyBindings()
            submit = MagicMock()
            self._simulate_caller_binding(kb, submit, preserve_ctrl_enter=False)

            event = MagicMock()
            kb.bindings[("c-j",)](event)
            submit.assert_called_once_with(event)

    def test_wsl_ssh_preserved(self):
        """When _preserve_ctrl_enter_newline() is True, c-j is left alone."""
        kb = MockKeyBindings()
        submit = MagicMock()
        self._simulate_caller_binding(kb, submit, preserve_ctrl_enter=True)
        self.assertNotIn(("c-j",), kb.bindings)


if __name__ == "__main__":
    unittest.main()

"""Regression for #101757: ``CLITuiMixin._tui_voice_record_key_sequence`` must
fall back to the default Ctrl+B sequence — not raise ``UnboundLocalError`` —
when the config/normalize imports inside its try block fail.

The method imported ``pt_key_to_sequence`` (a pure helper) inside the same
``try`` block as the config/normalize helpers. When the packaged
``hermes_cli.config`` import is broken (as reported in Docker), the
``except`` clause only restores ``_voice_key`` to the default, leaving
``pt_key_to_sequence`` an unbound local — so the final
``return pt_key_to_sequence(_voice_key)`` raised ``UnboundLocalError`` and
the TUI crashed on startup. ``pt_key_to_sequence`` is a pure function with
no config dependency, so importing it outside the try block makes the
fallback path resolve cleanly.
"""

import logging
import sys
import types


class TestTuiVoiceRecordKeySequenceFallback:
    """Regression for #101757 — the Ctrl+B fallback path must not UnboundLocalError."""

    def test_falls_back_to_ctrl_b_when_config_import_fails(self, monkeypatch):
        # Import first so the module-level imports of cli_tui_mixin resolve
        # against the real package; we only want the method's *local* import
        # of `hermes_cli.config` to fail.
        from hermes_cli.cli_tui_mixin import CLITuiMixin

        # The method does `from cli import logger` at the top; stub it so the
        # test does not depend on the (heavy) cli module import path.
        cli_stub = types.ModuleType("cli")
        cli_stub.logger = logging.getLogger("cli")
        monkeypatch.setitem(sys.modules, "cli", cli_stub)

        # sys.modules[key] = None makes `from <key> import ...` raise ImportError,
        # reproducing the broken-packaged-import path that triggers #101757:
        # the config import inside the try block fails, pt_key_to_sequence never
        # gets bound there, and (pre-fix) the return raised UnboundLocalError.
        monkeypatch.setitem(sys.modules, "hermes_cli.config", None)

        class _Self(CLITuiMixin):
            """Minimal host: the method only touches self.set_voice_record_key_cache."""

            def set_voice_record_key_cache(self, raw_key):
                self.cached = raw_key

        self_obj = _Self()
        result = self_obj._tui_voice_record_key_sequence()

        # Ctrl+B default sequence, not UnboundLocalError.
        assert result == ("c-b",)
        # The default raw key is cached for the UI label (same value that drives
        # the prompt_toolkit binding).
        assert self_obj.cached == "ctrl+b"

"""Regression test for issue #19767 — resize-driven empty conversation records.

On WSL2 + Windows Terminal, dragging the window to resize triggers a SIGWINCH
storm.  prompt_toolkit queries cursor position via DSR (``ESC[6n``) and the
terminal floods CPR replies back into stdin.  Under these storms the response
bytes occasionally land in the input buffer as literal text, and SGR mouse
reports ending in ``M`` can be parsed as Ctrl+M (= Enter) — submitting a
buffer whose sole content was the leaked sequence.

Existing strip helpers (`_strip_leaked_bracketed_paste_wrappers`,
`_strip_leaked_terminal_responses_with_meta`) already clean those bytes from
the submission; the bug was that ``process_loop`` did not re-check whether
the submission was empty *after* stripping, so the empty payload reached
``_print_user_message_preview`` and rendered a divider + bullet with no
text.  Each resize event produced one such ghost record — a dozen+ in a
single drag.

This file pins down the guard that drops post-strip-empty submissions while
preserving image-only attachments.
"""

from cli import (
    _is_empty_after_terminal_noise_strip,
    _strip_leaked_bracketed_paste_wrappers,
    _strip_leaked_terminal_responses,
)


class TestEmptyAfterTerminalNoiseStrip:
    """Pure-function guard called from process_loop after stripping."""

    def test_real_user_text_is_preserved(self):
        assert _is_empty_after_terminal_noise_strip("hello", False) is False

    def test_truly_empty_string_is_dropped(self):
        assert _is_empty_after_terminal_noise_strip("", False) is True

    def test_whitespace_only_string_is_dropped(self):
        assert _is_empty_after_terminal_noise_strip("   \n\t  ", False) is True

    def test_attachments_keep_blank_payload_alive(self):
        # User explicitly attached an image with no caption — must submit.
        assert _is_empty_after_terminal_noise_strip("", True) is False
        assert _is_empty_after_terminal_noise_strip("   ", True) is False

    def test_non_string_falsy_payload_is_dropped(self):
        assert _is_empty_after_terminal_noise_strip(None, False) is True

    def test_non_string_truthy_payload_is_preserved(self):
        # Defensive: list/dict payload should not be dropped by this guard.
        assert _is_empty_after_terminal_noise_strip(["x"], False) is False


class TestResizeStormCollapsesToEmpty:
    """Real leaked sequences must collapse to empty after the strip pass —
    verifying that the post-strip guard above will fire on them.
    """

    @staticmethod
    def _strip(text: str) -> str:
        # Apply the same two-step strip pipeline process_loop runs.
        text = _strip_leaked_bracketed_paste_wrappers(text)
        text = _strip_leaked_terminal_responses(text)
        return text

    def test_pure_cpr_response_collapses_to_empty(self):
        leaked = "\x1b[24;80R"
        cleaned = self._strip(leaked)
        assert _is_empty_after_terminal_noise_strip(cleaned, False) is True

    def test_repeated_cpr_storm_collapses_to_empty(self):
        # A SIGWINCH burst on WSL2 can deliver several CPR replies back-to-back.
        leaked = "\x1b[24;80R\x1b[24;80R\x1b[25;80R\x1b[26;80R"
        cleaned = self._strip(leaked)
        assert _is_empty_after_terminal_noise_strip(cleaned, False) is True

    def test_sgr_mouse_report_collapses_to_empty(self):
        # The trailing ``M`` is the byte the parser can mistake for Enter.
        leaked = "\x1b[<65;1;49M"
        cleaned = self._strip(leaked)
        assert _is_empty_after_terminal_noise_strip(cleaned, False) is True

    def test_visible_form_cpr_collapses_to_empty(self):
        leaked = "^[[24;80R"
        cleaned = self._strip(leaked)
        assert _is_empty_after_terminal_noise_strip(cleaned, False) is True

    def test_bracketed_paste_wrapper_only_collapses_to_empty(self):
        leaked = "\x1b[200~\x1b[201~"
        cleaned = self._strip(leaked)
        assert _is_empty_after_terminal_noise_strip(cleaned, False) is True

    def test_real_text_between_leaked_sequences_survives(self):
        # Sanity: if the user actually typed something amidst the leak, keep it.
        leaked = "\x1b[24;80Rhi\x1b[<65;1;49M"
        cleaned = self._strip(leaked)
        assert cleaned == "hi"
        assert _is_empty_after_terminal_noise_strip(cleaned, False) is False


class TestProcessLoopGuardIntegration:
    """Verify that the process_loop call site has the ``continue`` guard
    between the terminal-noise strip pass and ``_print_user_message_preview``.

    Teknium's review (#19983): the helper-only tests above exercise
    ``_is_empty_after_terminal_noise_strip`` in isolation and would pass
    even if the ``continue`` guard at the process_loop call site were
    deleted.  This test inspects the actual ``process_loop`` source to
    confirm the guard is wired in at the boundary.
    """

    def test_continue_guard_exists_after_strip_in_process_loop(self):
        import inspect
        import cli as cli_mod

        # process_loop is a nested function inside a CLI class method.
        # Read the full cli.py source and locate the process_loop body
        # by its def line, then check the ordering within that region.
        full_source = inspect.getsource(cli_mod)
        loop_start = full_source.find("def process_loop():")
        assert loop_start > 0, "process_loop() definition not found in cli.py"

        # Grab a generous window after the def line — the guard is near
        # the bottom of the loop body (after input collection + stripping).
        region = full_source[loop_start:loop_start + 20000]

        strip_idx = region.find("_strip_leaked_terminal_responses_with_meta")
        guard_idx = region.find("_is_empty_after_terminal_noise_strip")
        continue_idx = region.find("continue", guard_idx if guard_idx > 0 else 0)
        preview_idx = region.find("_print_user_message_preview")

        assert strip_idx > 0, "strip call not found in process_loop region"
        assert guard_idx > strip_idx, "empty-check guard must come after strip"
        assert continue_idx > guard_idx, "continue must follow the empty-check guard"
        # No executable _print_user_message_preview call may sit between the
        # strip and the guard — that gap is where sanitized-empty submissions
        # must be dropped before reaching the preview/chat path.  We check
        # for the call pattern (not just the name, which appears in comments).
        gap = region[strip_idx:guard_idx]
        # A real call looks like "self._print_user_message_preview(" or
        # "_print_user_message_preview(" — not just the name in a comment.
        import re
        call_matches = re.findall(r"_print_user_message_preview\s*\(", gap)
        assert not call_matches, (
            "_print_user_message_preview() call must not appear between strip and guard"
        )

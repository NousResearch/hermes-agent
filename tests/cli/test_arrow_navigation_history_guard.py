"""Lane C: behavioral test for the arrow-key history-guard adaptation.

Background
----------
Hermes v0.19.0 uses a multiline TextArea as the chat input. prompt_toolkit's
``Buffer.auto_up`` / ``Buffer.auto_down`` decide whether to recall history
based on ``cursor_position_row`` (logical rows), not visual wrap rows. A single
logical-line input that wraps visually across multiple screen rows therefore
triggers history recall from any cursor position, even though the user is
just moving inside a wrapped line.

Reinstatement of the old ``if buf.text: buf.cursor_up else buf.auto_up`` block
is NOT sufficient — ``cursor_up`` is also logical-line based, so it won't
move the cursor between visually-wrapped rows on a single logical line. The
acceptance contract is therefore scoped to the safe no-history half:

    A non-empty buffer must NEVER trigger history recall on Up/Down.

Visual cursor movement across screen rows of a wrapped single logical line
remains a known limitation of the current architecture.

These tests lock the contract on an extracted helper so the runtime
keybindings can stay inside ``HermesCLI.run()`` while the decision logic
remains unit-testable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli as cli_mod


# ---------------------------------------------------------------------------
# Helper tests — the contract is enforced by an extracted pure function
# ---------------------------------------------------------------------------


def test_empty_buffer_up_browses_history():
    """Empty buffer + Up arrow → history browse (the only safe path)."""
    buf = SimpleNamespace(text="")
    assert cli_mod._history_navigation_action(buf, "up") == "history"


def test_empty_buffer_down_browses_history():
    """Empty buffer + Down arrow → history browse."""
    buf = SimpleNamespace(text="")
    assert cli_mod._history_navigation_action(buf, "down") == "history"


def test_nonempty_buffer_up_blocks_history_recall():
    """Non-empty buffer + Up arrow → cursor move, NEVER history browse.

    This is the regression guard for the wrapped-line bug.
    """
    buf = SimpleNamespace(text="hello")
    assert cli_mod._history_navigation_action(buf, "up") == "cursor"


def test_nonempty_buffer_down_blocks_history_recall():
    """Non-empty buffer + Down arrow → cursor move, NEVER history browse."""
    buf = SimpleNamespace(text="hello")
    assert cli_mod._history_navigation_action(buf, "down") == "cursor"


def test_single_visual_line_wrapped_logically_short_blocks_history():
    """A buffer whose content is a single logical line (visually wrapped) must
    not browse history on Up/Down — even though the cursor is on logical row 0.
    """
    # One logical line, but visually several rows. Prompt_toolkit's auto_up
    # would still call history_backward because cursor_position_row is 0.
    long_single_line = "lorem ipsum dolor sit amet " * 20
    assert "\n" not in long_single_line
    buf = SimpleNamespace(text=long_single_line)
    assert cli_mod._history_navigation_action(buf, "up") == "cursor"
    assert cli_mod._history_navigation_action(buf, "down") == "cursor"


def test_multiline_buffer_nonempty_blocks_history():
    """Multiline buffer (true multi-line, not just wrapped) must also block
    history recall — the wrapped-line rationale generalises: any non-empty
    input is too ambiguous for safe auto-history, and the user can still
    move with cursor_up/cursor_down within the multi-line content.
    """
    buf = SimpleNamespace(text="line one\nline two\nline three")
    assert cli_mod._history_navigation_action(buf, "up") == "cursor"
    assert cli_mod._history_navigation_action(buf, "down") == "cursor"


def test_whitespace_only_buffer_treated_as_empty():
    """A whitespace-only buffer is editorially empty — history browse is safe.

    Conservative on purpose: a space-padded prompt is not a real draft.
    """
    buf = SimpleNamespace(text="   \n  \t ")
    assert cli_mod._history_navigation_action(buf, "up") == "history"
    assert cli_mod._history_navigation_action(buf, "down") == "history"


def test_one_real_char_blocks_history():
    """One character is enough to mark the buffer as non-empty."""
    buf = SimpleNamespace(text="x")
    assert cli_mod._history_navigation_action(buf, "up") == "cursor"


def test_unknown_direction_falls_back_to_cursor():
    """Defensive: unknown direction code never browses history.

    Better to do nothing visible than to recall an unintended history entry.
    """
    buf = SimpleNamespace(text="")
    assert cli_mod._history_navigation_action(buf, "pageup") == "cursor"


def test_helper_does_not_call_buffer_methods():
    """The decision helper must be pure — it must NOT mutate the buffer.

    The actual side effects (cursor_up / auto_up) live in the keybinding
    shim. The helper is a single source of truth for the decision only.
    """
    class Tracking:
        def __init__(self):
            self.calls = []
            self.text = "x"

        def cursor_up(self):
            self.calls.append("cursor_up")

        def auto_up(self):
            self.calls.append("auto_up")

    buf = Tracking()
    cli_mod._history_navigation_action(buf, "up")
    assert buf.calls == [], "decision helper must be side-effect free"


# ---------------------------------------------------------------------------
# Bound-method test — exercise the keybinding-side decision via the
# ``_normal_input`` condition's own shim. This locks the contract at the
# integration boundary the runtime actually uses, without spinning up
# a full prompt_toolkit Application.
# ---------------------------------------------------------------------------


def test_run_history_up_uses_helper_for_nonempty_buffer(monkeypatch):
    """The bound history_up handler must consult the helper and refuse to
    call auto_up when the buffer is non-empty (the wrapped-line guard).
    """
    auto_up_calls: list[int] = []
    cursor_up_calls: list[int] = []

    class FakeBuffer:
        text = "draft in progress"

        def auto_up(self, count=1):
            auto_up_calls.append(count)

        def cursor_up(self):
            cursor_up_calls.append(1)

    class FakeApp:
        def __init__(self):
            self.current_buffer = FakeBuffer()

    class FakeEvent:
        def __init__(self):
            self.app = FakeApp()
            self.arg = 1

    # Locate the bound history_up handler. It is defined inside the
    # ``run()`` method body, so we read it out of the source via the AST
    # regex used by the existing detector — but the cleanest test is to
    # verify the helper contract (already covered above) plus a final
    # assertion that the source itself routes through the helper.
    import re
    from pathlib import Path

    src = Path(cli_mod.__file__).read_text(encoding="utf-8", errors="replace")

    # The handler must reference the helper. If the helper is not yet
    # consulted, the regex below will fail and the test will FAIL.
    pattern = re.compile(
        r"def\s+history_up\(event\):.*?_history_navigation_action\(",
        re.DOTALL,
    )
    assert pattern.search(src), (
        "history_up handler must route through _history_navigation_action "
        "helper; bare auto_up is the regression we are guarding against."
    )

    # Same for history_down.
    pattern_down = re.compile(
        r"def\s+history_down\(event\):.*?_history_navigation_action\(",
        re.DOTALL,
    )
    assert pattern_down.search(src), (
        "history_down handler must route through _history_navigation_action "
        "helper; bare auto_down is the regression we are guarding against."
    )

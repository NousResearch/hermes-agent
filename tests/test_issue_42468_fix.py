"""Regression test for issue #42468 – nested Radix menu conflict.

The SessionContextMenu (Radix ContextMenu wrapping the entire row) was
intercepting pointer/click events from the inner SessionActionsMenu
(Radix DropdownMenu for the '...' button), preventing ``onSelect`` from
firing on DropdownMenuItem (including "Copy ID").

Fix: added ``onContextMenu`` and ``onPointerDown`` handlers with
``stopPropagation()`` on the ``DropdownMenuTrigger`` so the outer
``ContextMenu`` cannot swallow the events.
"""

from pathlib import Path

import pytest

# Path to the fixed source file (relative to repo root).
SRC = (
    Path(__file__).resolve().parent.parent
    / "apps"
    / "desktop"
    / "src"
    / "app"
    / "chat"
    / "sidebar"
    / "session-actions-menu.tsx"
)


def _read_source() -> str:
    return SRC.read_text(encoding="utf-8")


class TestDropdownMenuTriggerPropagationGuard:
    """Verify that the DropdownMenuTrigger prevents event propagation to the
    outer ContextMenu."""

    def test_source_file_exists(self) -> None:
        """The fixed source file must exist."""
        assert SRC.is_file(), f"Expected source file at {SRC}"

    def test_has_stop_propagation_on_pointer_down(self) -> None:
        """DropdownMenuTrigger must have onPointerDown with stopPropagation."""
        src = _read_source()
        assert "onPointerDown" in src, (
            "DropdownMenuTrigger should have onPointerDown handler"
        )
        assert "stopPropagation" in src, (
            "stopPropagation must be called to prevent ContextMenu interception"
        )

    def test_has_stop_propagation_on_context_menu(self) -> None:
        """DropdownMenuTrigger must have onContextMenu with stopPropagation."""
        src = _read_source()
        assert "onContextMenu" in src, (
            "DropdownMenuTrigger should have onContextMenu handler"
        )

    def test_stop_propagation_is_on_dropdown_trigger(self) -> None:
        """The stopPropagation handlers must appear on the DropdownMenuTrigger
        (not just anywhere in the file)."""
        src = _read_source()
        # Find the DropdownMenuTrigger block and verify it contains
        # the propagation guards.
        trigger_start = src.find("<DropdownMenuTrigger")
        assert trigger_start != -1, "DropdownMenuTrigger not found"
        # Grab until the closing tag of the trigger
        trigger_end = src.find("</DropdownMenuTrigger>", trigger_start)
        trigger_block = src[trigger_start:trigger_end]
        assert "stopPropagation" in trigger_block, (
            "stopPropagation must be on the DropdownMenuTrigger, "
            "not elsewhere in the file"
        )
        assert "onPointerDown" in trigger_block, (
            "onPointerDown must be on the DropdownMenuTrigger"
        )
        assert "onContextMenu" in trigger_block, (
            "onContextMenu must be on the DropdownMenuTrigger"
        )

    def test_as_child_is_preserved(self) -> None:
        """DropdownMenuTrigger must still use asChild so the children merge."""
        src = _read_source()
        trigger_start = src.find("<DropdownMenuTrigger")
        trigger_end = src.find("</DropdownMenuTrigger>", trigger_start)
        trigger_block = src[trigger_start:trigger_end]
        assert "asChild" in trigger_block, (
            "DropdownMenuTrigger must keep asChild attribute"
        )

    def test_comment_references_issue(self) -> None:
        """The fix should reference issue #42468 in a comment."""
        src = _read_source()
        assert "42468" in src, (
            "Source should contain a reference to issue #42468"
        )

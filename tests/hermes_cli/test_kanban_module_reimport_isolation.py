"""Re-importing the Hermes modules must not outlive the test that asked for it.

A handful of kanban tests need a fresh ``HERMES_HOME`` to be read at *import*
time, so they drop ``hermes_cli`` from ``sys.modules`` and import it again.
Dropping the entries and walking away leaves two copies of the same code alive
for the rest of the session: the one every already-imported test module holds,
and the one ``sys.modules`` hands to the next importer.

That split has teeth. ``_kanban_write_guard`` in ``tests/conftest.py`` patches
``connect`` on the copy it finds in ``sys.modules``, so a test module whose
global still points at the other copy calls an UNGUARDED ``connect``: the
deny-list that keeps the suite off the operator's real ``~/.hermes`` (#69283)
stops working, and says nothing about it. It also lets a test observe a
module-level registry on one copy while the code under test mutates the other,
which is what made a dozen hook and lifecycle tests fail in a full run and pass
one file at a time.

These tests pin the restore that ``fresh_hermes_modules`` performs.
"""
from __future__ import annotations

import sys

import pytest

from hermes_cli import kanban_db
from tests.conftest import fresh_hermes_modules


def test_the_original_module_object_is_back_afterwards() -> None:
    before = sys.modules["hermes_cli.kanban_db"]
    assert before is kanban_db

    with fresh_hermes_modules():
        from hermes_cli import kanban_db as inner

        # The point of the block: a genuinely fresh module, so import-time
        # reads of HERMES_HOME happen again.
        assert inner is not before

    assert sys.modules["hermes_cli.kanban_db"] is before


def test_the_write_guard_still_covers_the_module_everyone_holds() -> None:
    """The consequence, stated directly.

    The autouse guard patched ``connect`` on the module object this file
    imported. A re-import that left the fresh copy in ``sys.modules`` would
    hand the next importer an unguarded ``connect`` -- so the invariant worth
    asserting is that the entry, after the block, is still the guarded one.
    """
    guarded = sys.modules["hermes_cli.kanban_db"].connect
    assert guarded.__qualname__.startswith("_kanban_write_guard"), (
        "the autouse guard did not patch connect -- this test cannot say "
        "anything about the restore"
    )

    with fresh_hermes_modules():
        from hermes_cli import kanban_db as inner

        # The hazard, named: the fresh copy is NOT guarded.
        assert inner.connect is not guarded

    assert sys.modules["hermes_cli.kanban_db"].connect is guarded


def test_the_guard_refuses_the_real_root_after_a_reimport() -> None:
    """End to end: the deny-list still fires for this module's own reference."""
    import tests.conftest as _conftest

    with fresh_hermes_modules():
        from hermes_cli import kanban_db as inner  # noqa: F401

    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kanban_db.connect(_conftest._REAL_KANBAN_ROOT / "kanban.db")


def test_the_block_restores_even_when_the_body_raises() -> None:
    before = sys.modules["hermes_cli.kanban_db"]
    with pytest.raises(ZeroDivisionError):
        with fresh_hermes_modules():
            import hermes_cli.kanban_db  # noqa: F401

            1 / 0
    assert sys.modules["hermes_cli.kanban_db"] is before

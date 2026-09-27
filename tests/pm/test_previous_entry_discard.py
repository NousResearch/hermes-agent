"""Locked stale previous entries must not fail the install that replaced them."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from pm.install import _discard_previous_entry


def test_discard_previous_entry_swallows_locked_file():
    store = SimpleNamespace()
    previous = SimpleNamespace(name=".previous-python-test")
    with patch("pm.install._remove_entry", side_effect=PermissionError(5, "Access is denied")):
        _discard_previous_entry(store, previous)


def test_discard_previous_entry_removes_when_unlocked():
    store = SimpleNamespace()
    previous = SimpleNamespace(name=".previous-python-test")
    with patch("pm.install._remove_entry") as remove:
        _discard_previous_entry(store, previous)
    remove.assert_called_once_with(store, ".previous-python-test")

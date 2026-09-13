"""Regression: ensure_matrix_deps() fresh-dependency rebind must not NameError.

sdk-bindings-review.md #43: ``_import()`` referenced ``PaginationDirection`` /
``SyncToken`` that were never imported, and ``pm.extras.ensure_and_bind`` catches
only ImportError — so a fresh install (missing packages) crashed adapter creation
instead of rebinding the type globals or returning False with the install hint.

Tests run through the real ``ensure_and_bind`` boundary with the install seam
(``pm.extras.ensure_import``) no-op'd — no network, no live Matrix.
"""
import sys
import types

import pytest
from unittest.mock import patch

import pm.extras as pm_extras
from plugins.platforms.matrix import adapter as matrix_adapter


def _fake_mautrix_types():
    """Minimal mautrix.types with the 8 names the adapter imports/binds."""
    mod = types.ModuleType("mautrix.types")

    class EventType:
        ROOM_MESSAGE = "m.room.message"
        REACTION = "m.reaction"

    class UserID(str):
        pass

    class RoomID(str):
        pass

    class EventID(str):
        pass

    class ContentURI(str):
        pass

    class RoomCreatePreset:
        PRIVATE = "private_chat"

    class PresenceState:
        ONLINE = "online"

    class TrustState:
        UNVERIFIED = 0

    mod.EventType = EventType
    mod.UserID = UserID
    mod.RoomID = RoomID
    mod.EventID = EventID
    mod.ContentURI = ContentURI
    mod.RoomCreatePreset = RoomCreatePreset
    mod.PresenceState = PresenceState
    mod.TrustState = TrustState
    return mod


@pytest.fixture
def fresh_dependency_boundary(monkeypatch):
    """Simulate a fresh install: ``missing()`` reports a gap, the install is a
    no-op success, and ``from mautrix.types import ...`` resolves against a fake."""
    monkeypatch.setattr(pm_extras, "missing", lambda extra: ("asyncpg",))
    monkeypatch.setattr(pm_extras, "ensure_import", lambda *a, **kw: None)
    monkeypatch.delenv("MATRIX_E2EE_MODE", raising=False)
    monkeypatch.delenv("MATRIX_ENCRYPTION", raising=False)
    # ensure_and_bind writes module globals; isolate even successful rebinding.
    for name in ("EventType", "UserID", "RoomID", "EventID", "ContentURI", "RoomCreatePreset", "PresenceState", "TrustState"):
        monkeypatch.setattr(matrix_adapter, name, getattr(matrix_adapter, name))
    fake_types = _fake_mautrix_types()
    mautrix = types.ModuleType("mautrix")
    mautrix.types = fake_types
    with patch.dict(sys.modules, {"mautrix": mautrix, "mautrix.types": fake_types}):
        yield fake_types


def test_fresh_install_rebinds_type_globals_without_nameerror(fresh_dependency_boundary):
    assert matrix_adapter.ensure_matrix_deps() is True
    # The rebind actually landed on the adapter module globals.
    assert matrix_adapter.EventType is fresh_dependency_boundary.EventType
    assert matrix_adapter.UserID is fresh_dependency_boundary.UserID
    assert matrix_adapter.TrustState is fresh_dependency_boundary.TrustState


def test_failed_install_returns_false_with_hint_and_never_raises(
    fresh_dependency_boundary, caplog
):
    # Same fresh state but the post-install import genuinely fails (no mautrix):
    # must return False with the install hint — and, per the bug class, must not
    # leak anything other than ImportError out of ensure_and_bind.
    with patch.dict(sys.modules, {"mautrix": None, "mautrix.types": None}):
        with caplog.at_level("WARNING", logger="plugins.platforms.matrix.adapter"):
            assert matrix_adapter.ensure_matrix_deps() is False
    assert any("required packages not installed" in r.message for r in caplog.records)

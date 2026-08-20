"""Regression: rebind mautrix EventType after import-time stubs.

When the Matrix plugin module is imported before mautrix is importable, the
adapter installs string stubs (``EventType.ROOM_MESSAGE = "m.room.message"``).
If mautrix later becomes available without rebinding those globals,
``Client.add_event_handler`` raises ``ValueError("Invalid event type")`` right
after E2EE setup — Matrix never finishes connecting.

These tests lock the rebind path in ``ensure_matrix_deps``.
"""
from __future__ import annotations

import sys
import types

import plugins.platforms.matrix.adapter as matrix_adapter


class _StubEventType:
    """Mirrors the import-time fallback in adapter.py when mautrix is missing."""

    ROOM_MESSAGE = "m.room.message"
    REACTION = "m.reaction"
    ROOM_ENCRYPTED = "m.room.encrypted"
    ROOM_NAME = "m.room.name"


class _LiveEventType:
    """Stand-in for a real mautrix EventType class (identity matters)."""

    ROOM_MESSAGE = object()
    REACTION = object()
    ROOM_ENCRYPTED = object()
    ROOM_NAME = object()


def _install_fake_mautrix_types(monkeypatch):
    """Provide ``mautrix.types`` symbols that ``ensure_matrix_deps`` imports."""
    mautrix = types.ModuleType("mautrix")
    mautrix_types = types.ModuleType("mautrix.types")
    for name, value in {
        "ContentURI": str,
        "EventID": str,
        "EventType": _LiveEventType,
        "PaginationDirection": object(),
        "PresenceState": object(),
        "RoomCreatePreset": object(),
        "RoomID": str,
        "SyncToken": str,
        "TrustState": object(),
        "UserID": str,
    }.items():
        setattr(mautrix_types, name, value)
    mautrix.types = mautrix_types
    monkeypatch.setitem(sys.modules, "mautrix", mautrix)
    monkeypatch.setitem(sys.modules, "mautrix.types", mautrix_types)


def test_ensure_matrix_deps_rebinds_string_stubs_via_ensure_and_bind(monkeypatch):
    """Stubs must be replaced even when feature_missing reports nothing missing."""
    monkeypatch.setattr(matrix_adapter, "EventType", _StubEventType)
    assert isinstance(matrix_adapter.EventType.ROOM_MESSAGE, str)

    import tools.lazy_deps as lazy_deps_mod

    monkeypatch.setattr(lazy_deps_mod, "feature_missing", lambda feature: ())
    _install_fake_mautrix_types(monkeypatch)

    def _ensure_and_bind(feature, importer, target_globals, prompt=False):
        assert feature == "platform.matrix"
        target_globals.update(importer())
        return True

    monkeypatch.setattr(lazy_deps_mod, "ensure_and_bind", _ensure_and_bind)
    monkeypatch.setattr(matrix_adapter, "_resolve_e2ee_mode", lambda *a, **k: "off")

    assert matrix_adapter.ensure_matrix_deps() is True
    assert matrix_adapter.EventType is _LiveEventType
    assert not isinstance(matrix_adapter.EventType.ROOM_MESSAGE, str)


def test_ensure_matrix_deps_direct_rebind_when_ensure_and_bind_missing(monkeypatch):
    """If ensure_and_bind is unavailable, still rebind when types can import."""
    monkeypatch.setattr(matrix_adapter, "EventType", _StubEventType)

    import tools.lazy_deps as lazy_deps_mod

    monkeypatch.setattr(lazy_deps_mod, "feature_missing", lambda feature: ())
    # Force the ``ensure_and_bind is None`` branch.
    monkeypatch.setattr(lazy_deps_mod, "ensure_and_bind", None)
    _install_fake_mautrix_types(monkeypatch)
    monkeypatch.setattr(matrix_adapter, "_resolve_e2ee_mode", lambda *a, **k: "off")

    assert matrix_adapter.ensure_matrix_deps() is True
    assert matrix_adapter.EventType is _LiveEventType
    assert not isinstance(matrix_adapter.EventType.ROOM_MESSAGE, str)


def test_string_event_type_fails_mautrix_isinstance_guard():
    """Document the mautrix Syncer guard this fix prevents tripping."""
    pytest = __import__("pytest")
    try:
        from mautrix.client import InternalEventType
        from mautrix.types import EventType
    except ImportError:
        pytest.skip("mautrix not installed in this environment")

    assert not isinstance("m.room.message", (EventType, InternalEventType))
    assert not isinstance(_StubEventType.ROOM_MESSAGE, (EventType, InternalEventType))

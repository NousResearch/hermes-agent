"""Tests for plugins/memory/holographic/__init__.py.

Most coverage of the holographic provider lives in
``tests/agent/test_memory_provider.py`` (discovery + manager wiring). This
file holds focused tests for behaviour that's specific to the provider
itself — currently the null-store guard that turns silent ``AttributeError``
fallout into an actionable error after a failed ``initialize()``.
"""

from __future__ import annotations

import json

import pytest

from plugins.memory.holographic import HolographicMemoryProvider


def _unwrap_tool_error(payload: str) -> str:
    """Return the human-readable error text from a tool_error JSON envelope."""
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return payload
    if isinstance(parsed, dict):
        for key in ("error", "message", "detail"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value
    return payload


@pytest.fixture
def uninitialized_provider() -> HolographicMemoryProvider:
    """Provider in the state it lands in after a failed ``initialize()``.

    ``MemoryStore.__init__`` raises on a corrupt DB (e.g. crash during WAL
    checkpoint), ``HolographicMemoryProvider.initialize()`` re-raises, and
    ``MemoryManager.initialize_all()`` catches the exception, logs a
    WARNING, and leaves ``self._store`` at its constructor default of
    ``None``. The tool dispatcher keeps the provider registered, so the
    agent can still call ``fact_store`` / ``fact_feedback`` — every call
    hits ``None`` and falls through to the catch-all in the handlers.
    """
    return HolographicMemoryProvider(config={})


@pytest.mark.parametrize(
    "args",
    [
        {"action": "list"},
        {"action": "add", "content": "hi"},
        {"action": "search", "query": "x"},
        {"action": "probe", "entity": "alice"},
    ],
)
def test_fact_store_returns_actionable_error_when_store_uninitialized(
    uninitialized_provider, args,
):
    payload = uninitialized_provider._handle_fact_store(args)
    text = _unwrap_tool_error(payload).lower()

    # The error must NOT be the cryptic AttributeError we get from None._method()
    assert "nonetype" not in text, (
        f"handler should guard None _store instead of leaking AttributeError: {payload!r}"
    )
    # It SHOULD tell the user what's wrong and roughly how to recover.
    assert "memory" in text and "unavailable" in text, (
        f"error should name the affected subsystem and surface the unavailability: {payload!r}"
    )


def test_fact_feedback_returns_actionable_error_when_store_uninitialized(
    uninitialized_provider,
):
    payload = uninitialized_provider._handle_fact_feedback(
        {"action": "helpful", "fact_id": 1}
    )
    text = _unwrap_tool_error(payload).lower()

    assert "nonetype" not in text, (
        f"handler should guard None _store instead of leaking AttributeError: {payload!r}"
    )
    assert "memory" in text and "unavailable" in text, payload


def test_unavailable_message_names_configured_db_path(tmp_path):
    """When initialize() fails, the error must point recovery at the actual
    configured database, not the hard-coded default path.

    A directory can't be opened as a SQLite file, so pointing ``db_path`` at
    one drives ``MemoryStore.__init__`` to raise — the real failure mode —
    while ``initialize()`` has already captured the resolved path.
    """
    custom_db = str(tmp_path)  # a directory: sqlite can't open it as a DB file
    provider = HolographicMemoryProvider(config={"db_path": custom_db})

    with pytest.raises(Exception):
        provider.initialize(session_id="s")

    # Path was captured before the failing MemoryStore construction.
    assert provider._db_path == custom_db
    assert provider._store is None

    text = _unwrap_tool_error(provider._handle_fact_store({"action": "list"}))
    # Recovery text names the configured file, not the default memory_store.db.
    assert custom_db in text, (
        f"error should point recovery at the configured db_path: {text!r}"
    )


def test_unavailable_message_is_generic_before_initialize():
    """Before initialize() runs there is no resolved path, so the message
    falls back to a generic reference instead of naming a wrong file."""
    provider = HolographicMemoryProvider(config={})
    assert provider._db_path is None

    text = _unwrap_tool_error(provider._handle_fact_store({"action": "list"})).lower()
    assert "db_path" in text, (
        f"generic fallback should reference the db_path setting: {text!r}"
    )

"""Configured Kanban spawn guards are a fail-closed admission boundary."""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb


def test_configured_spawn_guard_wraps_custom_dispatch_spawn(monkeypatch):
    calls = []
    task = object()

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return 4242

    def guard(task, workspace, board, native_spawn):
        assert task is not None
        return native_spawn(task, workspace, board=board)

    monkeypatch.setattr(kb, "_load_configured_spawn_guard", lambda: guard)
    assert kb._spawn_with_guard(task, "/tmp/workspace", "board", native) == 4242
    assert len(calls) == 1


def test_configured_spawn_guard_failure_never_calls_native_spawn(monkeypatch):
    called = []

    def native(*args, **kwargs):
        called.append(True)
        return 4242

    def guard(*args, **kwargs):
        raise RuntimeError("governor unavailable")

    monkeypatch.setattr(kb, "_load_configured_spawn_guard", lambda: guard)
    with pytest.raises(kb.SpawnAdmissionError, match="governor unavailable"):
        kb._spawn_with_guard(object(), "/tmp/workspace", "board", native)
    assert called == []

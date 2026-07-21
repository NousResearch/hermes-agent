"""Tests for gateway/hooks.py — HookRegistry discover_and_load semantics.

Regression coverage for Issue #11902: scalar ``events: agent:start`` in a
HOOK.yaml was iterated character-by-character, registering handlers for
``[':', 'a', 'e', 'g', 'n', 'r', 's', 't']`` instead of ``['agent:start']``.
"""

import os
from pathlib import Path

import pytest


def _make_hook(hooks_dir: Path, name: str, manifest_yaml: str, handler_py: str) -> Path:
    """Create a hook directory with HOOK.yaml and handler.py."""
    hook_dir = hooks_dir / name
    hook_dir.mkdir(parents=True)
    (hook_dir / "HOOK.yaml").write_text(manifest_yaml, encoding="utf-8")
    (hook_dir / "handler.py").write_text(handler_py, encoding="utf-8")
    return hook_dir


def _fresh_registry():
    """Return a fresh HookRegistry with a reloaded gateway.hooks module.

    HOOKS_DIR is computed at import time from HERMES_HOME, so we reload
    after the conftest autouse fixture has set the env var.
    """
    import importlib
    import gateway.hooks as hooks_mod
    importlib.reload(hooks_mod)
    return hooks_mod.HookRegistry()


def test_scalar_events_normalized_to_list(tmp_path):
    """Scalar string events should be treated as a single-item list."""
    hermes_home = Path(os.environ["HERMES_HOME"])
    hooks_dir = hermes_home / "hooks"
    hooks_dir.mkdir()

    _make_hook(
        hooks_dir,
        "scalar_events_hook",
        manifest_yaml="name: scalar_events_hook\nevents: agent:start\n",
        handler_py="def handle(event_type, context): pass\n",
    )

    registry = _fresh_registry()
    registry.discover_and_load()

    assert "agent:start" in registry._handlers
    # Ensure no character-level keys were registered
    for char in "agent:start":
        assert char not in registry._handlers, f"unexpected char key: {char!r}"

    loaded = registry.loaded_hooks
    assert len(loaded) == 1
    assert loaded[0]["events"] == ["agent:start"]


def test_list_events_preserved(tmp_path):
    """Normal list-of-strings events should work unchanged."""
    hermes_home = Path(os.environ["HERMES_HOME"])
    hooks_dir = hermes_home / "hooks"
    hooks_dir.mkdir()

    _make_hook(
        hooks_dir,
        "list_events_hook",
        manifest_yaml="name: list_events_hook\nevents:\n  - gateway:startup\n  - agent:start\n",
        handler_py="def handle(event_type, context): pass\n",
    )

    registry = _fresh_registry()
    registry.discover_and_load()

    assert "gateway:startup" in registry._handlers
    assert "agent:start" in registry._handlers
    loaded = registry.loaded_hooks
    assert len(loaded) == 1
    assert loaded[0]["events"] == ["gateway:startup", "agent:start"]


def test_empty_events_skipped(tmp_path):
    """Hooks with empty or missing events should be skipped."""
    hermes_home = Path(os.environ["HERMES_HOME"])
    hooks_dir = hermes_home / "hooks"
    hooks_dir.mkdir()

    _make_hook(
        hooks_dir,
        "no_events_hook",
        manifest_yaml="name: no_events_hook\n",
        handler_py="def handle(event_type, context): pass\n",
    )

    registry = _fresh_registry()
    registry.discover_and_load()

    assert not registry._handlers
    assert not registry.loaded_hooks

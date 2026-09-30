"""Phase 4.4 loading deadline invariants."""

from __future__ import annotations

import contextvars

import pytest

import plugin_runtime.loading as loading


class _Context:
    def __init__(self) -> None:
        self.abandoned = False

    def _abandon_load(self) -> None:
        self.abandoned = True

    def _tool_override_allowed(self, _tool_name: str) -> bool:
        return False

    def register_skill(self, *args, **kwargs):
        raise AssertionError("not used by deadline tests")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, loading._LOAD_TIMEOUT_SECS),
        (0, 0.0),
        (-1, loading._LOAD_TIMEOUT_SECS),
        (loading._MAX_LOAD_TIMEOUT_SECS + 1, loading._MAX_LOAD_TIMEOUT_SECS),
        ("not-a-number", loading._LOAD_TIMEOUT_SECS),
    ],
)
def test_load_timeout_resolution_preserves_bounds(monkeypatch, raw, expected):
    monkeypatch.setattr(loading, "read_plugin_load_timeout_seconds", lambda: raw)

    assert loading._resolve_plugin_load_timeout() == expected


def test_abandoned_loader_cap_rejects_new_deadline_worker(monkeypatch):
    class AliveThread:
        @staticmethod
        def is_alive() -> bool:
            return True

    monkeypatch.setattr(
        loading,
        "_ABANDONED_LOADERS",
        [AliveThread() for _ in range(loading._MAX_ABANDONED_LOADERS)],
    )

    with pytest.raises(loading.PluginLoadTimeout, match="abandoned plugin loader thread"):
        loading._reserve_abandoned_loader_slot()


def test_deadline_worker_inherits_contextvars(monkeypatch):
    marker = contextvars.ContextVar("phase44_loader_marker", default="missing")
    marker.set("inherited")
    monkeypatch.setattr(loading, "read_plugin_load_timeout_seconds", lambda: 1.0)

    assert loading.run_with_load_deadline(
        "context-fixture",
        _Context(),
        marker.get,
    ) == "inherited"


def test_reentrant_discovery_returns_from_loader_worker(monkeypatch, tmp_path):
    from plugin_runtime.manager import PluginManager
    from plugin_runtime.manifest import PluginManifest

    manager = PluginManager(scope_key=str(tmp_path))
    manager._discovered = True
    monkeypatch.setattr(
        manager,
        "_discover_and_load_inner",
        lambda: pytest.fail("re-entrant loader worker must not start another discovery sweep"),
    )
    monkeypatch.setattr(loading, "read_plugin_load_timeout_seconds", lambda: 1.0)
    context = manager.context_for(PluginManifest(name="fixture", key="fixture", source="user"))

    assert loading.run_with_load_deadline(
        "reentrant-fixture",
        context,
        manager.discover_and_load,
    ) is None

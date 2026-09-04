"""The opportunistic auto-archive sweep must return its registry reference.

``_maybe_auto_archive_for_profile`` borrows a shared writable ``SessionDB`` from
the registry and returns it in a ``finally``. Its whole body is wrapped in
``except Exception`` with a debug log, so a cleanup that raises is invisible:
the sweep still archives, the endpoint still answers, and only the refcount is
wrong. Every eligible sweep then leaks one reference, which pins the generation
open and defeats the physical teardown the registry owns (#102827 / #103118).

Asserting on the archive result or on "the helper did not raise" cannot see
this. These regressions assert the refcount itself, on both the success and the
failure path, and that the last real holder can still tear the handle down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_state_registry as registry
import hermes_cli.web_server_sessions as sessions


@pytest.fixture(autouse=True)
def _reset_throttle():
    """The sweep is throttled per profile for 300s; tests must not inherit it."""
    sessions._last_auto_archive_check.clear()
    yield
    sessions._last_auto_archive_check.clear()


def _arm_sweep(monkeypatch, tmp_path: Path, *, enabled: bool = True) -> Path:
    """Point one sweep at ``tmp_path/state.db`` with a config double."""
    db_path = tmp_path / "state.db"
    config = {
        "sessions": {
            "auto_archive": enabled,
            "auto_archive_days": 3,
            "min_interval_hours": 0,
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **kw: config)
    monkeypatch.setattr(
        sessions, "_open_session_db_for_profile",
        lambda profile, *, read_only: registry.acquire(db_path))
    return db_path


def _refcount_for(db_path: Path) -> int:
    generation = registry._generations.get(Path(db_path).resolve())
    return 0 if generation is None else generation.refcount


class TestAutoArchiveReleasesItsRegistryReference:
    def test_successful_sweep_returns_the_borrowed_reference(self, tmp_path, monkeypatch):
        db_path = _arm_sweep(monkeypatch, tmp_path)
        holder = registry.acquire(db_path)
        try:
            before = _refcount_for(db_path)
            sessions._maybe_auto_archive_for_profile(None)
            assert _refcount_for(db_path) == before, "auto-archive leaked a registry reference"
        finally:
            assert registry.release(holder) is True
        # The independent holder was the last one: teardown must now be reachable.
        assert holder._conn is None

    def test_raising_sweep_still_returns_the_borrowed_reference(self, tmp_path, monkeypatch):
        db_path = _arm_sweep(monkeypatch, tmp_path)
        holder = registry.acquire(db_path)

        def _boom(self, **kwargs):
            raise RuntimeError("archive failed")

        monkeypatch.setattr(type(holder), "maybe_auto_archive", _boom, raising=False)
        try:
            before = _refcount_for(db_path)
            sessions._maybe_auto_archive_for_profile(None)
            assert _refcount_for(db_path) == before, "failed sweep leaked a registry reference"
        finally:
            assert registry.release(holder) is True
        assert holder._conn is None

    def test_repeated_eligible_sweeps_do_not_accumulate_references(self, tmp_path, monkeypatch):
        db_path = _arm_sweep(monkeypatch, tmp_path)
        holder = registry.acquire(db_path)
        try:
            before = _refcount_for(db_path)
            for _ in range(3):
                sessions._last_auto_archive_check.clear()
                sessions._maybe_auto_archive_for_profile(None)
            assert _refcount_for(db_path) == before
        finally:
            assert registry.release(holder) is True
        assert holder._conn is None

    def test_disabled_auto_archive_never_acquires(self, tmp_path, monkeypatch):
        db_path = _arm_sweep(monkeypatch, tmp_path, enabled=False)
        sessions._maybe_auto_archive_for_profile(None)
        assert registry._generations.get(Path(db_path).resolve()) is None

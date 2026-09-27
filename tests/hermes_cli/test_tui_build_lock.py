"""Concurrent TUI launches must never exec a half-published bundle (#121286).

The frontend publisher stages the product in a scratch dir and swaps it in with two
renames, so ``dist/entry.js`` is absent for a moment mid-swap. Every ``hermes --tui``
launch and every dashboard Chat-tab PTY spawn rebuilds before exec'ing that file, and
the dashboard's argv lock only serializes inside ONE dashboard process. A second launch
resolving argv inside the window execs a missing or half-swapped bundle and dies on the
spot with a SyntaxError in the truncated text.

A staleness re-check alone cannot fix this: two launches that both see "stale" still race
each other through the same swap. The build therefore runs under a cross-process lock,
and the staleness check re-runs *after* acquiring so the second process builds nothing.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from hermes_cli import main_tui_launch as mtl


class TestTuiBuildLock:
    @pytest.fixture(autouse=True)
    def _fast_poll(self, monkeypatch):
        """Keep the real 0.2s poll, but shrink the wait budget: these tests release
        promptly, so a long timeout only buys dead time if one ever hangs."""
        monkeypatch.setattr(mtl._TuiBuildLock, "POLL_S", 0.02)

    def test_second_holder_waits_for_the_first(self, tmp_path):
        """The whole point: two launches must not build at the same time."""
        lock_path = tmp_path / "tui.lock"
        first = mtl._TuiBuildLock(lock_path, timeout=5.0)
        second = mtl._TuiBuildLock(lock_path, timeout=5.0)
        assert first.acquire() is True

        outcome: list[bool] = []
        waiter = threading.Thread(target=lambda: outcome.append(second.acquire()))
        waiter.start()
        time.sleep(0.15)
        assert outcome == [], "a second build started while the first held the lock"
        first.release()
        waiter.join(timeout=10.0)
        assert outcome == [True]
        second.release()

    def test_lock_from_a_dead_process_is_taken_over(self, tmp_path):
        """A killed launcher must not wedge every later launch forever."""
        lock_path = tmp_path / "tui.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, b"abandoned\n")
        os.close(fd)  # the owner is gone: no fd held
        stale = time.time() - (mtl._TuiBuildLock.STALE_LOCK_S + 60)
        os.utime(lock_path, (stale, stale))

        lock = mtl._TuiBuildLock(lock_path, timeout=5.0)
        assert lock.acquire() is True
        lock.release()

    def test_a_live_holders_lock_is_never_stolen(self, tmp_path):
        """Age alone must not let a second builder displace a build still running."""
        lock_path = tmp_path / "tui.lock"
        holder = mtl._TuiBuildLock(lock_path, timeout=0.5)
        assert holder.acquire() is True
        stale = time.time() - (mtl._TuiBuildLock.STALE_LOCK_S + 60)
        os.utime(lock_path, (stale, stale))

        intruder = mtl._TuiBuildLock(lock_path, timeout=0.5)
        assert intruder.acquire() is False
        holder.release()

    def test_release_is_idempotent(self, tmp_path):
        lock_path = tmp_path / "tui.lock"
        lock = mtl._TuiBuildLock(lock_path, timeout=1.0)
        assert lock.acquire() is True
        lock.release()
        lock.release()
        assert not lock_path.exists()


class TestBuildIsExclusiveAndSkipsRedundantWork:
    def test_a_waiter_does_not_rebuild_when_the_holder_already_did(self, tmp_path, monkeypatch):
        """Second launch must observe the first launch's bundle, not build over it.

        Modelled with a short lock budget and a lock file already aged past the stale
        threshold, standing in for a peer that has just finished: the waiter must take the
        lock over, find the bundle current, and build nothing.
        """
        tui_dir = tmp_path / "ui-tui"
        (tui_dir / "dist").mkdir(parents=True)
        project_root = tui_dir.parent
        lock_path = project_root / ".hermes-tui-build.lock"

        # A peer left this behind and is gone (no fd held), aged out of the stale window.
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, b"peer\n")
        os.close(fd)
        stale = time.time() - (mtl._TuiBuildLock.STALE_LOCK_S + 60)
        os.utime(lock_path, (stale, stale))
        monkeypatch.setattr(mtl._TuiBuildLock, "POLL_S", 0.02)
        monkeypatch.setattr(mtl._TuiBuildLock, "DEFAULT_TIMEOUT_S", 5.0)

        calls: list[str] = []
        import hermes_cli.source_build as sb

        monkeypatch.setattr(sb, "source_product_current", lambda *a, **k: True)
        monkeypatch.setattr(sb, "build_source_tui", lambda *a, **k: calls.append("built"))

        mtl._build_tui_exclusive(tui_dir, project_root, {})

        assert calls == [], "rebuilt even though the bundle was already current"

    def test_a_stale_bundle_is_built_when_nobody_else_holds_the_lock(self, tmp_path, monkeypatch):
        tui_dir = tmp_path / "ui-tui"
        (tui_dir / "dist").mkdir(parents=True)
        project_root = tui_dir.parent

        calls: list[str] = []
        import hermes_cli.source_build as sb

        monkeypatch.setattr(sb, "source_product_current", lambda *a, **k: False)
        monkeypatch.setattr(sb, "build_source_tui", lambda *a, **k: calls.append("built"))

        mtl._build_tui_exclusive(tui_dir, project_root, {})

        assert calls == ["built"]
        # The lock must not outlive the build, or every later launch would wait on it.
        assert not (project_root / ".hermes-tui-build.lock").exists()

    def test_the_lock_is_released_even_when_the_build_raises(self, tmp_path, monkeypatch):
        tui_dir = tmp_path / "ui-tui"
        (tui_dir / "dist").mkdir(parents=True)
        project_root = tui_dir.parent
        import hermes_cli.source_build as sb

        monkeypatch.setattr(sb, "source_product_current", lambda *a, **k: False)

        def boom(*_a, **_k):
            raise RuntimeError("esbuild exploded")

        monkeypatch.setattr(sb, "build_source_tui", boom)

        with pytest.raises(RuntimeError):
            mtl._build_tui_exclusive(tui_dir, project_root, {})
        assert not (project_root / ".hermes-tui-build.lock").exists()

"""Web launch keeps freshness/serialization but never masks a failed build."""
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.main_web_build import (
    _build_web_ui, _web_ui_build_needed, _compute_web_ui_content_hash,
    _web_ui_stamp_path, _write_web_ui_build_stamp,
)
from tests.hermes_cli.test_source_build import source_checkout, source_products, _events  # noqa: F401


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path, monkeypatch):
    """Keep web-build-stamp writes inside the test's tmp dir, never the real home."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "_hermes_home"))


def _touch(path: Path, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    if offset:
        t = time.time() + offset
        os.utime(path, (t, t))


def _make_web_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Return (web_dir, dist_dir) matching real repo layout."""
    web_dir = tmp_path / "web"
    web_dir.mkdir(parents=True)
    (web_dir / "package.json").touch()
    dist_dir = tmp_path / "hermes_cli" / "web_dist"
    return web_dir, dist_dir


class TestWebUIBuildNeeded:
    """Content-hash staleness — replaces the old mtime comparison.

    The dashboard build hashes the web source tree (like the desktop build)
    instead of comparing mtimes, so git operations that rewrite mtimes
    without changing content no longer fool the freshness check.
    """

    @staticmethod
    def _root(web_dir: Path) -> Path:
        return web_dir.parent.parent if web_dir.parent.name == "apps" else web_dir.parent

    def _stamp_current(self, web_dir: Path) -> None:
        """Record a stamp matching web_dir's current source content."""
        _write_web_ui_build_stamp(self._root(web_dir), web_dir)


    def test_mtime_only_change_is_not_stale(self, tmp_path):
        """The whole point: bumping mtimes without changing bytes (what
        ``git pull`` / ``hermes update`` do) must NOT report stale."""
        web_dir, dist_dir = _make_web_dir(tmp_path)
        src = web_dir / "src" / "App.tsx"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("export const A = 1\n")
        (dist_dir / ".vite").mkdir(parents=True, exist_ok=True)
        (dist_dir / ".vite" / "manifest.json").write_text("{}")
        self._stamp_current(web_dir)
        assert _web_ui_build_needed(web_dir) is False
        future = time.time() + 10_000
        os.utime(src, (future, future))
        os.utime(web_dir / "package.json", (future, future))
        assert _web_ui_build_needed(web_dir) is False


    def test_content_hash_is_deterministic(self, tmp_path):
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "src").mkdir(parents=True, exist_ok=True)
        (web_dir / "src" / "App.tsx").write_text("export const A = 1\n")
        root = self._root(web_dir)
        h1 = _compute_web_ui_content_hash(root, web_dir)
        h2 = _compute_web_ui_content_hash(root, web_dir)
        assert h1 == h2
        assert len(h1) == 64

    def test_write_stamp_creates_file_with_hash(self, tmp_path):
        import json as _json
        web_dir, _ = _make_web_dir(tmp_path)
        (web_dir / "src").mkdir(parents=True, exist_ok=True)
        (web_dir / "src" / "App.tsx").write_text("export const A = 1\n")
        self._stamp_current(web_dir)
        stamp = _web_ui_stamp_path()
        assert stamp.is_file()
        data = _json.loads(stamp.read_text())
        assert data["contentHash"] == _compute_web_ui_content_hash(self._root(web_dir), web_dir)


@pytest.mark.platforms("linux")
class TestBuildWebUIFlock:
    def test_contended_lock_without_dist_waits_then_skips_fresh_build(self, tmp_path):
        """First-ever build race: the waiter blocks, and once it acquires the
        lock the callee's own staleness check (running under the lock) sees
        the winner's output and skips a duplicate build."""
        import fcntl
        import threading
        from hermes_cli.main_web_build import _build_web_ui as build

        web_dir, dist_dir = _make_web_dir(tmp_path)
        # No dist yet — contender must take the blocking-wait path.
        lock_path = tmp_path / ".web_ui_build.lock"
        holder = open(lock_path, "a")
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)

        def release_after_building():
            # Simulate the winning process finishing its build.
            _touch(dist_dir / ".vite" / "manifest.json")
            _write_web_ui_build_stamp(tmp_path, web_dir)
            holder.close()  # releases the flock

        t = threading.Timer(0.2, release_after_building)
        t.start()
        try:
            with patch("hermes_cli.source_build.source_build_env", side_effect=AssertionError("fresh build must skip preparation")) as mock_run:
                result = build(web_dir)
        finally:
            t.join()

        assert result is True
        mock_run.assert_not_called()  # fresh after the wait -> no rebuild


@pytest.mark.platforms("posix")
def test_web_build_prepares_once_and_skips_a_current_product(source_products):
    root, acquired = source_products
    assert _build_web_ui(root / "web", fatal=True)
    assert [event["step"] for event in _events(root)] == ["deps", "icons", "web"]
    assert acquired == ["npm"]
    assert not _web_ui_build_needed(root / "web")
    assert _build_web_ui(root / "web", fatal=True)
    assert acquired == ["npm"]
    assert len(_events(root)) == 3


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("fatal", [False, True])
def test_web_failure_is_not_success_even_with_an_old_dist(source_products, fatal):
    root, acquired = source_products
    dist = root / "hermes_cli/web_dist/index.html"
    dist.parent.mkdir(parents=True)
    dist.write_text("old product")
    (root / "fail-web").touch()
    assert not _build_web_ui(root / "web", fatal=fatal)
    assert acquired == ["npm"]
    assert [event["step"] for event in _events(root)] == ["deps", "icons", "web"]
    assert dist.read_text() == "old product"
    assert not _web_ui_stamp_path().exists()


@pytest.mark.platforms("posix")
def test_failed_preparation_never_runs_web_compilation(source_products):
    root, acquired = source_products
    (root / "package-lock.json").write_text("not json")
    assert not _build_web_ui(root / "web", fatal=True)
    assert acquired == ["npm"]
    assert _events(root) == []
    assert not _web_ui_stamp_path().exists()


@pytest.mark.platforms("linux")
def test_web_rebuild_reuses_the_existing_desktop_union(source_products):
    from hermes_cli.source_build import build_update_products

    root, acquired = source_products
    build_update_products(root, desktop=True)
    before = _events(root)
    (root / "web/changed.ts").write_text("changed web source")
    assert _build_web_ui(root / "web", fatal=True)
    assert _events(root) == [*before, {"step": "icons"}, {"step": "web"}]
    assert acquired == ["npm", "npm"]
    assert (root / "node_modules/apps-desktop").exists()


@pytest.mark.platforms("linux")
def test_contended_stale_dist_waits_for_the_lock_holder(tmp_path):
    import fcntl
    import threading

    web, dist = _make_web_dir(tmp_path)
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("stale")
    holder = open(tmp_path / ".web_ui_build.lock", "a")
    fcntl.flock(holder, fcntl.LOCK_EX)

    def finish_build():
        (dist / "index.html").write_text("winner")
        _write_web_ui_build_stamp(tmp_path, web)
        holder.close()

    worker = threading.Timer(2, finish_build)
    worker.start()
    try:
        assert _build_web_ui(web, fatal=True)
        at_return = (dist / "index.html").read_text()
    finally:
        worker.join()
    assert at_return == "winner", "a contended stale index must not count as success"


@pytest.mark.platforms("posix")
def test_lock_open_failure_does_not_start_an_unprotected_build(source_products):
    root, acquired = source_products
    (root / ".web_ui_build.lock").mkdir()
    assert not _build_web_ui(root / "web", fatal=True)
    assert acquired == []
    assert _events(root) == []

"""Web launch keeps freshness/serialization but never masks a failed build."""
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.main_web_build import _build_web_ui, _web_ui_build_needed
from tests.hermes_cli.test_source_build import stamp_product, copy_freshness_scripts
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
    copy_freshness_scripts(tmp_path)
    web_dir = tmp_path / "web"
    web_dir.mkdir(parents=True)
    (web_dir / "package.json").touch()
    dist_dir = tmp_path / "hermes_cli" / "web_dist"
    return web_dir, dist_dir



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
    assert not (root / "hermes_cli/web_dist/hermes-build.json").exists()


@pytest.mark.platforms("posix")
def test_failed_preparation_never_runs_web_compilation(source_products):
    root, acquired = source_products
    (root / "package-lock.json").write_text("not json")
    assert not _build_web_ui(root / "web", fatal=True)
    assert acquired == ["npm"]
    assert _events(root) == []
    assert not (root / "hermes_cli/web_dist/hermes-build.json").exists()


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
@pytest.mark.parametrize('existing', [False, True])
def test_contended_build_waits_and_rechecks_winner(tmp_path, monkeypatch, existing):
    import fcntl
    import threading

    web, dist = _make_web_dir(tmp_path)
    if existing:
        _touch(dist / 'index.html')
    holder = open(tmp_path / '.web_ui_build.lock', 'a', encoding='utf-8')
    fcntl.flock(holder, fcntl.LOCK_EX)
    real_flock = fcntl.flock
    contended = threading.Event()
    def flock(fd, operation):
        contended.set()
        return real_flock(fd, operation)
    monkeypatch.setattr(fcntl, 'flock', flock)
    def finish():
        try:
            assert contended.wait(10), 'waiter never attempted the lock'
            _touch(dist / 'index.html')
            (dist / 'index.html').write_text('winner', encoding='utf-8')
            stamp_product(tmp_path, 'web', dist)
        finally:
            holder.close()
    worker = threading.Thread(target=finish)
    worker.start()
    try:
        with patch('hermes_cli.source_build.source_build_env', side_effect=AssertionError('duplicate preparation')):
            assert _build_web_ui(web, fatal=True)
        assert (dist / 'index.html').read_text(encoding='utf-8') == 'winner'
    finally:
        worker.join(timeout=15)
        holder.close()
    assert not worker.is_alive()


@pytest.mark.platforms("posix")
def test_lock_open_failure_does_not_start_an_unprotected_build(source_products):
    root, acquired = source_products
    (root / ".web_ui_build.lock").mkdir()
    assert not _build_web_ui(root / "web", fatal=True)
    assert acquired == []
    assert _events(root) == []

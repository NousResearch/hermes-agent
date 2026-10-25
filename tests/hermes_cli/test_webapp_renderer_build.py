"""The Webapp renderer (``dist-webapp``): verified reuse, one build at a time, rebuilt only when stale.

``hermes_cli/webapp.py`` is what a Webapp launcher calls before handing the browser a link; these
cases come from #93508's ``test_webapp_command.py`` (its renderer-build half).
"""
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from hermes_cli import webapp


def _workspace_tree(root: Path) -> None:
    (root / "apps" / "desktop").mkdir(parents=True)
    (root / "apps" / "shared").mkdir(parents=True)
    (root / "web").mkdir()
    (root / "ui-tui").mkdir()
    (root / "package.json").write_text('{"private":true}', encoding="utf-8")
    (root / "package-lock.json").write_text("locked\n", encoding="utf-8")
    for path in (
        root / "apps" / "desktop" / "package.json",
        root / "apps" / "shared" / "package.json",
        root / "web" / "package.json",
        root / "ui-tui" / "package.json",
    ):
        path.write_text("{}", encoding="utf-8")


def test_skip_build_requires_the_separate_webapp_bundle(tmp_path: Path):
    _workspace_tree(tmp_path)
    (tmp_path / "apps" / "desktop" / "dist").mkdir()
    (tmp_path / "apps" / "desktop" / "dist" / "index.html").write_text(
        "native", encoding="utf-8"
    )

    with pytest.raises(webapp.WebappBuildError, match="dist-webapp"):
        webapp.prepare_webapp_renderer(tmp_path, skip_build=True)


def test_skip_build_writes_nothing_into_the_install_tree(tmp_path: Path):
    """A read-only install can reuse a prebuilt renderer only if --skip-build
    never writes the install tree, build lock included; a missing renderer is
    still refused by its path."""
    _workspace_tree(tmp_path)
    dist = webapp.webapp_dist_dir(tmp_path)
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")

    install_tree = sorted(tmp_path.rglob("*"))
    assert webapp.prepare_webapp_renderer(tmp_path, skip_build=True) == dist
    assert sorted(tmp_path.rglob("*")) == install_tree

    (dist / "index.html").unlink()
    install_tree = sorted(tmp_path.rglob("*"))
    with pytest.raises(webapp.WebappBuildError, match="no Webapp renderer exists"):
        webapp.prepare_webapp_renderer(tmp_path, skip_build=True)
    assert sorted(tmp_path.rglob("*")) == install_tree


def _assert_build_lock_excludes_second_open(tmp_path: Path):
    from pm.filesystem import lock_fd, unlock_fd

    lock_path = tmp_path / "webapp.lock"

    with webapp._exclusive_build_lock(lock_path):
        with lock_path.open("a+b") as contender:
            assert lock_fd(contender.fileno(), wait=False) is False

    with lock_path.open("a+b") as contender:
        assert lock_fd(contender.fileno(), wait=False) is True
        unlock_fd(contender.fileno())
        # Released while its handle stays open, the lock is free for the next open.
        with lock_path.open("a+b") as successor:
            assert lock_fd(successor.fileno(), wait=False) is True
            unlock_fd(successor.fileno())


def test_webapp_build_lock_excludes_a_second_open(tmp_path: Path):
    _assert_build_lock_excludes_second_open(tmp_path)


@pytest.mark.platforms("windows")
def test_webapp_build_lock_excludes_a_second_open_on_windows(tmp_path: Path):
    _assert_build_lock_excludes_second_open(tmp_path)


def test_launch_rebuilds_only_stale_or_forced_renderers(tmp_path: Path, monkeypatch):
    from hermes_cli import source_build

    _workspace_tree(tmp_path)
    receipt_current = True
    builds = []
    monkeypatch.setattr(source_build, "source_product_current", lambda *_a: receipt_current)
    monkeypatch.setattr(source_build, "source_build_env", lambda *, explicit=False: {})
    monkeypatch.setattr(source_build, "build_source_webapp", lambda _root, *, env, explicit: builds.append(explicit))

    dist = webapp.prepare_webapp_renderer(tmp_path)
    assert dist == tmp_path / "apps" / "desktop" / "dist-webapp"
    assert builds == []
    webapp.prepare_webapp_renderer(tmp_path, force=True, explicit=True)
    receipt_current = False
    webapp.prepare_webapp_renderer(tmp_path)
    assert builds == [True, False]


@pytest.mark.parametrize("failure", [
    PermissionError("workspace node_modules is locked"),
    subprocess.CalledProcessError(1, ["node", "build-webapp.mjs"]),
    RuntimeError("npm is unavailable and lazy installs are disabled"),
])
def test_renderer_build_failures_are_reported_not_raised_raw(tmp_path: Path, monkeypatch, failure):
    from hermes_cli import source_build

    _workspace_tree(tmp_path)

    def fail(*_a, **_k):
        raise failure

    monkeypatch.setattr(source_build, "source_product_current", lambda *_a: False)
    monkeypatch.setattr(source_build, "source_build_env", lambda *, explicit=False: {})
    monkeypatch.setattr(source_build, "build_source_webapp", fail)

    with pytest.raises(webapp.WebappBuildError, match="build failed"):
        webapp.prepare_webapp_renderer(tmp_path)


def test_a_build_or_a_current_bundle_clears_the_out_of_date_mark(tmp_path: Path, monkeypatch):
    """``hermes update`` marks the bundle out of date when it cannot rebuild it, and the gateway
    reports the Webapp unavailable while the mark is there. The next build that succeeds (``hermes
    webapp``, Web Access Start or the next update), or that finds the bundle current, clears it.
    Reusing the bundle with ``--skip-build`` writes nothing and leaves it marked."""
    from gateway.webapp_listener import bundle_missing_reason
    from hermes_cli import source_build

    _workspace_tree(tmp_path)
    dist = webapp.webapp_dist_dir(tmp_path)
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html>", encoding="utf-8")
    receipt_current = False
    monkeypatch.setattr(source_build, "source_product_current", lambda *_a: receipt_current)
    monkeypatch.setattr(source_build, "source_build_env", lambda *, explicit=False: {})
    monkeypatch.setattr(source_build, "build_source_webapp", lambda _root, *, env, explicit: None)

    assert webapp.mark_webapp_stale(tmp_path) is True
    assert bundle_missing_reason(dist) is not None
    webapp.prepare_webapp_renderer(tmp_path, skip_build=True)
    assert bundle_missing_reason(dist) is not None

    webapp.prepare_webapp_renderer(tmp_path)
    assert bundle_missing_reason(dist) is None

    assert webapp.mark_webapp_stale(tmp_path) is True
    receipt_current = True
    webapp.prepare_webapp_renderer(tmp_path)
    assert bundle_missing_reason(dist) is None


def test_no_bundle_is_marked_out_of_date_before_one_is_built(tmp_path: Path):
    """A failed rebuild with no bundle built leaves nothing to mark: the gateway already reports the
    Webapp as not built, and the build output folder is left for the builder to create."""
    _workspace_tree(tmp_path)
    install_tree = sorted(tmp_path.rglob("*"))

    assert webapp.mark_webapp_stale(tmp_path) is False
    assert sorted(tmp_path.rglob("*")) == install_tree


def test_dashboard_and_webapp_builds_share_the_workspace_lock(tmp_path: Path, monkeypatch):
    """While the Dashboard builder holds its lock, the Webapp build lock on the same checkout is taken."""
    from pm.filesystem import lock_fd
    from hermes_cli import main_web_build

    _workspace_tree(tmp_path)
    webapp_lock_free = []

    def build_while_locked(_web_dir, *, fatal=False):
        with (tmp_path / webapp._LOCK_NAME).open("ab") as contender:
            webapp_lock_free.append(lock_fd(contender.fileno(), wait=False))
        return True

    monkeypatch.setattr(main_web_build, "_do_build_web_ui", build_while_locked)
    assert main_web_build._build_web_ui(tmp_path / "web") is True
    assert webapp_lock_free == [False]

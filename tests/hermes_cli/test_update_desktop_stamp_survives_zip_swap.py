"""A lost release tree must not strand an installed Desktop unbuilt.

Reproduction (2026-08-20 incident, Windows): run 1 of ``hermes update``
failed on the git path (stash conflict) and fell back to the ZIP path,
whose two-phase replace swapped the ``apps/desktop`` tree — deleting the
git-ignored ``release/win-unpacked`` artifact — and then died before the
desktop rebuild step. Run 2 (a fresh process) snapshotted
``_desktop_app_present() == False``, so the rebuild was silently skipped
and a previously working Desktop stayed unbuilt.

The persistent build stamp under $HERMES_HOME survives the swap: it is
written only by successful desktop builds and removed by ``gui
uninstall``. ``_desktop_rebuild_warranted`` consults it so the retry
rebuilds instead of skipping.
"""

import pytest

from hermes_cli.update_cmd import (
    _desktop_app_present,
    _desktop_rebuild_warranted,
    _rebuild_desktop_after_update,
)


class _Result:
    def __init__(self, returncode: int, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


@pytest.fixture()
def stamp_env(tmp_path, monkeypatch):
    """Desktop dir without any built artifact + a faked CLI main module.

    Mirrors the post-failure state of the incident: package.json exists
    (the source tree is intact), but ``release/`` and ``dist/`` are gone.
    """
    desktop_dir = tmp_path / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}", encoding="utf-8")

    class _FakeMain:
        PROJECT_ROOT = tmp_path

        @staticmethod
        def _desktop_packaged_executable(_desktop_dir):
            return None

        @staticmethod
        def _desktop_dist_exists(_desktop_dir):
            return False

        @staticmethod
        def _desktop_stamp_path():
            return tmp_path / "hermes-home" / "desktop-build-stamp.json"

        @staticmethod
        def _resolve_node_runtime_npm():
            return "/fake/npm"

        @staticmethod
        def _desktop_build_needed(*_a, **_kw):
            return True

        @staticmethod
        def _run_logged_subprocess(cmd, cwd=None, env=None):
            return _Result(0)

    monkeypatch.setattr(update_cmd_module(), "_m", lambda: _FakeMain)
    return desktop_dir, tmp_path / "hermes-home"


def update_cmd_module():
    from hermes_cli import update_cmd

    return update_cmd


def test_stamp_alone_warrants_rebuild(stamp_env):
    """The incident: no artifact on disk, but the stamp proves Desktop existed."""
    desktop_dir, hermes_home = stamp_env
    assert _desktop_app_present(desktop_dir) is False

    hermes_home.mkdir(parents=True)
    (hermes_home / "desktop-build-stamp.json").write_text("{}", encoding="utf-8")

    assert _desktop_rebuild_warranted(desktop_dir) is True


def test_no_stamp_and_no_artifact_skips_rebuild(stamp_env):
    """Users who never used Desktop pay nothing for an Electron build."""
    desktop_dir, hermes_home = stamp_env
    assert not hermes_home.exists()
    assert _desktop_rebuild_warranted(desktop_dir) is False


def test_artifact_on_disk_warrants_rebuild_without_stamp(stamp_env):
    """Pre-existing behavior preserved when the artifact is present."""
    desktop_dir, _hermes_home = stamp_env
    import unittest.mock as mock

    from hermes_cli import update_cmd

    main_cls = update_cmd._m()
    # Simulate a dist build existing.
    with mock.patch.object(
        main_cls, "_desktop_dist_exists", staticmethod(lambda _d: True)
    ):
        assert _desktop_rebuild_warranted(desktop_dir) is True


def test_stamp_exception_falls_back_to_false(stamp_env):
    """Stamp introspection must never break the update path."""
    desktop_dir, _hermes_home = stamp_env
    import unittest.mock as mock

    from hermes_cli import update_cmd

    def boom():
        raise RuntimeError("fs gone")

    with mock.patch.object(update_cmd._m(), "_desktop_stamp_path", staticmethod(boom)):
        assert _desktop_rebuild_warranted(desktop_dir) is False


def test_retry_after_lost_release_tree_runs_the_build(stamp_env, capsys, monkeypatch):
    """End-to-end shape of run 2: rebuild proceeds despite no artifacts."""
    desktop_dir, hermes_home = stamp_env
    hermes_home.mkdir(parents=True)
    (hermes_home / "desktop-build-stamp.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_constants.with_hermes_node_path", lambda: {}, raising=False
    )
    monkeypatch.setattr(
        "hermes_constants.display_hermes_home",
        lambda: str(hermes_home),
        raising=False,
    )

    assert (
        _rebuild_desktop_after_update(
            desktop_dir, had_desktop_app_before_update=False
        )
        is True
    )
    out = capsys.readouterr().out
    assert "Checking if desktop app needs rebuilding" in out

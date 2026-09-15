"""A packaged Desktop UI must not ride along behind a "current" build stamp.

#106670: the desktop build stamp records only the SOURCE content hash. It never says
WHICH artifact that build produced, so a stamp can be perfectly "current" while
``resources/app.asar`` is weeks old (an interrupted promote that kept the previous app, a
pack that failed after the stamp landed, a replaced/restored artifact). The freshness
predicate then answers "up to date", ``hermes update`` prints ``✓ Desktop app up to
date``, and the stale renderer keeps crashing — exactly the clarify/choice-card crash
#87857 fixed that this bug kept live.

These tests pin the two behaviours the fix adds, without building Electron:

* ``_desktop_build_needed`` also requires the packaged artifact to be the one the stamp
  describes (stale/missing ``app.asar`` ⇒ rebuild).
* the update path SCHEDULES that rebuild instead of printing "up to date", and says so
  loudly when it cannot rebuild at all (no npm).

``apps/desktop/release/`` is git-ignored, so the packaged artifact is outside the content
hash by construction: a stale ``app.asar`` cannot be caught by the hash alone.
"""

import json
import os
import sys

import pytest

from hermes_cli import main_desktop, update_cmd
from hermes_cli.main_desktop import (
    _desktop_artifact_id,
    _desktop_build_needed,
    _write_desktop_build_stamp,
)
from hermes_cli.update_cmd import _rebuild_desktop_after_update


class _Result:
    def __init__(self, returncode: int, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


def _packaged_app(root):
    """A minimal git-install Desktop: source tree + packed exe + app.asar.

    Returns ``(desktop_dir, asar)``. The platform exe path is the real one
    ``_desktop_packaged_executable`` looks for, so the real helper chain runs.
    """
    desktop = root / "apps" / "desktop"
    (desktop / "src").mkdir(parents=True)
    (desktop / "package.json").write_text('{"name": "hermes-desktop"}', encoding="utf-8")
    (desktop / "src" / "index.ts").write_text("export const x = 1\n", encoding="utf-8")
    # Mirror the real repo: the packaged tree is not part of the source content hash.
    (root / ".gitignore").write_text("apps/desktop/release/\n", encoding="utf-8")

    if sys.platform == "darwin":
        exe = desktop / "release" / "mac" / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
        resources = exe.parent.parent / "Resources"
    elif sys.platform == "win32":
        exe = desktop / "release" / "win-unpacked" / "Hermes.exe"
        resources = exe.parent / "resources"
    else:
        exe = desktop / "release" / "linux-unpacked" / "hermes"
        resources = exe.parent / "resources"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    resources.mkdir(parents=True)
    asar = resources / "app.asar"
    asar.write_bytes(b"asar-generation-A")
    return desktop, asar


def _age_by_days(path, days):
    """Backdate *path* (an app.asar copied/left from an older build)."""
    old = path.stat().st_mtime - days * 86400
    os.utime(path, (old, old))


@pytest.fixture()
def packaged(tmp_path, monkeypatch):
    desktop, asar = _packaged_app(tmp_path)
    stamp = tmp_path / "hermes-home" / "desktop-build-stamp.json"
    monkeypatch.setattr(main_desktop, "_desktop_stamp_path", lambda: stamp)
    return tmp_path, desktop, asar, stamp


def _fake_update_main(root, *, spawned, npm="/fake/npm"):
    """The frozen updater surface ``_rebuild_desktop_after_update`` resolves via ``_m()``,
    wired to the REAL freshness predicate and the REAL artifact lookup."""

    class _FakeMain:
        PROJECT_ROOT = root
        _desktop_build_needed = staticmethod(_desktop_build_needed)
        _desktop_packaged_executable = staticmethod(main_desktop._desktop_packaged_executable)
        _desktop_dist_exists = staticmethod(main_desktop._desktop_dist_exists)
        _resolve_node_runtime_npm = staticmethod(lambda: npm)
        _run_logged_subprocess = staticmethod(
            lambda cmd, cwd=None, env=None: spawned.append(cmd) or _Result(0)
        )

    return _FakeMain


# ─── the predicate: stamp is not enough, the artifact must match it ────────────────────────


def test_a_build_binds_its_stamp_to_the_artifact_it_produced(packaged):
    root, desktop, _asar, stamp = packaged
    _write_desktop_build_stamp(root, source_mode=False)
    recorded = json.loads(stamp.read_text(encoding="utf-8"))
    assert recorded["sourceMode"] is False
    assert recorded["artifact"] == _desktop_artifact_id(desktop)


def test_freshly_packed_artifact_is_current(packaged, capsys):
    root, desktop, _asar, _stamp = packaged
    _write_desktop_build_stamp(root, source_mode=False)
    assert _desktop_build_needed(desktop, root, source_mode=False) is False
    assert "stale" not in capsys.readouterr().out.lower()


def test_three_week_old_asar_behind_a_current_stamp_is_stale(packaged, capsys):
    """The reported shape: source stamp matches, packaged UI is 3 weeks old."""
    root, desktop, asar, _stamp = packaged
    _write_desktop_build_stamp(root, source_mode=False)
    _age_by_days(asar, 21)
    assert _desktop_build_needed(desktop, root, source_mode=False) is True
    assert "stale" in capsys.readouterr().out.lower()


def test_an_artifact_that_is_not_the_recorded_one_is_stale(packaged):
    """A pack that never landed (or was replaced): different bytes, same source stamp."""
    root, desktop, asar, _stamp = packaged
    _write_desktop_build_stamp(root, source_mode=False)
    asar.write_bytes(b"asar-generation-B-from-another-build")
    assert _desktop_build_needed(desktop, root, source_mode=False) is True


def test_missing_packaged_asar_is_stale(packaged):
    root, desktop, asar, _stamp = packaged
    _write_desktop_build_stamp(root, source_mode=False)
    asar.unlink()
    assert _desktop_build_needed(desktop, root, source_mode=False) is True


def test_legacy_stamp_without_an_artifact_binding_is_rebuilt(packaged):
    """Stamps written before this fix prove nothing about the artifact: rebuild once."""
    root, desktop, _asar, stamp = packaged
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(
        json.dumps(
            {
                "contentHash": main_desktop._compute_desktop_content_hash(root),
                "sourceMode": False,
            }
        ),
        encoding="utf-8",
    )
    assert _desktop_build_needed(desktop, root, source_mode=False) is True


def test_an_unchanged_source_stamp_alone_does_not_catch_a_stale_artifact(packaged):
    """Guard on the guard: the content hash cannot see release/ (git-ignored), so the
    artifact binding is the only thing that can fail this case."""
    root, desktop, asar, stamp = packaged
    _write_desktop_build_stamp(root, source_mode=False)
    before = json.loads(stamp.read_text(encoding="utf-8"))["contentHash"]
    _age_by_days(asar, 21)
    assert main_desktop._compute_desktop_content_hash(root) == before


# ─── the update path: schedule the repair, never pass silently ─────────────────────────────


def test_update_schedules_a_rebuild_for_a_stale_packaged_ui(packaged, monkeypatch, capsys):
    root, desktop, asar, _stamp = packaged
    _write_desktop_build_stamp(root, source_mode=False)
    _age_by_days(asar, 21)
    spawned = []
    monkeypatch.setattr(update_cmd, "_m", lambda: _fake_update_main(root, spawned=spawned))
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", lambda: {}, raising=False)

    assert _rebuild_desktop_after_update(desktop, had_desktop_app_before_update=True) is True

    assert spawned, "a stale packaged UI must schedule `desktop --build-only`"
    assert "desktop" in spawned[0] and "--build-only" in spawned[0]
    assert "stale" in capsys.readouterr().out.lower()


def test_update_does_not_spawn_a_build_for_a_matching_artifact(packaged, monkeypatch, capsys):
    root, desktop, _asar, _stamp = packaged
    _write_desktop_build_stamp(root, source_mode=False)
    spawned = []
    monkeypatch.setattr(update_cmd, "_m", lambda: _fake_update_main(root, spawned=spawned))
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", lambda: {}, raising=False)

    assert _rebuild_desktop_after_update(desktop, had_desktop_app_before_update=True) is True
    assert spawned == []
    assert "up to date" in capsys.readouterr().out


def test_installed_desktop_that_cannot_be_rebuilt_warns_instead_of_passing_silently(
    packaged, monkeypatch, capsys
):
    root, desktop, _asar, _stamp = packaged
    spawned = []
    monkeypatch.setattr(update_cmd, "_m", lambda: _fake_update_main(root, spawned=spawned, npm=None))

    assert _rebuild_desktop_after_update(desktop, had_desktop_app_before_update=True) is True
    out = capsys.readouterr().out
    assert spawned == []
    assert "npm" in out and "stale" in out.lower()


def test_a_desktop_that_was_never_installed_stays_silent(tmp_path, monkeypatch, capsys):
    """People who never used Desktop must not pay — neither CPU nor noise."""
    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        update_cmd, "_m", lambda: _fake_update_main(tmp_path, spawned=[], npm=None)
    )
    assert _rebuild_desktop_after_update(desktop, had_desktop_app_before_update=False) is True
    assert "npm" not in capsys.readouterr().out

"""Installed desktop install-stamp vs HEAD must trigger a rebuild (#107542).

After a failed stage-and-swap, $HERMES_HOME/desktop-build-stamp.json can still
match the source tree while the packaged resources/install-stamp.json is weeks
behind HEAD. ``_desktop_build_needed`` must treat that packaged stamp as the
installed truth — and fail-open when the stamp is missing, fallback, or unreadable.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hermes_cli import main_desktop

STALE_COMMIT = "eca85e81deadbeefcafe00000000000000000000"
HEAD_COMMIT = "67764dc086cafebabe1111111111111111111111"
PREFIX_STAMP = "67764dc086"


def _mock_git_head(monkeypatch, sha: str) -> None:
    real_run = main_desktop.subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if (
            isinstance(cmd, (list, tuple))
            and cmd
            and cmd[0] == "git"
            and "rev-parse" in cmd
        ):
            return subprocess.CompletedProcess(list(cmd), 0, stdout=f"{sha}\n", stderr="")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(main_desktop.subprocess, "run", fake_run)


def _resources_dir_for_exe(exe: Path) -> Path:
    """Same derivation ``_desktop_build_needed`` / ``_renderer_bundle_dir`` use."""
    return main_desktop._packaged_resources_dir(exe)


def _packaged_setup(tmp_path, monkeypatch, *, stamp: dict | None, head: str = HEAD_COMMIT):
    desktop_dir = tmp_path / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    exe = tmp_path / "packaged" / "Hermes.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("fake-exe", encoding="utf-8")
    monkeypatch.setattr(main_desktop, "_desktop_packaged_executable", lambda _d: exe)
    monkeypatch.setattr(main_desktop, "_stamp_is_current", lambda *_a, **_k: True)
    _mock_git_head(monkeypatch, head)

    resources = _resources_dir_for_exe(exe)
    resources.mkdir(parents=True, exist_ok=True)
    if stamp is not None:
        (resources / "install-stamp.json").write_text(json.dumps(stamp), encoding="utf-8")
    return desktop_dir, tmp_path


def test_lagging_installed_stamp_needs_rebuild_even_when_content_stamp_matches(tmp_path, monkeypatch):
    desktop_dir, project_root = _packaged_setup(
        tmp_path,
        monkeypatch,
        stamp={"schemaVersion": 1, "commit": STALE_COMMIT},
    )

    assert main_desktop._desktop_build_needed(desktop_dir, project_root, source_mode=False) is True


def test_missing_install_stamp_fail_open(tmp_path, monkeypatch):
    desktop_dir, project_root = _packaged_setup(tmp_path, monkeypatch, stamp=None)

    assert main_desktop._desktop_build_needed(desktop_dir, project_root, source_mode=False) is False


def test_fallback_commit_fail_open(tmp_path, monkeypatch):
    desktop_dir, project_root = _packaged_setup(
        tmp_path,
        monkeypatch,
        stamp={"schemaVersion": 1, "commit": "0000000", "source": "fallback"},
    )

    assert main_desktop._desktop_build_needed(desktop_dir, project_root, source_mode=False) is False


def test_prefix_match_is_not_stale(tmp_path, monkeypatch):
    desktop_dir, project_root = _packaged_setup(
        tmp_path,
        monkeypatch,
        stamp={"schemaVersion": 1, "commit": PREFIX_STAMP},
        head=HEAD_COMMIT,
    )

    assert main_desktop._desktop_build_needed(desktop_dir, project_root, source_mode=False) is False


def test_source_mode_ignores_lagging_packaged_stamp(tmp_path, monkeypatch):
    desktop_dir, project_root = _packaged_setup(
        tmp_path,
        monkeypatch,
        stamp={"schemaVersion": 1, "commit": STALE_COMMIT},
    )
    dist = desktop_dir / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")

    assert main_desktop._desktop_build_needed(desktop_dir, project_root, source_mode=True) is False

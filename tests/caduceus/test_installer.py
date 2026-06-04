"""Regression tests for the Caduceus overlay installer."""

from __future__ import annotations

import subprocess
from pathlib import Path

import install_caduceus as installer


def _init_minimal_git_target(path: Path) -> str:
    path.mkdir()
    for name in ("run_agent.py", "cli.py", "toolsets.py"):
        (path / name).write_text("# stock hermes placeholder\n", encoding="utf-8")
    (path / "pyproject.toml").write_text('[project]\nversion = "0.15.1"\n', encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "review@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Review"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "synthetic stock target"], cwd=path, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def test_install_blocks_when_base_commit_is_unknown(tmp_path, capsys):
    target = tmp_path / "target"
    _init_minimal_git_target(target)

    rc = installer.do_install(str(target), dry_run=True, force=False)

    out = capsys.readouterr().out
    assert rc == 3
    assert "does not contain this build's base" in out


def test_install_blocks_when_target_is_behind_base(tmp_path, capsys, monkeypatch):
    target = tmp_path / "target"
    stock_commit = _init_minimal_git_target(target)
    (target / "base-marker.txt").write_text("newer base\n", encoding="utf-8")
    subprocess.run(["git", "add", "base-marker.txt"], cwd=target, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "newer base"], cwd=target, check=True)
    base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
    subprocess.run(["git", "checkout", "-q", stock_commit], cwd=target, check=True)
    monkeypatch.setattr(installer, "BASE_COMMIT", base_commit)

    rc = installer.do_install(str(target), dry_run=True, force=False)

    out = capsys.readouterr().out
    assert rc == 3
    assert "predates this build's base" in out


def test_find_packaged_asar_detects_macos_app_bundle(tmp_path):
    desktop = tmp_path / "apps" / "desktop"
    asar = (
        desktop
        / "release"
        / "mac-arm64"
        / "Hermes.app"
        / "Contents"
        / "Resources"
        / "app.asar"
    )
    asar.parent.mkdir(parents=True)
    asar.write_bytes(b"asar")

    assert installer._find_packaged_asar(str(desktop)) == str(asar)

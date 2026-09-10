"""Regression coverage for desktop bootstrap over immutable release runtimes.

A signed Hermes Setup installer may bootstrap a desktop app on top of an
existing release runtime. That runtime is intentionally git-free, carries an
immutable release manifest, and must never be fetched, reset, or replaced by
the generic git installer path.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = ROOT / "scripts" / "install.sh"
COMMIT = "a" * 40
INSTALLER_COMMIT = "b" * 40


def _release_manifest(path: Path, *, valid: bool = True, commit: str = COMMIT) -> None:
    payload = {
        "schema": "hermes-agent-release/v1" if valid else "unexpected/v1",
        "commit": commit,
        "final_runtime_git_free": True,
    }
    (path / ".hermes-release.json").write_text(json.dumps(payload), encoding="utf-8")


def _run_clone(install_dir: Path) -> subprocess.CompletedProcess[str]:
    script = f"""
set -e
source {shlex.quote(str(INSTALL_SH))} --manifest >/dev/null
INSTALL_DIR={shlex.quote(str(install_dir))}
INSTALL_COMMIT={INSTALLER_COMMIT}
BRANCH=main
clone_repo
printf 'PWD=%s\\n' "$PWD"
"""
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=30)


def _write_marker(install_dir: Path) -> subprocess.CompletedProcess[str]:
    script = f"""
set -e
source {shlex.quote(str(INSTALL_SH))} --manifest >/dev/null
INSTALL_DIR={shlex.quote(str(install_dir))}
INSTALL_COMMIT={INSTALLER_COMMIT}
BRANCH=main
write_bootstrap_marker
"""
    return subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=30)


def test_valid_git_free_release_runtime_survives_repository_stage(tmp_path: Path) -> None:
    install_dir = tmp_path / "hermes-agent"
    install_dir.mkdir()
    _release_manifest(install_dir)
    sentinel = install_dir / "local-patch-marker"
    sentinel.write_text("must survive", encoding="utf-8")

    result = _run_clone(install_dir)

    assert result.returncode == 0, result.stderr
    assert "git-free release runtime" in result.stdout
    assert sentinel.read_text(encoding="utf-8") == "must survive"
    assert f"PWD={install_dir}" in result.stdout


def test_non_release_directory_keeps_existing_git_safety_block(tmp_path: Path) -> None:
    install_dir = tmp_path / "hermes-agent"
    install_dir.mkdir()
    (install_dir / "arbitrary-file").write_text("not a runtime", encoding="utf-8")

    result = _run_clone(install_dir)

    assert result.returncode != 0
    assert "Directory exists but is not a git repository" in result.stdout


def test_release_manifest_must_be_strictly_valid(tmp_path: Path) -> None:
    install_dir = tmp_path / "hermes-agent"
    install_dir.mkdir()
    _release_manifest(install_dir, valid=False)

    result = _run_clone(install_dir)

    assert result.returncode != 0
    assert "Directory exists but is not a git repository" in result.stdout


def test_bootstrap_marker_prefers_verified_release_commit(tmp_path: Path) -> None:
    install_dir = tmp_path / "hermes-agent"
    install_dir.mkdir()
    _release_manifest(install_dir)

    result = _write_marker(install_dir)

    assert result.returncode == 0, result.stderr
    marker = json.loads((install_dir / ".hermes-bootstrap-complete").read_text(encoding="utf-8"))
    assert marker["pinnedCommit"] == COMMIT
    assert marker["pinnedCommit"] != INSTALLER_COMMIT
    assert re.fullmatch(r"[0-9a-f]{40}", marker["pinnedCommit"])
    assert "Ignoring installer --commit" in result.stdout


def test_bootstrap_marker_normalizes_uppercase_release_commit(tmp_path: Path) -> None:
    install_dir = tmp_path / "hermes-agent"
    install_dir.mkdir()
    _release_manifest(install_dir, commit="A" * 40)

    result = _write_marker(install_dir)

    assert result.returncode == 0, result.stderr
    marker = json.loads((install_dir / ".hermes-bootstrap-complete").read_text(encoding="utf-8"))
    assert marker["pinnedCommit"] == COMMIT

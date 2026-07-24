"""Tests for ``release.update_version_files`` write/return contract.

The release commit stages exactly the files ``update_version_files`` reports
back. A file that gets written but not returned is silently dropped from the
commit — which is how the desktop ``package.json`` bump went stale for several
releases (#68783). These tests pin the contract that every file the function
writes is also returned.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))

import release  # noqa: E402


def _prepare_tree(monkeypatch, root: Path, *, with_desktop: bool) -> None:
    """Pin release's module-level path constants at a throwaway temp tree so a
    bump never scribbles on the real repo files."""
    (root / "pyproject.toml").write_text(
        '[project]\nname = "hermes-agent"\nversion = "0.13.0"\n', encoding="utf-8"
    )
    version_dir = root / "hermes_cli"
    version_dir.mkdir()
    (version_dir / "__init__.py").write_text(
        '__version__ = "0.13.0"\n__release_date__ = "2026-05-14"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(release, "VERSION_FILE", version_dir / "__init__.py")
    monkeypatch.setattr(release, "PYPROJECT_FILE", root / "pyproject.toml")
    monkeypatch.setattr(release, "REPO_ROOT", root)

    if with_desktop:
        pkg_dir = root / "apps" / "desktop"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "package.json").write_text(
            json.dumps({"name": "hermes", "version": "0.13.0"}, indent=2) + "\n",
            encoding="utf-8",
        )


def test_returns_desktop_package_when_present(monkeypatch, tmp_path):
    """When the desktop package.json exists it must be both bumped AND returned
    so the release commit can stage it (#68783)."""
    _prepare_tree(monkeypatch, tmp_path, with_desktop=True)

    modified = release.update_version_files("0.14.0", "2026-05-21")

    desktop_pkg = tmp_path / "apps" / "desktop" / "package.json"
    assert desktop_pkg in modified, "desktop package.json written but not returned"
    # The bump actually landed on disk.
    assert json.loads(desktop_pkg.read_text())["version"] == "0.14.0"
    # Core version files are always reported.
    assert release.VERSION_FILE in modified
    assert release.PYPROJECT_FILE in modified


def test_omits_desktop_package_when_absent(monkeypatch, tmp_path):
    """Older release branches predate the desktop app — no package.json means
    it must not be reported (staging a missing path would abort the release)."""
    _prepare_tree(monkeypatch, tmp_path, with_desktop=False)

    modified = release.update_version_files("0.14.0", "2026-05-21")

    assert all("package.json" not in str(p) for p in modified)
    assert release.VERSION_FILE in modified
    assert release.PYPROJECT_FILE in modified

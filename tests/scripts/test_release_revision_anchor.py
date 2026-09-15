"""Version files record the Git count of the release commit they describe."""
import json
import runpy
import subprocess
from pathlib import Path

import pytest

from scripts import release


@pytest.mark.parametrize("existing", [False, True])
def test_version_writer_creates_and_refreshes_release_revision(tmp_path, monkeypatch, existing):
    repo = tmp_path / "source"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True,
            text=True, encoding="utf-8", timeout=20,
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Release fixture")
    git("config", "user.email", "release@example.invalid")
    files = {
        "VERSION_FILE": ("hermes_cli/__init__.py", '__version__ = "1.0.0"\n__release_date__ = "2000.1.1"\n'),
        "PYPROJECT_FILE": ("pyproject.toml", '[project]\nname="hermes-agent"\nversion = "1.0.0"\n'),
        "DESKTOP_PKG_FILE": ("apps/desktop/package.json", '{"name":"desktop", "version":"1.0.0"}\n'),
        "PKG_LOCK_FILE": ("package-lock.json", json.dumps({"version": "9.9.9", "packages": {"apps/desktop": {"name": "desktop", "version": "1.0.0"}}}) + "\n"),
        "UV_LOCK_FILE": ("uv.lock", 'version = 1\n[[package]]\nname = "hermes-agent"\nversion = "1.0.0"\n'),
    }
    if existing:
        rel, content = files["VERSION_FILE"]
        files["VERSION_FILE"] = (rel, content + "__release_rev_count__ = 999\n")
    for name, (rel, content) in files.items():
        file = repo / rel
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
        monkeypatch.setattr(release, name, file)
    monkeypatch.setattr(release, "REPO_ROOT", repo)
    git("add", ".")
    git("commit", "-qm", "fixture")
    before = git("rev-parse", "HEAD")

    paths = release.update_version_files("1.1.0", "2000.1.2")
    assert set(map(Path, paths)) == {repo / rel for rel, _ in files.values()}
    assert git("rev-parse", "HEAD") == before
    git("add", ".")
    git("commit", "-qm", "release fixture")
    version = runpy.run_path(str(repo / "hermes_cli/__init__.py"))
    assert version["__version__"] == "1.1.0"
    assert version["__release_rev_count__"] == int(git("rev-list", "--count", "HEAD"))
    lock = json.loads((repo / "package-lock.json").read_text(encoding="utf-8"))
    assert lock["version"] == "9.9.9"
    assert lock["packages"]["apps/desktop"]["version"] == "1.1.0"

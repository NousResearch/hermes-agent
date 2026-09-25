"""A fresh clone never lands inside, or on top of, an unrelated destination."""
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _origin(tmp_path: Path) -> Path:
    origin = tmp_path / "origin"
    origin.mkdir()
    git = ["git", "-C", str(origin)]
    for args in (["init", "-b", "main"], ["config", "user.email", "f@example.invalid"],
                 ["config", "user.name", "Fixture"]):
        subprocess.run([*git, *args], check=True, capture_output=True)
    (origin / "README").write_text("complete checkout\n")
    subprocess.run([*git, "add", "README"], check=True, capture_output=True)
    subprocess.run([*git, "commit", "-m", "fixture"], check=True, capture_output=True)
    return origin


def _stage_repository(tmp_path: Path, dest: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=(tmp_path / "home").as_posix(),
               HERMES_INSTALL_DIR=dest.as_posix(), HERMES_REPO_URL=_origin(tmp_path).as_posix())
    script = f"source {shlex.quote((ROOT / 'scripts/install.sh').as_posix())} --manifest\nstage_repository\n"
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("shape", ["non-empty-dir", "file"])
def test_unrelated_destination_is_refused_untouched(tmp_path, shape):
    dest = tmp_path / "install"
    if shape == "file":
        dest.write_text("not a checkout\n")
    else:
        (dest / "notes").mkdir(parents=True)
        (dest / "notes" / "keep.txt").write_text("user data\n")
    before = sorted(str(p.relative_to(dest)) for p in dest.rglob("*")) if dest.is_dir() else dest.read_text()
    result = _stage_repository(tmp_path, dest)
    assert result.returncode != 0
    assert "not a Hermes git checkout" in result.stderr
    after = sorted(str(p.relative_to(dest)) for p in dest.rglob("*")) if dest.is_dir() else dest.read_text()
    assert after == before, "refusal must not nest a tree/ inside or alter the destination"
    assert not (dest / "tree").exists()
    assert not list(tmp_path.glob(".hermes-clone-*"))


def test_empty_destination_directory_becomes_the_checkout(tmp_path):
    dest = tmp_path / "install"
    dest.mkdir()
    result = _stage_repository(tmp_path, dest)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (dest / "README").read_text() == "complete checkout\n", "checkout is AT the destination, not under tree/"
    assert not (dest / "tree").exists()

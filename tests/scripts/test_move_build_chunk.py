"""scripts/move/build_chunk.py: a ROOT outside the repository is refused before anything is removed."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "move" / "build_chunk.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_chunk", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_absolute_root_is_refused_and_left_in_place(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.py").write_text("x = 1\n", encoding="utf-8")
    build_chunk = _load()

    with pytest.raises(build_chunk.Failure, match="relative to the repository root"):
        build_chunk.move_roots(repo, "01-x", [str(outside)])

    assert (outside / "keep.py").is_file()

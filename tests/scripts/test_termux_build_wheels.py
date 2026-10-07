"""Pinned source dependencies build online and install from wheelhouse offline."""
from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from scripts.termux import build_wheels as builder


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
def test_requirement_readers_preserve_pins_and_reject_malformed_rows(tmp_path, bom):
    lock, export, resolved, reqs = (tmp_path / name for name in
                                    ("uv.lock", "export.txt", "resolved.txt", "reqs.txt"))
    lock.write_bytes(bom + 'package = []\n# café\n'.encode("utf-8"))
    export.write_bytes(bom + 'example==1.0\n# café\n'.encode("utf-8"))
    builder.normalize_reqs(export, lock, resolved)
    assert resolved.read_bytes() == b"example\t==1.0\t\t\n"
    resolved.write_bytes(bom + resolved.read_bytes())
    assert builder.load_entries(resolved) == {"example": "==1.0"}
    builder.write_reqs_file(resolved, reqs)
    assert reqs.read_bytes() == b"example==1.0\n"
    resolved.write_bytes(bom + b"example\t==1.0\n")
    with pytest.raises(ValueError, match="fields"):
        builder.load_entries(resolved)


def test_source_pin_survives_resolution_build_and_offline_install(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=75.3.4"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '[project]\nname = "hermes-source-fixture"\nversion = "8.0.0"\n', encoding="utf-8")
    (repo / "source_fixture.py").write_text('VALUE = "from pinned source"\n', encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-m", "fixture"],
                   check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    (repo / "source_fixture.py").write_text('VALUE = "wrong moving head"\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-am", "advance head"],
                   check=True, capture_output=True)
    url = repo.as_uri()
    source = f"git+{url}@{commit}"
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "hermes-source-fixture"\nversion = "8.0.0"\n'
        f'source = {{ git = "{url}?rev={commit}#{commit}" }}\n', encoding="utf-8")
    export = tmp_path / "requirements.txt"
    export.write_text(
        'hermes-source-fixture==7.2.2 ; python_version < "3.11"\n'
        f'hermes-source-fixture @ {source} ; python_version >= "3.11"\n', encoding="utf-8")
    resolved = tmp_path / "resolved.txt"
    builder.normalize_reqs(export, lock, resolved)
    specs = builder.load_entries(resolved)
    assert specs == {"hermes-source-fixture": f" @ {source}"}
    reqs = tmp_path / "offline.txt"
    builder.write_reqs_file(resolved, reqs)
    assert "@" not in reqs.read_text(encoding="utf-8")
    assert 'hermes-source-fixture==8.0.0 ; python_version >= "3.11"' in reqs.read_text(encoding="utf-8")

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    build_env = tmp_path / "build-env"
    venv.EnvBuilder(with_pip=True).create(build_env)
    python = build_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "pip==26.2.1"],
                   check=True, capture_output=True)
    builder.build_wheels(list(specs), specs, wheels, python=str(python))
    assert len(list(wheels.glob("hermes_source_fixture-8.0.0-*.whl"))) == 1
    # Remove the original source URL to rule out a Git fallback.
    repo.rename(tmp_path / "unavailable-source")
    target = tmp_path / "installed"
    subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps",
                    "--find-links", str(wheels), "--target", str(target), "-r", str(reqs)],
                   check=True, capture_output=True)
    result = subprocess.check_output(
        [sys.executable, "-c", "import source_fixture; print(source_fixture.VALUE)"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(target)}, text=True)
    assert result.strip() == "from pinned source"


def test_source_pin_rejects_moved_revision_and_missing_locked_metadata(tmp_path):
    commit = "a" * 40
    source = "https://example.invalid/package.git"
    export, lock, resolved = (tmp_path / name for name in ("requirements.txt", "uv.lock", "resolved.txt"))
    lock.write_text('[[package]]\nname = "example"\nversion = "8.0.0"\n'
                    f'source = {{ git = "{source}?rev={commit}#{commit}" }}\n', encoding="utf-8")
    for revision in ("main", "b" * 40):
        export.write_text(f"example @ git+{source}@{revision}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="commit|lock"):
            builder.normalize_reqs(export, lock, resolved)
    export.write_text(f"example @ git+{source}@{commit}\n", encoding="utf-8")
    lock.write_text("package = []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lock"):
        builder.normalize_reqs(export, lock, resolved)

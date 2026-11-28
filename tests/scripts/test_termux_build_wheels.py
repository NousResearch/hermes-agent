"""Pinned source dependencies build online and install from wheelhouse offline."""
from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from scripts.termux import build_wheels as builder

# A PEP 517 backend that lives in the fixture's own source tree: the wheel build needs nothing
# from an index (the setuptools backend it replaces was fetched from PyPI on every run). It
# packages ``source_fixture.py`` as checked out, so the pinned commit still decides the content.
_IN_TREE_BACKEND = '''\
import base64, hashlib, zipfile
from pathlib import Path

_DIST, _VERSION = "hermes_source_fixture", "8.0.0"


def get_requires_for_build_wheel(config_settings=None):
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    info = f"{_DIST}-{_VERSION}.dist-info"
    files = {
        "source_fixture.py": Path("source_fixture.py").read_bytes(),
        f"{info}/METADATA": b"Metadata-Version: 2.1\\nName: hermes-source-fixture\\nVersion: 8.0.0\\n",
        f"{info}/WHEEL": b"Wheel-Version: 1.0\\nGenerator: fixture\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n",
    }
    record = "".join(
        f"{name},sha256={base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()},{len(data)}\\n"
        for name, data in files.items()) + f"{info}/RECORD,,\\n"
    files[f"{info}/RECORD"] = record.encode()
    name = f"{_DIST}-{_VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(Path(wheel_directory) / name, "w") as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    return name
'''


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
def test_requirement_readers_preserve_pins_and_reject_malformed_rows(tmp_path, bom):
    lock, export, resolved, reqs = (tmp_path / name for name in
                                    ("uv.lock", "export.txt", "resolved.txt", "reqs.txt"))
    lock.write_bytes(bom + 'package = []\n# café\n'.encode())
    export.write_bytes(bom + 'example==1.0\n# café\n'.encode())
    builder.normalize_reqs(export, lock, resolved)
    assert resolved.read_bytes() == b"example\t==1.0\t\t\n"
    resolved.write_bytes(bom + resolved.read_bytes())
    assert builder.load_entries(resolved) == {"example": "==1.0"}
    builder.write_reqs_file(resolved, reqs)
    assert reqs.read_bytes() == b"example==1.0\n"
    resolved.write_bytes(bom + b"example\t==1.0\n")
    with pytest.raises(ValueError, match="fields"):
        builder.load_entries(resolved)


def test_source_pin_survives_resolution_build_and_offline_install(tmp_path, monkeypatch):
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n'
        '[project]\nname = "hermes-source-fixture"\nversion = "8.0.0"\n', encoding="utf-8")
    (repo / "backend.py").write_text(_IN_TREE_BACKEND, encoding="utf-8")
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
    # The interpreter's bundled pip builds the wheel; no index is consulted at any step, so the
    # build proves the git pin alone produces the wheel.
    monkeypatch.setenv("PIP_NO_INDEX", "1")
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


def test_google_re2_builds_in_the_pinned_toolchain_environment(tmp_path, monkeypatch):
    # Stands in for the real sdist: named google-re2, and buildable only where the build runs in
    # the caller's environment. Its declared build requirement exists on no index, so an isolated
    # build (which must install it first) cannot start; the real sdist needs the pybind11 that
    # TOOLCHAIN_PINS installs and declares nothing. Its backend refuses GITHUB_ACTIONS, the
    # variable that switches the real setup.py to a bazel build.
    project = tmp_path / "google-re2-source"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["hermes-absent-build-requirement==1.0"]\n'
        'build-backend = "backend"\nbackend-path = ["."]\n'
        '[project]\nname = "google-re2"\nversion = "8.0.0"\n', encoding="utf-8")
    backend = _IN_TREE_BACKEND.replace("hermes_source_fixture", "google_re2").replace(
        "hermes-source-fixture", "google-re2")
    (project / "backend.py").write_text(
        'import os\nassert "GITHUB_ACTIONS" not in os.environ\n' + backend, encoding="utf-8")
    (project / "source_fixture.py").write_text("VALUE = 1\n", encoding="utf-8")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    build_env = tmp_path / "build-env"
    venv.EnvBuilder(with_pip=True).create(build_env)
    python = build_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    specs = {"google-re2": f" @ {project.as_uri()}"}
    builder.build_wheels(["google-re2"], specs, wheels, python=str(python))
    assert len(list(wheels.glob("google_re2-8.0.0-*.whl"))) == 1


def test_import_gate_names_the_module_google_re2_installs(tmp_path):
    # The distribution is google-re2, the importable extension re2._re2: the gate's default name
    # (google_re2) exists nowhere. A stand-in package answers for the real one.
    package = tmp_path / "re2"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "_re2.py").write_text("print('stand-in extension loaded')\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(Path(builder.__file__).resolve()), "--import-modules", "google-re2"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert "stand-in extension loaded" in result.stdout and "imported re2._re2" in result.stdout


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


@pytest.mark.platforms("posix")
def test_termux_and_musl_builds_compile_google_re2_against_the_same_pybind11():
    # Termux builds google-re2 inside the TOOLCHAIN_PINS venv, musl through uv's
    # extra-build-dependencies table: a bump of one would split the two source builds.
    import tomllib

    from packaging.requirements import Requirement

    repo = Path(__file__).resolve().parents[2]
    shell = subprocess.run(
        ["bash", "-c", 'source scripts/termux/build_config.sh && printf "%s\\n" "${TOOLCHAIN_PINS[@]}"'],
        cwd=repo, capture_output=True, text=True, check=True)
    toolchain = [Requirement(pin) for pin in shell.stdout.split()]
    uv = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8-sig"))["tool"]["uv"]
    extra = [Requirement(spec) for spec in uv["extra-build-dependencies"]["google-re2"]]
    pybind11 = [str(requirement) for requirement in toolchain if requirement.name == "pybind11"]
    assert pybind11 and pybind11 == [str(requirement) for requirement in extra if requirement.name == "pybind11"]

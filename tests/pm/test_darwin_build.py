"""Native static-library builds use PM's archiver, not PBS's build directory."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from tests.pm._fixtures import build_worker as build_worker, client as client, isolated_python as isolated_python


@pytest.mark.platforms("macos")
def test_managed_build_archives_native_code(tmp_path, build_worker):
    from pm import build_requirements_environment

    source = tmp_path / "native_archive_probe-1.0"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[build-system]\nrequires=[]\nbuild-backend="backend"\nbackend-path=["."]\n',
        encoding="utf-8",
    )
    (source / "answer.c").write_text("int answer(void) { return 42; }\n", encoding="utf-8")
    (source / "probe.c").write_text('''#include <Python.h>
extern int answer(void);
static PyObject *value(PyObject *self, PyObject *args) { return PyLong_FromLong(answer()); }
static PyMethodDef methods[] = {{"value", value, METH_NOARGS, "answer"}, {NULL,NULL,0,NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "native_archive_probe", NULL, -1, methods};
PyMODINIT_FUNC PyInit_native_archive_probe(void) { return PyModule_Create(&module); }
''', encoding="utf-8")
    (source / "backend.py").write_text('''import os, shlex, subprocess, sys, sysconfig
from pathlib import Path
from zipfile import ZipFile

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    cc = shlex.split(os.environ.get("CC") or sysconfig.get_config_var("CC"))
    ar = shlex.split(os.environ.get("AR") or sysconfig.get_config_var("AR"))
    subprocess.run(cc + ["-fPIC", "-c", "answer.c", "-o", "answer.o"], check=True)
    subprocess.run(ar + ["rcs", "libanswer.a", "answer.o"], check=True)
    extension = "native_archive_probe" + sysconfig.get_config_var("EXT_SUFFIX")
    subprocess.run(cc + ["-bundle", "-undefined", "dynamic_lookup",
        "-I" + sysconfig.get_path("include"), "probe.c", "libanswer.a", "-o", extension], check=True)
    platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    abi = "cp" + str(sys.version_info.major) + str(sys.version_info.minor)
    tag = abi + "-" + abi + "-" + platform
    name = "native_archive_probe-1.0-" + tag + ".whl"
    dist = "native_archive_probe-1.0.dist-info"
    entries = {
        extension: Path(extension).read_bytes(),
        dist + "/METADATA": "Metadata-Version: 2.1\\nName: native-archive-probe\\nVersion: 1.0\\n",
        dist + "/WHEEL": "Wheel-Version: 1.0\\nRoot-Is-Purelib: false\\nTag: " + tag + "\\n",
    }
    entries[dist + "/RECORD"] = "".join(path + ",,\\n" for path in entries)
    with ZipFile(Path(wheel_directory) / name, "w") as wheel:
        for path, body in entries.items():
            wheel.writestr(path, body)
    return name
''', encoding="utf-8")
    archive = tmp_path / (source.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname=source.name)
    env = {key: value for key, value in os.environ.items() if key not in ("AR", "CC")}
    executable = build_requirements_environment(
        ["native-archive-probe @ " + archive.as_uri()], out=tmp_path / "built",
        python=Path(sys.executable), env=env, cache=tmp_path / "cache",
        offline=True, explicit=True,
    )
    probe = subprocess.run(
        [str(executable), "-I", "-c", "import native_archive_probe; print(native_archive_probe.value())"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    assert probe.stdout.strip() == "42"


@pytest.mark.platforms("macos")
def test_archiver_defaults_preserve_explicit_and_custom_toolchains(tmp_path, monkeypatch):
    from pm.environment import managed_environment

    pinned = Path(sys.executable)
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (tmp_path / "uv", pinned))
    alias = tmp_path / "python-alias"
    alias.symlink_to(pinned)
    caller_env = {"AR": "/custom/ar", "CC": "/custom/clang"}
    explicit = managed_environment(tmp_path / "explicit", python=alias, env=caller_env, cache=tmp_path)
    assert explicit.env == caller_env
    default = managed_environment(tmp_path / "default", python=alias, env={}, cache=tmp_path)
    assert default.env == {"AR": "/usr/bin/ar"}
    # A separately selected interpreter (e.g. Nix) retains its own sysconfig.
    custom = managed_environment(tmp_path / "custom", python=tmp_path / "custom-python", env={}, cache=tmp_path)
    assert custom.env == {}
    assert caller_env == {"AR": "/custom/ar", "CC": "/custom/clang"}

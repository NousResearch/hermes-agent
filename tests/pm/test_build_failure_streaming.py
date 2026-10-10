"""A verbose local PEP 517 failure must retain its production classification."""

import io
import os
from pathlib import Path
import shutil
import sys

import pytest

from pm.environment import BuildFailure, PythonEnvironment


@pytest.mark.parametrize("output_mode", ["captured", "contained", "verbose"])
def test_local_backend_failure_keeps_classification_after_large_output(tmp_path, monkeypatch, output_mode):
    uv = shutil.which("uv")
    assert uv, "PM construction tests require uv on PATH"
    source = tmp_path / "broken-local-package"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname="broken-local-package"\nversion="1.0"\n'
        '[build-system]\nrequires=[]\nbuild-backend="local_backend"\nbackend-path=["."]\n',
        encoding="utf-8",
    )
    (source / "local_backend.py").write_text(
        'def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n'
        '    for index in range(300):\n'
        '        print("native build diagnostic " + str(index) + ": " + "x" * 100)\n'
        '    raise RuntimeError("local backend failed while compiling")\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_VERBOSE", "1" if output_mode == "verbose" else "0")
    environment = PythonEnvironment(
        uv=Path(uv), python=Path(sys.executable), destination=tmp_path / "environment",
        cache=tmp_path / "cache", env=dict(os.environ), offline=True,
        output=None if output_mode == "captured" else io.StringIO(),
    )
    environment.create()
    with pytest.raises(BuildFailure):
        environment.install_requirements([str(source)])

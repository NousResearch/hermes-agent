"""PM validation diagnostics survive native Windows code pages (#122772)."""
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest

from pm.environment import PythonEnvironment
from pm.environments import venv_python
from pm.package import InstallError
from pm.runtime import _validate, runtime_environment
from pm.runtime_stage import stage_runtime
from tests.pm._fixtures import _wheel


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("phase", ["staging", "reuse"])
def test_runtime_validation_preserves_unicode_failure(tmp_path, phase):
    """Use real offline uv installs and interpreter probes, not mocked output."""
    uv = shutil.which("uv")
    assert uv, "the runtime validation test requires real uv"
    message = "\u5957\u4ef6\u9a57\u8b49\u5931\u6557"
    wheel = _wheel(tmp_path, "packaging")
    with zipfile.ZipFile(wheel) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    contents["packaging/__init__.py"] = f"raise RuntimeError({message!r})\n".encode()
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname="pm-encoding-proof"\nversion="1"\nrequires-python=">=3.11"\n'
        f'dependencies=[{json.dumps("packaging @ " + wheel.as_uri())}]\n'
        '[tool.uv]\npackage=false\n', encoding="utf-8",
    )
    destination = tmp_path / "\u5957\u4ef6-runtime"
    cache = tmp_path / "cache"
    env = runtime_environment()
    engine = PythonEnvironment(
        uv=Path(uv), python=Path(sys.executable), destination=destination,
        cache=cache, env=env, offline=True, no_config=True,
    )
    engine.lock(project)
    with pytest.raises(InstallError, match="dependency validation failed") as failure:
        stage_runtime(Path(uv), Path(sys.executable), destination,
                      project=project, offline=True, cache=cache)

    diagnostic = str(failure.value) if phase == "staging" else _validate(venv_python(destination), env)
    assert f"RuntimeError: {message}" in diagnostic
    assert str(destination) in diagnostic

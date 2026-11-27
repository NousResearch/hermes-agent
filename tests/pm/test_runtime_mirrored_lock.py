"""The PM lock stays authoritative when provisioning through a mirrored index."""
from pathlib import Path
import re
import subprocess

import pytest

from pm.package import InstallError
from pm.runtime_stage import stage_runtime


@pytest.mark.parametrize("mirror,alter_hash", [(False, False), (True, False), (True, True)])
def test_stage_reconciles_mirror_only_in_scratch(tmp_path, monkeypatch, mirror, alter_hash):
    from pm import environment, runtime, runtime_stage

    project = Path(runtime_stage.__file__).parent
    original = (project / "uv.lock").read_bytes()
    seen = []

    class FakeEnvironment:
        def __init__(self, **kwargs):
            self.executable = tmp_path / "python"

        def create(self):
            pass

        def lock(self, snapshot, **kwargs):
            seen.append("lock")
            lock = snapshot / "uv.lock"
            updated = lock.read_text(encoding="utf-8").replace(
                'registry = "https://pypi.org/simple"',
                'registry = "https://mirror.example/simple"',
            )
            if alter_hash:
                updated = re.sub(r'(?<=hash = "sha256:)[0-9a-f]',
                                 lambda match: "f" if match.group() != "f" else "e", updated, count=1)
            lock.write_text(updated, encoding="utf-8")

        def sync(self, snapshot, **kwargs):
            seen.append("sync")
            assert kwargs["locked"] is True
            assert (snapshot / "uv.lock").read_bytes() != original if mirror else (snapshot / "uv.lock").read_bytes() == original

    monkeypatch.setattr(environment, "PythonEnvironment", FakeEnvironment)
    monkeypatch.setattr(runtime, "runtime_environment", lambda: {"UV_INDEX_URL": "https://mirror.example/simple"} if mirror else {})
    monkeypatch.setattr(runtime_stage.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "", ""))

    if alter_hash:
        with pytest.raises(InstallError, match="changed the pinned"):
            stage_runtime(tmp_path / "uv", tmp_path / "python", tmp_path / "dest", project=project, cache=tmp_path / "cache")
        assert seen == ["lock"]
    else:
        stage_runtime(tmp_path / "uv", tmp_path / "python", tmp_path / "dest", project=project, cache=tmp_path / "cache")
        assert seen == (["lock", "sync"] if mirror else ["sync"])
    assert (project / "uv.lock").read_bytes() == original

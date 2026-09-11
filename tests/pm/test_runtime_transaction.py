"""Preparing dependencies never mutates the selected environment or plugin config."""
from pathlib import Path
import importlib

import pytest

from pm.lock import Facts
from pm.package import InstallError


@pytest.mark.parametrize("failure", ["build", "record", "missing", None])
def test_sync_commits_only_a_successful_candidate(tmp_path, monkeypatch, failure):
    import pm.paths as paths
    import pm.registry as registry
    from hermes_cli.runtime_paths import selected_venv
    from pm.packages import Venv

    ensure = importlib.import_module("pm.ensure")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_text("initial-lock")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(paths, "repo_root", lambda: root)
    monkeypatch.setattr(ensure, "sealed", lambda: False)
    monkeypatch.setattr(ensure, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr("pm.workspace.enabled_member_dirs", lambda: [])
    config = tmp_path / "home" / "config.yaml"
    config.parent.mkdir()
    config.write_text("plugins:\n  enabled: [working]\n")
    original_config = config.read_bytes()
    runtime_facts = Facts(paths.runtime_facts_path())
    previous = paths.runtime_facts_path().parent / "environments" / "previous" / "venv"
    previous.mkdir(parents=True)
    (previous / "pyvenv.cfg").write_text("home = previous")
    runtime_facts.record_state("venv", "old", ["base"], environment=previous)
    if failure == "missing":
        (previous / "pyvenv.cfg").unlink()
        runtime_facts.record_state("venv", "new", ["base", "new-extra"], environment=previous)
    before = runtime_facts.path.read_bytes()
    candidate = previous.parent.parent / "candidate" / "venv"

    class TestVenv(Venv):
        def expected_stamp(self, extras):
            return "new"

        def apply(self, extras):
            if failure != "missing":
                assert selected_venv(root) == previous
            assert extras == ["base", "new-extra"]
            if failure == "build":
                raise InstallError("venv", "download unavailable")
            candidate.mkdir(parents=True)
            (candidate / "pyvenv.cfg").write_text("home = candidate")
            return {"environment": candidate}

    monkeypatch.setitem(registry._packages, "venv", TestVenv())
    if failure == "record":
        def refuse_record(self, *args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(Facts, "record_state", refuse_record)
    if failure in ("build", "record"):
        with pytest.raises((InstallError, OSError)):
            ensure.sync_venv(["new-extra"], explicit=True)
        assert runtime_facts.path.read_bytes() == before
        assert selected_venv(root) == previous
    else:
        ensure.sync_venv(["new-extra"], explicit=True)
        assert selected_venv(root) == candidate
        assert Facts(runtime_facts.path).get("venv")["extras"] == ["base", "new-extra"]
    assert previous.is_dir()
    assert config.read_bytes() == original_config


def test_real_uv_builds_separate_environment_before_selection(tmp_path, monkeypatch):
    import shutil
    import subprocess
    import sys
    import pm.paths as paths
    from pm.packages import Venv
    from hermes_cli.runtime_paths import selected_venv

    uv = shutil.which("uv")
    assert uv is not None, "uv is required for the real environment contract"
    core = tmp_path / "core"
    core.mkdir()
    (core / "pyproject.toml").write_text(
        '[project]\nname="environment-test"\nversion="1"\nrequires-python=">=3.11"\n'
        '[tool.uv]\npackage=false\n', encoding="utf-8",
    )
    subprocess.run([uv, "lock", "--offline", "--python", sys.executable], cwd=core, check=True, capture_output=True)
    original_lock = (core / "uv.lock").read_bytes()
    base = core / "venv"
    base.mkdir()
    (base / "pyvenv.cfg").write_text("home = previous")
    monkeypatch.setattr(paths, "repo_root", lambda: core)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("pm.workspace.enabled_member_dirs", lambda: [])
    ensure = importlib.import_module("pm.ensure")
    from pm.packages import uv_env
    monkeypatch.setattr(ensure, "uv", lambda **kw: (uv, {**uv_env(kw.get("base_env")), "UV_PYTHON": sys.executable}))
    prepared = Venv().apply([])
    assert selected_venv(core) == base
    candidate = prepared["environment"]
    assert candidate != base and (candidate / "pyvenv.cfg").is_file()
    python = candidate / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    probe = subprocess.run([str(python), "-c", "import sys; print(sys.prefix)"], capture_output=True, text=True, check=True)
    assert Path(probe.stdout.strip()).resolve() == candidate.resolve()
    assert (core / "uv.lock").read_bytes() == original_lock


def test_lazy_import_reports_restart_instead_of_importing_mixed_versions(tmp_path, monkeypatch):
    from hermes_cli.runtime_paths import runtime_facts_path
    import pm.extras as extras
    import pm.paths as paths

    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(paths, "repo_root", lambda: root)
    monkeypatch.setattr(extras, "available", lambda _: False)
    monkeypatch.setattr(extras, "extra_supported", lambda _: True)
    candidate = runtime_facts_path(root).parent / "environments" / "new" / "venv"
    candidate.mkdir(parents=True)
    (candidate / "pyvenv.cfg").write_text("home = test")
    def prepare(requested):
        Facts(runtime_facts_path(root)).record_state("venv", "new", requested, environment=candidate)
    monkeypatch.setattr("pm.client.sync_venv", prepare)
    with pytest.raises(InstallError, match="restart"):
        extras.ensure_import("new-extra")

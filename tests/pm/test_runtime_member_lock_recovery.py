"""Regression for #122593: PM bootstrap restores a stripped member lock first.

A workspace materialized by the old snapshot code carries pm/pyproject.toml but
not pm/uv.lock. The PM runtime preflight must restore the exact bytes from this
install's codebase repo before computing its identity digest — or refuse with an
actionable InstallError naming the missing file, never a bare FileNotFoundError.
"""
import json
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from pm.environments import dependency_home_root
from pm.package import InstallError
from pm.runtime import _inputs, prepare_runtime

MANIFEST = b'[project]\nname = "pm"\nversion = "0"\n'
SOURCE_LOCK = b"the lock that shipped with this manifest\n"


@pytest.fixture()
def staged_runtime(monkeypatch):
    """Publish a PM runtime generation without building one (no uv, no network)."""

    def fake_stage(uv, python, destination, **kwargs):
        destination.mkdir(parents=True)
        return destination / "python"

    monkeypatch.setattr("pm.runtime_stage.stage_runtime", fake_stage)


def _broken_project(tmp_path: Path) -> Path:
    project = tmp_path / "snapshot" / "workspace" / "pm"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_bytes(MANIFEST)
    return project


def _codebase_pm(lock: bytes) -> Path:
    """The codebase repo's pm member: where recovery reads authoritative bytes."""
    homes = list(dict.fromkeys([dependency_home_root(), get_hermes_home()]))
    for home in homes:
        member = home / "hermes-agent" / "pm"
        member.mkdir(parents=True, exist_ok=True)
        (member / "pyproject.toml").write_bytes(MANIFEST)
        (member / "uv.lock").write_bytes(lock)
    return homes[0] / "hermes-agent" / "pm"


def test_preflight_restores_stripped_member_lock_before_identity(tmp_path, staged_runtime):
    project = _broken_project(tmp_path)
    source_pm = _codebase_pm(SOURCE_LOCK)
    python, root = tmp_path / "python", tmp_path / "pm-runtime"

    prepare_runtime(Path("uv"), python, root, project=project)

    assert (project / "uv.lock").read_bytes() == SOURCE_LOCK
    identity = _inputs(project, python)
    assert identity == _inputs(source_pm, python)
    published = json.loads((root / "selected.json").read_text(encoding="utf-8"))
    assert published["inputs"] == identity, "identity must digest the restored bytes"


def test_preflight_refuses_without_a_source_and_never_touches_a_present_lock(tmp_path, staged_runtime):
    project = _broken_project(tmp_path)
    python, root = tmp_path / "python", tmp_path / "pm-runtime"

    with pytest.raises(InstallError) as raised:
        prepare_runtime(Path("uv"), python, root, project=project)
    message = str(raised.value)
    assert str(project / "uv.lock") in message, "the error must name the missing input"
    assert "hermes-agent" in message, "the error must say how to restore it"
    assert not (project / "uv.lock").exists(), "a missing lock is never synthesized"

    kept = b"kept lock bytes\n"
    (project / "uv.lock").write_bytes(kept)
    _codebase_pm(b"a different lock\n")
    before = (project / "uv.lock").stat().st_mtime_ns

    prepare_runtime(Path("uv"), python, root, project=project)

    assert (project / "uv.lock").read_bytes() == kept, "a present lock is authoritative"
    assert (project / "uv.lock").stat().st_mtime_ns == before, "a present lock is never rewritten"

"""Dependency-only bootstrap retries must preserve the restored generation."""

from pathlib import Path

import pytest

from tests.windows_installer_fixtures import (
    ORIGINAL, VALIDATED, _generation, _stage,
    fake_uv as fake_uv,
    install as install,
    powershell_host as powershell_host,
)

pytestmark = pytest.mark.windows_only


@pytest.mark.parametrize("mode", ["deps-fail", "crash-after-restore", "crash-after-rollback-clear"])
def test_dependency_only_retry_protects_restored_original(install: Path, mode: str):
    assert _stage(install, "venv")[0] == 0
    code, output = _stage(install, "dependencies", mode)
    assert code != 0, output
    assert _generation(install / "venv") == ORIGINAL
    # The native bootstrap retries this same stage on no-frame exit -1. Do not
    # run venv here: retry must establish recovery before uv mutates its target.
    code, output = _stage(install, "dependencies", "deps-fail")
    assert code != 0, output
    assert _generation(install / "venv") == ORIGINAL
    assert not (install / "venv.pending-backup").exists()
    code, output = _stage(install, "dependencies")
    assert code == 0, output
    assert _generation(install / "venv") == VALIDATED
    assert not (install / "venv.pending-backup").exists()


@pytest.mark.parametrize("mode", ["ok", "deps-fail"])
def test_direct_dependencies_establishes_recovery_before_mutation(install: Path, mode: str):
    code, output = _stage(install, "dependencies", mode)
    assert (code == 0) == (mode == "ok"), output
    assert _generation(install / "venv") == (VALIDATED if mode == "ok" else ORIGINAL)
    if mode == "ok":
        events = (install.parent.parent / "validation-events.txt").read_text()
        assert "baseline:pending=True" in events
    assert not (install / "venv.pending-backup").exists()

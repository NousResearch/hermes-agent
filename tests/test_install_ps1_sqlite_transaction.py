"""SQLite verification participates in the existing recoverable venv transaction.

Builds on nahyeongjin1's #72034 runtime gate and native stage tests, while
retaining egilewski's transaction and fangliquanflq's original-generation retry.
"""

from pathlib import Path

import pytest

from tests.windows_installer_fixtures import (
    ORIGINAL,
    VALIDATED,
    _generation,
    _stage,
    fake_uv as fake_uv,
    install as install,
    powershell_host as powershell_host,
)

pytestmark = pytest.mark.windows_only


def test_runtime_repair_failure_restores_original_before_same_stage_retry(
    install: Path,
):
    assert _stage(install, "venv")[0] == 0
    code, output = _stage(install, "dependencies", "sqlite-fail")
    assert code != 0, output
    assert _generation(install / "venv") == ORIGINAL
    assert not (install / "venv.pending-backup").exists()
    code, output = _stage(install, "dependencies", "sqlite-repaired")
    assert code == 0, output
    assert _generation(install / "venv") == VALIDATED
    events = (install.parent.parent / "validation-events.txt").read_text()
    assert "sqlite:pending=True" in events
    assert not (install / "venv.pending-backup").exists()


def test_runtime_verification_executes_outside_the_venv_it_can_replace(install: Path):
    code, output = _stage(install, "dependencies", "sqlite-repaired")
    assert code == 0, output
    events = (install.parent.parent / "validation-events.txt").read_text()
    assert "sqlite:external=True" in events
    assert events.index("sqlite:pending=True") < events.rindex("baseline:pending=True")


def test_repaired_interpreter_survives_later_venv_recreation(install: Path):
    assert _stage(install, "dependencies", "sqlite-repaired")[0] == 0
    code, output = _stage(install, "venv", "sqlite-repaired")
    assert code == 0, output
    native = (install.parent.parent / "native-events.txt").read_text()
    assert "operation:uv-venv-repaired-python" in native

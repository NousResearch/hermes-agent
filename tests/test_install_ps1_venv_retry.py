"""Native stage/exit/filesystem contracts; Python/package responses are synthetic.

Preserves egilewski's transaction contracts from #83149/#83194 and builds on
fangliquanflq's interrupted-retry recovery in #103771.
"""

from pathlib import Path
import shutil

import pytest

from tests.windows_installer_fixtures import (
    BACKUP_NAME, ORIGINAL, PARTIAL, VALIDATED,
    _generation, _junction, _pending, _stage,
    fake_uv as fake_uv,
    install as install,
    powershell_host as powershell_host,
)

pytestmark = pytest.mark.windows_only


@pytest.mark.parametrize("failure", ["deps-fail", "import-fail"])
def test_separate_stage_retry_restores_original_after_dependency_failure(install: Path, failure: str):
    assert _stage(install, "venv")[0] == 0
    first_backup = install / _pending(install)
    assert _generation(first_backup) == ORIGINAL
    assert _stage(install, "venv")[0] == 0
    assert _stage(install, "dependencies", failure)[0] != 0
    if failure == "import-fail":
        validation = (install.parent.parent / "validation-events.txt").read_text()
        assert "baseline:pending=True" in validation
    assert _generation(install / "venv") == ORIGINAL
    assert not (install / "venv.pending-backup").exists()


def test_retry_reconciles_before_managed_python_resolution_can_fail(install: Path):
    assert _stage(install, "venv")[0] == 0
    assert _stage(install, "venv", "find-fail")[0] != 0
    assert _generation(install / "venv") == ORIGINAL
    assert not (install / "venv.pending-backup").exists()


def test_retry_creation_failure_keeps_the_original_working_generation(install: Path):
    assert _stage(install, "venv")[0] == 0
    assert _stage(install, "venv", "venv-fail")[0] != 0
    assert _generation(install / "venv") == ORIGINAL
    assert not (install / "venv.pending-backup").exists()


@pytest.mark.parametrize("marker", ["", "../outside", "venv", "unowned-folder", BACKUP_NAME + "\nother"])
def test_invalid_pending_marker_fails_without_mutating_install(install: Path, marker: str):
    marker_path = install / "venv.pending-backup"
    marker_path.write_text(marker, encoding="ascii")
    code, _ = _stage(install, "venv")
    assert code != 0
    assert _generation(install / "venv") == ORIGINAL
    assert marker_path.read_text(encoding="ascii") == marker
    assert not (install.parent.parent / "native-events.txt").exists(), "invalid ownership must fail before provisioning"


def test_missing_pending_source_and_live_venv_preserves_recovery_evidence(install: Path):
    shutil.rmtree(install / "venv")
    marker = install / "venv.pending-backup"
    marker.write_text(BACKUP_NAME, encoding="ascii")
    assert _stage(install, "venv")[0] != 0
    assert marker.read_text(encoding="ascii") == BACKUP_NAME
    assert not (install / "venv").exists()


@pytest.mark.parametrize("checkpoint,code", [("crash-before-park", 91), ("crash-after-park", 92)])
def test_crash_checkpoint_retains_durable_original_recovery(install: Path, checkpoint: str, code: int):
    assert _stage(install, "venv", checkpoint)[0] == code
    backup = install / _pending(install)
    original_location = install / "venv" if checkpoint == "crash-before-park" else backup
    assert _generation(original_location) == ORIGINAL
    assert _stage(install, "venv")[0] == 0
    assert _stage(install, "dependencies", "deps-fail")[0] != 0
    assert _generation(install / "venv") == ORIGINAL


@pytest.mark.parametrize("failure", ["rollback-park-fail", "rollback-restore-fail"])
def test_failed_rollback_keeps_backup_and_marker_for_the_next_process(install: Path, failure: str):
    assert _stage(install, "venv")[0] == 0
    backup_name = _pending(install)
    assert _stage(install, "dependencies", failure)[0] != 0
    assert _pending(install) == backup_name
    assert _generation(install / backup_name) == ORIGINAL
    assert _stage(install, "dependencies", "deps-fail")[0] != 0
    assert _generation(install / "venv") == ORIGINAL


def test_marker_publication_failure_never_loses_original(install: Path):
    assert _stage(install, "venv", "marker-write-fail")[0] != 0
    assert _generation(install / "venv") == ORIGINAL


def test_crash_after_restore_preserves_original_and_reconciles_marker(install: Path):
    assert _stage(install, "venv")[0] == 0
    backup = install / _pending(install)
    assert _stage(install, "dependencies", "crash-after-restore")[0] in (-1, 4294967295)
    assert _generation(install / "venv") == ORIGINAL
    assert not backup.exists()
    assert (install / "venv.pending-backup").exists()
    assert _stage(install, "venv", "find-fail")[0] != 0
    assert _generation(install / "venv") == ORIGINAL
    assert not (install / "venv.pending-backup").exists()


def test_pending_marker_directory_is_not_treated_as_missing(install: Path):
    marker = install / "venv.pending-backup"
    marker.mkdir()
    assert _stage(install, "venv")[0] != 0
    assert marker.is_dir()
    assert _generation(install / "venv") == ORIGINAL
    assert not (install.parent.parent / "native-events.txt").exists()


def test_pending_backup_junction_cannot_redirect_recovery(install: Path):
    root = install.parent.parent
    target = root / "unrelated-environment"
    target.mkdir()
    (target / "generation.txt").write_text("UNRELATED_ENVIRONMENT", encoding="utf-8")
    link = install / BACKUP_NAME
    _junction(install, link, target)
    marker = install / "venv.pending-backup"
    marker.write_text(BACKUP_NAME, encoding="ascii")
    assert _stage(install, "venv")[0] != 0
    assert marker.read_text(encoding="ascii") == BACKUP_NAME
    assert _generation(target) == "UNRELATED_ENVIRONMENT"
    assert _generation(install / "venv") == ORIGINAL
    assert not (root / "native-events.txt").exists()


def test_commit_clears_pending_before_backup_cleanup_after_imports(install: Path):
    assert _stage(install, "venv")[0] == 0
    backup = install / _pending(install)
    assert _stage(install, "dependencies", "crash-before-commit-cleanup")[0] == 93
    assert _generation(install / "venv") == VALIDATED
    assert not (install / "venv.pending-backup").exists(), "a committed replacement must not remain eligible for rollback"
    assert _generation(backup) == ORIGINAL
    validation = (install.parent.parent / "validation-events.txt").read_text()
    assert "baseline:pending=True" in validation, "backup must remain pending until imports pass"
    assert _stage(install, "dependencies", "deps-fail")[0] != 0
    assert _generation(install / "venv") == VALIDATED
    assert _generation(backup) == ORIGINAL


def test_venv_stage_preserves_unowned_stale_backup(install: Path):
    unrelated = install / BACKUP_NAME
    unrelated.mkdir()
    (unrelated / "generation.txt").write_text("UNOWNED_BACKUP", encoding="utf-8")
    assert _stage(install, "venv")[0] == 0
    assert _generation(unrelated) == "UNOWNED_BACKUP", "venv creation must not sweep unrelated recovery evidence"


def test_successful_dependency_validation_commits_only_the_owned_backup(install: Path):
    assert _stage(install, "venv")[0] == 0
    backup = install / _pending(install)
    # Seed after venv creation to isolate dependency commit from the venv sweep.
    unrelated = install / BACKUP_NAME
    unrelated.mkdir()
    (unrelated / "generation.txt").write_text("UNOWNED_BACKUP", encoding="utf-8")
    code, output = _stage(install, "dependencies")
    assert code == 0, output
    assert _generation(install / "venv") == VALIDATED
    assert not (install / "venv.pending-backup").exists()
    assert not backup.exists()
    assert _generation(unrelated) == "UNOWNED_BACKUP"


def test_no_venv_does_not_consume_pending_recovery(install: Path):
    assert _stage(install, "venv")[0] == 0
    pending_before = _pending(install)
    assert _stage(install, "venv", no_venv=True)[0] == 0
    assert _pending(install) == pending_before
    assert _generation(install / pending_before) == ORIGINAL
    assert _generation(install / "venv") == PARTIAL


@pytest.mark.parametrize("mode,expected_success", [("ok", True), ("deps-fail", False)])
def test_no_venv_dependencies_preserve_pending_original(install: Path, mode: str, expected_success: bool):
    assert _stage(install, "venv")[0] == 0
    pending_before = _pending(install)
    code, output = _stage(install, "dependencies", mode, no_venv=True)
    assert (code == 0) == expected_success, output
    assert _pending(install) == pending_before
    assert _generation(install / pending_before) == ORIGINAL


def test_locked_sync_retains_backup_through_validation_then_commits(install: Path):
    assert _stage(install, "venv")[0] == 0
    backup = install / _pending(install)
    (install / "uv.lock").write_text("# synthetic resolver boundary fixture\n", encoding="utf-8")
    code, output = _stage(install, "dependencies")
    assert code == 0, output
    events = (install.parent.parent / "native-events.txt").read_text(encoding="utf-8")
    assert "operation:uv-sync" in events
    assert "operation:uv-pip-install" not in events
    assert "baseline:pending=True" in (install.parent.parent / "validation-events.txt").read_text()
    assert _generation(install / "venv") == VALIDATED
    assert not backup.exists()
    assert not (install / "venv.pending-backup").exists()


def test_dashboard_syntax_failure_after_baseline_restores_original(install: Path):
    assert _stage(install, "venv")[0] == 0
    code, output = _stage(install, "dependencies", "dashboard-syntax-fail")
    assert code != 0, output
    validation = (install.parent.parent / "validation-events.txt").read_text()
    assert "baseline:pending=True" in validation
    assert "dashboard-syntax:pending=True" in validation
    assert _generation(install / "venv") == ORIGINAL
    assert not (install / "venv.pending-backup").exists()


def test_optional_web_install_failure_warns_and_commits(install: Path):
    assert _stage(install, "venv")[0] == 0
    backup = install / _pending(install)
    code, output = _stage(install, "dependencies", "optional-web-fail")
    assert code == 0, output
    assert "Could not install [web] extra" in output
    events = (install.parent.parent / "native-events.txt").read_text(encoding="utf-8")
    assert "operation:dashboard-import" in events
    assert "operation:uv-web-repair" in events
    assert _generation(install / "venv") == VALIDATED
    assert not backup.exists()
    assert not (install / "venv.pending-backup").exists()


@pytest.mark.parametrize("mode", ["ok", "venv-fail"])
def test_first_install_has_no_prior_generation_to_recover(install: Path, mode: str):
    shutil.rmtree(install / "venv")
    code, output = _stage(install, "venv", mode)
    assert (code == 0) == (mode == "ok"), output
    assert not (install / "venv.pending-backup").exists()
    if mode == "ok":
        assert _stage(install, "dependencies")[0] == 0
        assert _generation(install / "venv") == VALIDATED
    else:
        assert not (install / "venv").exists()
        assert any(_generation(path) == PARTIAL for path in install.glob("venv.failed.*"))


@pytest.mark.parametrize("mode", ["commit-marker-clear-fail", "rollback-marker-clear-fail"])
def test_marker_clear_failure_retains_original_and_retryable_evidence(install: Path, mode: str):
    assert _stage(install, "venv")[0] == 0
    pending = _pending(install)
    code, output = _stage(install, "dependencies", mode)
    assert code != 0, output
    assert "Rollback failed" in output
    assert _generation(install / "venv") == ORIGINAL
    assert _pending(install) == pending
    assert not (install / pending).exists(), "the original has already been restored"
    assert _stage(install, "venv", "find-fail")[0] != 0
    assert _generation(install / "venv") == ORIGINAL
    assert not (install / "venv.pending-backup").exists()


def test_committed_backup_with_child_junction_is_retained_without_traversal(install: Path):
    assert _stage(install, "venv")[0] == 0
    backup = install / _pending(install)
    target = install.parent.parent / "unrelated-junction-target"
    target.mkdir()
    (target / "generation.txt").write_text("UNRELATED_ENVIRONMENT", encoding="utf-8")
    _junction(install, backup / "external", target)
    code, output = _stage(install, "dependencies")
    assert code == 0, output
    assert "reparse point" in output
    assert not (install / "venv.pending-backup").exists()
    assert _generation(install / "venv") == VALIDATED
    assert _generation(backup) == ORIGINAL
    assert _generation(target) == "UNRELATED_ENVIRONMENT"
    assert (backup / "external").is_dir()


def test_final_baseline_probe_rejects_dependency_changes_from_repair(install: Path):
    assert _stage(install, "venv")[0] == 0
    (install / "venv/Scripts/hermes.exe").unlink()
    code, output = _stage(install, "dependencies", "final-import-fail")
    assert code != 0, output
    assert "Reinstalling entry points" in output
    events = (install.parent.parent / "validation-events.txt").read_text()
    assert events.count("baseline:pending=True") == 2, "probe before and after the repair"
    # Windows may still hold the just-exited native executable during rollback.
    # The original must survive either restored or parked with its pointer, and
    # the next invocation must reconcile it before interpreter discovery fails.
    candidates = [install / "venv"]
    if (install / "venv.pending-backup").exists():
        candidates.append(install / _pending(install))
    assert any(path.is_dir() and _generation(path) == ORIGINAL for path in candidates)
    assert _stage(install, "venv", "find-fail")[0] != 0
    assert _generation(install / "venv") == ORIGINAL
    assert not (install / "venv.pending-backup").exists()


def test_success_without_a_venv_interpreter_restores_the_original(install: Path):
    code, output = _stage(install, "venv", "venv-no-interpreter")
    assert code != 0, output
    assert "interpreter is missing" in output
    assert _generation(install / "venv") == ORIGINAL
    assert not (install / "venv.pending-backup").exists()


def test_non_ascii_marker_encoding_is_rejected_before_provisioning(install: Path):
    marker = install / "venv.pending-backup"
    contents = b"\xef\xbb\xbf" + BACKUP_NAME.encode("ascii")
    marker.write_bytes(contents)
    code, output = _stage(install, "venv")
    assert code != 0, output
    assert marker.read_bytes() == contents
    assert _generation(install / "venv") == ORIGINAL
    assert not (install.parent.parent / "native-events.txt").exists()

"""Transaction-boundary regression for Windows venv recreation (#83149).

Review finding on PR #83194 (egilewski): the rollback source (the parked
previous venv) was deleted as soon as ``Install-Venv`` saw a working
interpreter in the replacement — but ``Install-Dependencies`` is a separate,
later stage (a separate *process* under the stage-per-process bootstrap) and
every dependency tier or the baseline-import gate can still fail after that
point. Deleting the backup early re-creates exactly the availability failure
the transactional recreate exists to prevent.

The contract locked here:

* ``Install-Venv`` records the parked backup in ``venv.pending-backup``
  instead of deleting it, and its stale-tree sweep excludes that backup.
* ``Install-Dependencies`` restores the previous venv on failure
  (``Restore-VenvBackup``) and commits the cleanup only after the
  baseline-import gate passes (``Complete-VenvTransaction``).

The script only runs on Windows, so Linux CI locks the contract at the
source level, same approach as tests/test_install_ps1_venv_recreate_safety.py.
"""

from pathlib import Path

INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


def _function_body(source: str, function_name: str) -> str:
    start = source.index(f"function {function_name}")
    opening_brace = source.index("{", start)
    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace : index + 1]
    raise AssertionError(f"unterminated function: {function_name}")


def _source() -> str:
    return INSTALL_PS1.read_text(encoding="ascii")


def test_install_venv_does_not_delete_backup_before_dependency_stage() -> None:
    """The parked previous venv must survive Install-Venv's success path."""
    body = _function_body(_source(), "Install-Venv")

    # The success path records the rollback source instead of deleting it.
    assert "venv.pending-backup" in body
    # The only backup deletion allowed inside Install-Venv is the *rollback*
    # rename in the catch block; a Remove-Item of the backup must not appear.
    assert "Remove-Item -LiteralPath $venvBackupName" not in body


def test_install_venv_stale_sweep_excludes_current_backup() -> None:
    """The venv.stale.* sweep must not delete this run's rollback source."""
    body = _function_body(_source(), "Install-Venv")

    sweep_at = body.index('Get-ChildItem -Directory -Filter "venv.stale.*"')
    window = body[sweep_at : sweep_at + 400]
    assert "$_.Name -ne $venvBackupName" in window, (
        "the stale-tree sweep must exclude the backup parked by this run"
    )


def test_install_dependencies_restores_backup_on_failure() -> None:
    """A failed dependency tier or import gate must restore the parked venv."""
    body = _function_body(_source(), "Install-Dependencies")

    assert "Restore-VenvBackup" in body
    catch_at = body.index("Restore-VenvBackup")
    assert "throw" in body[catch_at : catch_at + 400], (
        "rollback must rethrow the original failure after restoring"
    )


def test_install_dependencies_commits_only_after_import_gate() -> None:
    """Backup cleanup must come after the baseline-import verification."""
    body = _function_body(_source(), "Install-Dependencies")

    import_gate = body.index("Baseline imports verified in venv")
    commit = body.index("Complete-VenvTransaction")
    assert import_gate < commit, (
        "the venv transaction must commit only after imports prove the "
        "replacement usable"
    )


def test_restore_helper_parks_failed_replacement_and_restores_previous() -> None:
    body = _function_body(_source(), "Restore-VenvBackup")

    park = body.index("venv.failed.")
    restore = body.index('-NewName "venv"')
    assert park < restore, (
        "the failed replacement must be parked before the previous venv is "
        "renamed back into place"
    )


def test_commit_helper_deletes_backup_and_clears_marker() -> None:
    body = _function_body(_source(), "Complete-VenvTransaction")

    assert "Remove-Item" in body
    assert "venv.pending-backup" in body

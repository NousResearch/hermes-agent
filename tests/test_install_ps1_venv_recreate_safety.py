"""Regression tests for transactional Windows venv recreation (#83149).

The installer must never delete the live venv in place. Windows can remove
unlocked files before it reaches a locked interpreter or native extension,
leaving a half-installed venv that the next health/blocker probe cannot use.
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


def _install_venv_body() -> str:
    return _function_body(INSTALL_PS1.read_text(encoding="ascii"), "Install-Venv")


def test_rename_failure_cannot_fall_back_to_destructive_delete() -> None:
    body = _install_venv_body()

    assert "falling back to in-place delete" not in body.lower()
    assert 'Remove-Item -Recurse -Force "venv"' not in body


def test_manual_recovery_hints_do_not_delete_live_venv() -> None:
    source = INSTALL_PS1.read_text(encoding="ascii")

    assert "Remove-Item -Recurse -Force venv" not in source
    assert "Do not delete venv in place" in source


def test_previous_venv_survives_until_replacement_is_ready() -> None:
    body = _install_venv_body()

    create = body.index("& $UvCmd venv venv")
    stale_cleanup = body.index('Get-ChildItem -Directory -Filter "venv.stale.*"')
    assert create < stale_cleanup, (
        "stale backups must not be cleaned before uv can succeed or rollback"
    )


def test_recreate_restores_parked_venv_after_failure() -> None:
    body = _install_venv_body()

    failure = body.index("Failed to create virtual environment")
    restore = body.index(
        'Rename-Item -LiteralPath $venvBackupName -NewName "venv"'
    )
    assert failure < restore
    assert "rollback failed" in body.lower()


def test_recreate_rejects_success_without_venv_interpreter() -> None:
    body = _install_venv_body()

    missing_python = body.index("$venvPythonExe")
    success = body.index("Virtual environment ready")
    assert "Test-Path -LiteralPath $venvPythonExe -PathType Leaf" in body[
        missing_python:success
    ]
    assert "throw" in body[missing_python:success]

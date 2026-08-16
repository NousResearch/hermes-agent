"""Regression: Windows installer must not gut the venv on rename failure (#83149).

When recreating an existing venv, ``Install-Venv`` renames the directory aside
first. An older fallback deleted the tree in place when rename was denied; a
partial ``Remove-Item`` could wipe most of ``site-packages`` and then fail on
one locked ``.pyd``, leaving Hermes unusable with no rollback.

These tests lock the contract at the source level (the script only runs on
Windows, so Linux CI cannot execute the PowerShell path).
"""

from pathlib import Path

import pytest

_INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


@pytest.fixture(scope="module")
def source() -> str:
    return _INSTALL_PS1.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    """Return the text of a PowerShell ``function <name> { ... }`` block."""
    start = source.index(f"function {name}")
    brace = source.index("{", start)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[brace : i + 1]
    raise AssertionError(f"unterminated function body for {name}")


def test_install_venv_aborts_when_rename_aside_fails(source: str):
    """Rename failure must throw with the previous install left intact."""
    body = _function_body(source, "Install-Venv")
    assert (
        "Rename-Item -LiteralPath \"venv\"" in body
        or "Rename-Item -LiteralPath 'venv'" in body
    )
    assert "falling back to in-place delete" not in body
    throw_at = body.find("Could not move the existing venv aside")
    assert throw_at != -1, "rename failure must abort with an actionable error"
    assert "throw" in body[max(0, throw_at - 80) : throw_at + 40]


def test_install_venv_never_removes_live_venv_in_place_on_rename_fail(source: str):
    """After a failed rename, Install-Venv must not Remove-Item the live venv.

    Parked ``venv.stale.*`` trees may still be deleted best-effort; the live
    ``venv`` directory must only disappear via Rename-Item.
    """
    body = _function_body(source, "Install-Venv")
    # The only Remove-Item targeting the active tree used to be the fallback:
    #   Remove-Item -Recurse -Force "venv"
    # That path must be gone. Stale-parking cleanup uses $staleName / filter.
    assert 'Remove-Item -Recurse -Force "venv"' not in body
    assert "Remove-Item -Recurse -Force 'venv'" not in body
    assert "venv.stale." in body

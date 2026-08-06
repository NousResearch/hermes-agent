"""Regression: the Windows installer must not spawn a bare ``powershell``.

A user on Windows reported the installer getting stuck; running
``irm https://hermes-agent.nousresearch.com/install.ps1 | iex`` failed at the
uv step with::

    [X] Failed to install uv: The term 'powershell' is not recognized as the
        name of a cmdlet, function, script file, or operable program.

Root cause: ``Install-Uv`` spawned the astral uv installer via a hardcoded
bare ``powershell`` command.  That name resolves only to *Windows PowerShell*
and only when its System32 directory is on ``PATH``.  Under PowerShell 7+
(``pwsh``) -- or any session where ``powershell`` isn't on ``PATH`` -- the
spawn dies and uv installation aborts.

The fix resolves the PowerShell host executable (preferring the absolute path
of the running host, then ``powershell``/``pwsh`` via ``Get-Command``) and
invokes *that* instead of a bare name.  These tests lock that contract at the
source level (the script only runs on Windows, so there's no runner to
execute it on Linux CI).
"""

from pathlib import Path

import pytest

_INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


@pytest.fixture(scope="module")
def source() -> str:
    return _INSTALL_PS1.read_text(encoding="utf-8")


def test_astral_uv_installer_not_spawned_via_bare_powershell(source: str):
    """The exact failing literal must be gone."""
    forbidden = 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv'
    assert forbidden not in source, (
        "Install-Uv still spawns the astral uv installer via a bare "
        "`powershell` — it must use the resolved PowerShell host exe so it "
        "works under pwsh / when powershell isn't on PATH."
    )


def test_astral_uv_installer_invoked_via_resolved_host_variable(source: str):
    """The astral uv installer line must use the call operator on a variable.

    i.e. ``& $psHostExe -ExecutionPolicy ... irm https://astral.sh/uv...``
    rather than naming a fixed executable.
    """
    lines = [ln for ln in source.splitlines() if "astral.sh/uv/install.ps1 | iex" in ln]
    # Exactly one invocation line carries the astral installer.
    invocation = [ln for ln in lines if "irm https://astral.sh/uv/install.ps1 | iex" in ln]
    assert invocation, "astral uv install invocation line not found"
    for ln in invocation:
        stripped = ln.strip()
        assert stripped.startswith("& $"), (
            f"astral uv installer must be invoked via the call operator on a "
            f"resolved host variable (`& $...`), got: {stripped!r}"
        )


def test_powershell_host_resolver_is_defined_and_portable(source: str):
    """A host-resolver helper must exist and be PATH-independent + pwsh-aware."""
    assert "function Get-PowerShellHostExe" in source, (
        "expected a Get-PowerShellHostExe helper that resolves the host exe"
    )
    # PATH-independent: derive the absolute path of the running host.
    assert "Get-Process -Id $PID" in source, (
        "resolver must derive the current host's absolute path "
        "(Get-Process -Id $PID), which is independent of PATH"
    )
    # pwsh-aware fallback: PowerShell 7's executable is `pwsh`, not `powershell`.
    assert "pwsh" in source, (
        "resolver must fall back to pwsh (PowerShell 7) when powershell is "
        "unavailable"
    )

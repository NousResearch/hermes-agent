"""install.ps1 must not write the operator's real ``HKCU\\Environment\\Path``.

The fourth persistent-PATH write point
--------------------------------------
``scripts/install.ps1::Set-LauncherUserPath`` prepends ``$HermesHome\\bin`` to the
*persisted* User PATH. It is reached by the real ``products`` stage
(``Stage-Products`` -> ``Publish-UserCommand`` -> ``Set-LauncherUserPath``), and
test isolation redirects ``HERMES_HOME`` to a per-session sandbox -- so under
pytest ``$HermesHome\\bin`` is a throwaway ``<tmp>\\...\\hermes-home\\bin``.

Unlike the three python write points
(``hermes_cli/_launchers._register_windows_user_path``,
``hermes_cli/_install_repair._write_user_path_raw``,
``hermes_cli/uninstall.remove_path_from_windows_registry``) this one goes through
``[Environment]::SetEnvironmentVariable(..., "User")``, and that makes it *lossy*
on top of leaking:

* it writes the value back as ``REG_SZ``, downgrading a stored
  ``REG_EXPAND_SZ``;
* .NET *expands* ``%VAR%`` when it reads the ``User`` target, so every expandable
  entry is frozen to a literal on the way back in.

Measured 2026-10-10 (card ``t_f674de17``, full-suite batch ``b021``): the real
value went from 663 chars / 15 entries / ``REG_EXPAND_SZ``
(sha256 prefix ``4952468110176a38``) to 802 chars / 16 entries / ``REG_SZ``
(``e6783f8292b3eb60``), with ``%USERPROFILE%\\.dotnet\\tools`` frozen to a literal
-- while the triggering test (``test_install_ps1_desktop_stage.py``) stayed 5✓
green, because its boundary wrapper intercepted icacls / ie4uinit / WScript.Shell
but *not the registry*.

Two arms, one boundary
----------------------
The registry choke point is a named function (``Set-UserPathValue``) precisely so
a wrapper can replace it: a .NET static setter cannot be intercepted from
PowerShell, and a boundary that cannot intercept the write cannot honestly claim
it does not touch the operator's real PATH. On top of that interception both arms
bracket the *real* ``HKCU\\Environment\\Path`` (raw value **and** registry type)
and require it byte-identical -- that bracket is what actually catches a tree
without the guard, where the seam does not exist and the write lands for real.

* ``test_install_ps1_user_path_is_inert_under_test_isolation`` -- the marker is
  present: zero writes, real PATH unchanged. **Red** on a tree without the guard.
* ``test_install_ps1_user_path_write_survives_without_the_marker`` -- the guard
  must not over-block: without the marker the write still happens, with the same
  payload the unguarded code produced, and the real PATH is still untouched.

``platforms("windows")`` rather than a bare ``skipif``: the OS lanes import the
files ``scripts/ci/list_os_marked_tests.py`` lists for their marker, so a bare
skipif would leave these running on no host (and would put ``winreg`` in the
module header, making the file a collection error off Windows).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("windows")

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"

# Dot-source the real install.ps1 (definitions only -- it returns before any work
# when dot-sourced), then replace the registry choke point. The override is
# defined AFTER the dot-source, the same way the desktop-stage wrapper replaces
# ``Get-BootstrapPython``: a later definition in the same session wins.
#
# ``$read`` is logged separately so the write payload can be compared against what
# the code under test itself read -- ``[Environment]`` expands ``%VAR%``, so the
# payload is deliberately NOT the raw registry value.
_WRAPPER = r'''
param(
    [Parameter(Mandatory = $true)][string]$InstallerPath,
    [Parameter(Mandatory = $true)][string]$BinDir
)
$ErrorActionPreference = "Stop"

. $InstallerPath

function Set-UserPathValue([string]$value) {
    Add-Content -Path $env:USERPATH_LOG -Value $value
}

$read = [Environment]::GetEnvironmentVariable("Path", "User")
Set-LauncherUserPath $BinDir
Add-Content -Path $env:USERPATH_READ_LOG -Value $read
exit 0
'''


def _raw_user_path() -> tuple[str, int]:
    """The stored value AND its type -- never the ``%VAR%``-expanded form.

    Same reading as ``tests/hermes_cli/test_windows_user_path_isolation.py``:
    comparing expanded values would hide a ``REG_EXPAND_SZ`` -> ``REG_SZ``
    downgrade, which is half of this write point's damage.
    """
    import winreg  # lazy: the module must stay importable off Windows

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ
    ) as key:
        value, kind = winreg.QueryValueEx(key, "Path")
    return str(value), int(kind)


def _describe(snapshot: tuple[str, int]) -> str:
    value, kind = snapshot
    name = {1: "REG_SZ", 2: "REG_EXPAND_SZ"}.get(kind, str(kind))
    return f"{len(value)} chars / {len([e for e in value.split(';') if e])} entries / {name}"


def _run_launcher_user_path(
    tmp_path: Path, bin_dir: Path, *, isolated: bool
) -> tuple[subprocess.CompletedProcess, list[str], list[str]]:
    """Drive the real ``Set-LauncherUserPath`` once, with the write intercepted.

    Returns ``(run, written, read)``. ``written`` is empty when the code under
    test never reached the choke point -- either because the guard returned
    early (the point of the red arm) or because the tree has no seam at all (in
    which case the real-registry bracket is the only thing that can see it).
    """
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")

    write_log = tmp_path / "user-path-writes.log"
    read_log = tmp_path / "user-path-read.log"
    wrapper = tmp_path / "user-path-wrapper.ps1"
    wrapper.write_text(_WRAPPER, encoding="utf-8-sig")

    env = {
        **os.environ,
        "USERPATH_LOG": str(write_log),
        "USERPATH_READ_LOG": str(read_log),
    }
    if isolated:
        # The hermetic conftest's subprocess-surviving marker, set explicitly so
        # this arm does not depend on who exported it.
        env["HERMES_TEST_ISOLATION"] = "1"
    else:
        # ...and explicitly cleared for the other arm: conftest exports it into
        # os.environ, so inheriting it would silently test the same arm twice.
        env.pop("HERMES_TEST_ISOLATION", None)

    run = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(wrapper),
         "-InstallerPath", str(INSTALL_PS1),
         "-BinDir", str(bin_dir)],
        cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, check=False, timeout=180,
    )

    def _lines(path: Path) -> list[str]:
        if not path.exists():
            return []
        return [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line]

    return run, _lines(write_log), _lines(read_log)


def _assert_real_path_untouched(before: tuple[str, int], bin_dir: Path) -> None:
    after = _raw_user_path()
    assert after == before, (
        f"{bin_dir} leaked into the operator's persisted User PATH "
        f"(HKCU\\Environment\\Path): {_describe(before)} -> {_describe(after)}"
    )


def test_install_ps1_user_path_is_inert_under_test_isolation(tmp_path: Path) -> None:
    """RED LIGHT: with the isolation marker set, zero registry writes.

    Fails on a tree without the guard -- and it fails on the *real registry*
    comparison, because such a tree has no ``Set-UserPathValue`` to intercept, so
    the throwaway ``<tmp>\\...\\bin`` really lands in the operator's PATH (that is
    the 2026-10-10 incident, reproduced by this very assertion).
    """
    bin_dir = tmp_path / "hermes-home" / "bin"
    bin_dir.mkdir(parents=True)
    before = _raw_user_path()

    run, written, _read = _run_launcher_user_path(tmp_path, bin_dir, isolated=True)

    assert run.returncode == 0, run.stdout + run.stderr
    assert written == [], (
        f"install.ps1 wrote the user PATH under test isolation: {written!r}"
    )
    _assert_real_path_untouched(before, bin_dir)


def test_install_ps1_user_path_write_survives_without_the_marker(tmp_path: Path) -> None:
    """GREEN LIGHT: the guard must not over-block a real install.

    Without the marker the write point still fires, with the same payload the
    unguarded code produced: ``<binDir>;<what the function read>``. The real
    registry is compared too -- the interception is the only reason this arm can
    assert "it wrote" without writing.
    """
    bin_dir = tmp_path / "hermes-home" / "bin"
    bin_dir.mkdir(parents=True)
    before = _raw_user_path()

    run, written, read = _run_launcher_user_path(tmp_path, bin_dir, isolated=False)

    assert run.returncode == 0, run.stdout + run.stderr
    assert len(read) == 1, f"wrapper failed to record what it read: {read!r}"
    assert len(written) == 1, (
        f"the user-PATH write point never fired without the marker "
        f"(guard over-blocking, or the write is no longer a named function): {written!r}"
    )
    # The payload is the unguarded code's own payload: prepend, then the value
    # the function read (the %VAR%-expanded form -- .NET expands on read).
    assert written[0] == f"{bin_dir};{read[0]}", written
    _assert_real_path_untouched(before, bin_dir)

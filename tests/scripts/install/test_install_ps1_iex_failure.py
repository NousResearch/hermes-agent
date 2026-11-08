"""A failed install run through `iex (irm ...)` must not end the caller's session.

`iex` runs the installer text inside the user's PowerShell session, so an
`exit` on failure closed their window. A script file run keeps its exit code.
"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.platforms("windows")
INSTALLER = Path(__file__).resolve().parents[3] / "scripts" / "install.ps1"
REFUSAL = "exists and is not a Hermes git checkout"
GIT_DOWNLOAD = "installing git"

# Get-PinnedGit keeps a staged slot only when `git --exec-path` names a dir
# holding git-remote-https; any other slot is deleted and the real PortableGit
# downloaded, which alone can outlast the timeout. This git answers that probe.
_FAKE_GIT = r'''
using System;

public static class FakeGit {
    public static int Main(string[] args) {
        if (args.Length == 1 && args[0] == "--exec-path") {
            Console.WriteLine(AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\'));
            return 0;
        }
        Console.Error.WriteLine("fake git: unexpected call: " + string.Join(" ", args));
        return 1;
    }
}
'''


@pytest.fixture(scope="module")
def fake_git(tmp_path_factory):
    out = tmp_path_factory.mktemp("fake-git")
    source = out / "fake_git.cs"
    source.write_text(_FAKE_GIT, encoding="utf-8")
    exe = out / "git.exe"
    compile_script = out / "compile.ps1"
    compile_script.write_text(
        "param([string]$Source, [string]$Output)\n"
        "Add-Type -Path $Source -OutputAssembly $Output -OutputType ConsoleApplication\n",
        encoding="utf-8")
    subprocess.run([_powershell(), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                    "-File", str(compile_script), "-Source", str(source), "-Output", str(exe)],
                   check=True, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    return exe


def _failing_install_env(tmp_path, fake_git):
    """An occupied non-checkout makes the repository stage Fail before any
    network or git call; a ready pinned-git slot satisfies prerequisites."""
    home = tmp_path / "home"
    (home / "hermes-agent").mkdir(parents=True)
    (home / "hermes-agent" / "user-file").write_text("preserve me", encoding="utf-8")
    tools = tmp_path / "tools"
    for arch in ("x64", "arm64"):
        cmd = tools / f"git-2.53.0+3-win32-{arch}" / "cmd"
        cmd.mkdir(parents=True)
        shutil.copyfile(fake_git, cmd / "git.exe")
        (cmd / "git-remote-https.exe").write_bytes(b"")
    return dict(os.environ, HERMES_HOME=str(home), HERMES_RUNTIME_DIR=str(tools))


def _powershell():
    powershell = shutil.which("powershell")
    assert powershell
    return powershell


def test_failed_iex_install_reports_and_returns_to_the_callers_session(tmp_path, fake_git):
    command = (f"iex (Get-Content -Raw -LiteralPath '{INSTALLER}'); "
               "Write-Output \"session alive: $LASTEXITCODE\"")
    result = subprocess.run([_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
                            env=_failing_install_env(tmp_path, fake_git), capture_output=True, text=True,
                            timeout=60, check=False)
    assert GIT_DOWNLOAD not in result.stdout, result.stdout
    assert REFUSAL in result.stdout
    assert "session alive: 1" in result.stdout, result.stdout + result.stderr
    assert (tmp_path / "home" / "hermes-agent" / "user-file").read_text(encoding="utf-8") == "preserve me"


def test_failed_file_install_still_exits_nonzero(tmp_path, fake_git):
    result = subprocess.run([_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(INSTALLER)],
                            env=_failing_install_env(tmp_path, fake_git), capture_output=True, text=True,
                            timeout=60, check=False)
    assert GIT_DOWNLOAD not in result.stdout, result.stdout
    assert REFUSAL in result.stdout
    assert result.returncode == 1

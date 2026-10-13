"""Execute SSH mutation scripts with a real local Bash transport seam."""
from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli.ssh_workspace_fs import SshWorkspaceFs


def _shell_path(path: Path) -> str:
    value = path.as_posix()
    if path.drive:
        return f"/{path.drive[0].lower()}{value[len(path.drive):]}"
    return value


@pytest.fixture
def shell_workspace(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required to execute SSH mutation scripts")

    class ShellTransport:
        cwd = _shell_path(tmp_path)
        _remote_home = cwd
        timeout = 15

        def execute(self, command, *, cwd=None, stdin_data=None, timeout=None):
            script = tmp_path / "transport.sh"
            script.write_bytes((f"cd -- {shlex.quote(cwd or self.cwd)}\n" + command).encode())
            result = subprocess.run(
                [bash, str(script)],
                input=stdin_data.encode() if stdin_data is not None else None,
                capture_output=True,
                timeout=timeout or self.timeout,
                check=False,
            )
            return {
                "returncode": result.returncode,
                "output": (result.stdout + result.stderr).decode("utf-8", errors="replace"),
            }

    return SshWorkspaceFs(ShellTransport())


def test_nonrecursive_file_delete_finishes_without_directory_removal(shell_workspace, tmp_path):
    target = tmp_path / "file with spaces.txt"
    shell_workspace.create(_shell_path(target), directory=False)
    assert target.exists()

    assert shell_workspace.delete(_shell_path(target), recursive=False) == _shell_path(target)
    assert not target.exists()


@pytest.mark.parametrize("recursive", [False, True])
def test_directory_create_and_delete_execute_quoted_target(shell_workspace, tmp_path, recursive):
    target = tmp_path / "folder with 'quotes' and spaces"
    assert shell_workspace.create(_shell_path(target), directory=True) == _shell_path(target)
    assert target.is_dir()
    if recursive:
        (target / "child.txt").write_bytes(b"remove")
    assert shell_workspace.delete(_shell_path(target), recursive=recursive) == _shell_path(target)
    assert not target.exists()


@pytest.mark.require_symlinks
@pytest.mark.parametrize("dangling", [False, True])
def test_nonrecursive_link_delete_preserves_its_referent(shell_workspace, tmp_path, dangling):
    protected = tmp_path / "protected"
    protected.mkdir()
    sentinel = protected / "sentinel.txt"
    sentinel.write_bytes(b"keep")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing" if dangling else protected, target_is_directory=True)

    shell_workspace.delete(_shell_path(link), recursive=False)

    assert not link.is_symlink()
    assert sentinel.read_bytes() == b"keep"

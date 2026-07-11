import os
import shlex
import subprocess
import sys
import shutil
from pathlib import Path

import pytest

from agent.claude_workspace_terminal import build_workspace_terminal_args


@pytest.mark.skipif(os.uname().sysname != "Darwin", reason="macOS sandbox-exec")
def test_workspace_terminal_denies_host_and_ambient_but_can_run_tests(tmp_path):
    host_home = tmp_path / "host"
    workspace = host_home / "worktree"
    workspace.mkdir(parents=True)
    (host_home / "sentinel").write_text("host-secret", encoding="utf-8")
    transformed = build_workspace_terminal_args(
        {
            "command": (
                f"test ! -r {shlex.quote(str(host_home / 'sentinel'))} && "
                'test -z "$AMBIENT_SENTINEL_SECRET" && '
                "printf passed > sandbox-proof.txt"
            )
        },
        workspace=workspace,
        host_home=host_home,
        exact_env={
            "HOME": str(host_home),
            "PATH": os.environ["PATH"],
            "USER": os.environ.get("USER", "worker"),
            "LOGNAME": os.environ.get("LOGNAME", "worker"),
        },
    )

    result = subprocess.run(
        ["/bin/bash", "-lc", transformed["command"]],
        cwd=workspace,
        env={**os.environ, "AMBIENT_SENTINEL_SECRET": "must-not-cross"},
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert (workspace / "sandbox-proof.txt").read_text(encoding="utf-8") == "passed"


@pytest.mark.skipif(os.uname().sysname != "Darwin", reason="macOS sandbox-exec")
def test_workspace_terminal_cannot_write_outside_or_follow_escape_symlink(tmp_path):
    workspace = tmp_path / "worktree"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    transformed = build_workspace_terminal_args(
        {
            "command": (
                f"touch {shlex.quote(str(outside / 'absolute.txt'))}; "
                "touch escape/symlink.txt"
            )
        },
        workspace=workspace,
        host_home=tmp_path / "host",
        exact_env={"PATH": os.environ["PATH"]},
    )

    subprocess.run(
        ["/bin/bash", "-lc", transformed["command"]],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert list(outside.iterdir()) == []


def test_workspace_terminal_preserves_process_controls(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    transformed = build_workspace_terminal_args(
        {"command": "echo ok", "timeout": 30, "background": True},
        workspace=workspace,
        host_home=tmp_path / "host",
        exact_env={"HOME": str(tmp_path / "host"), "PATH": "/usr/bin:/bin"},
        platform_name="Darwin",
    )

    assert transformed["timeout"] == 30
    assert transformed["background"] is True
    assert "sandbox-exec" in transformed["command"]
    argv = shlex.split(transformed["command"])
    profile_path = Path(argv[argv.index("-f") + 1])
    assert profile_path.is_file()
    assert not profile_path.is_relative_to(workspace.resolve())
    assert profile_path.stat().st_mode & 0o777 == 0o600
    assert "(deny default)" in profile_path.read_text(encoding="utf-8")
    assert "(allow default)" not in profile_path.read_text(encoding="utf-8")
    assert transformed["workdir"] == str(workspace.resolve())


def test_workspace_terminal_rejects_outside_workdir(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    with pytest.raises(RuntimeError, match="workdir is outside"):
        build_workspace_terminal_args(
            {"command": "pwd", "workdir": str(tmp_path / "outside")},
            workspace=workspace,
            host_home=tmp_path / "host",
            exact_env={"PATH": "/usr/bin:/bin"},
            platform_name="Darwin",
        )


def test_workspace_terminal_fails_closed_off_macos(tmp_path):
    with pytest.raises(RuntimeError, match="unsupported"):
        build_workspace_terminal_args(
            {"command": "echo no"},
            workspace=tmp_path,
            host_home=tmp_path / "host",
            exact_env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
            platform_name="Linux",
        )


def test_workspace_terminal_preflight_rejects_hardlinked_regular_file(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    os.link(outside, workspace / "alias.txt")

    with pytest.raises(RuntimeError, match="hard-linked"):
        build_workspace_terminal_args(
            {"command": "cat alias.txt"},
            workspace=workspace,
            host_home=tmp_path / "host",
            exact_env={"PATH": "/usr/bin:/bin"},
            platform_name="Darwin",
        )


@pytest.mark.skipif(os.uname().sysname != "Darwin", reason="macOS sandbox-exec")
def test_workspace_terminal_denies_symlink_to_outside_created_after_profile(tmp_path):
    workspace = tmp_path / "work"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    transformed = build_workspace_terminal_args(
        {"command": "cat late-link/secret.txt; touch late-link/escaped.txt"},
        workspace=workspace,
        host_home=tmp_path / "host",
        exact_env={"PATH": os.environ["PATH"]},
    )
    (workspace / "late-link").symlink_to(outside, target_is_directory=True)

    result = subprocess.run(
        ["/bin/bash", "-lc", transformed["command"]],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "secret" not in result.stdout
    assert not (outside / "escaped.txt").exists()


@pytest.mark.skipif(os.uname().sysname != "Darwin", reason="macOS sandbox-exec")
def test_workspace_terminal_can_execute_configured_python_toolchain(tmp_path):
    workspace = Path.cwd().resolve()
    proof = workspace / ".hermes-claude-runtime" / "toolchain-proof"
    proof.unlink(missing_ok=True)
    version_checks = " && ".join(
        f"{shlex.quote(path)} --version >/dev/null"
        for path in (shutil.which("uv"), shutil.which("rg"), shutil.which("git"))
        if path
    )
    transformed = build_workspace_terminal_args(
        {
                "command": (
                f"{version_checks} && {shlex.quote(sys.executable)} -c "
                "\"from pathlib import Path; "
                "Path('.hermes-claude-runtime/toolchain-proof').write_text('ok')\""
            )
        },
        workspace=workspace,
        host_home=tmp_path / "host",
        exact_env={"PATH": os.environ["PATH"]},
    )

    result = subprocess.run(
        ["/bin/bash", "-lc", transformed["command"]],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=10,
    )

    try:
        assert result.returncode == 0, result.stderr
        assert proof.read_text(encoding="utf-8") == "ok"
    finally:
        proof.unlink(missing_ok=True)


@pytest.mark.skipif(os.uname().sysname != "Darwin", reason="macOS sandbox-exec")
def test_workspace_terminal_denies_homebrew_configuration_reads(tmp_path):
    candidates = [
        path
        for root in (Path("/opt/homebrew/etc"), Path("/usr/local/etc"))
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    ]
    if not candidates:
        pytest.skip("no local package-manager configuration file is present")
    workspace = tmp_path / "work"
    workspace.mkdir()
    transformed = build_workspace_terminal_args(
        {"command": f"cat {shlex.quote(str(candidates[0]))}"},
        workspace=workspace,
        host_home=tmp_path / "host",
        exact_env={"PATH": os.environ["PATH"]},
    )

    result = subprocess.run(
        ["/bin/bash", "-lc", transformed["command"]],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "Operation not permitted" in result.stderr

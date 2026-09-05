"""Termux uv / PATH helpers for Android wheel-tag and glibc-first PATH."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_cli import _install_repair as install_repair
from hermes_cli._early_recovery import (
    _is_termux_env,
    fix_termux_node_shebangs,
    prefer_termux_bionic_path,
    termux_uv_python_platform,
    with_uv_termux_python_platform,
)


def test_is_termux_env_true_for_termux_prefix():
    assert _is_termux_env({"PREFIX": "/data/data/com.termux/files/usr"}) is True


def test_is_termux_env_false_for_generic_linux_prefix():
    assert _is_termux_env({"PREFIX": "/usr/local"}) is False
    assert _is_termux_env({"PREFIX": "/opt/com.termux/other"}) is False


def test_fix_termux_node_shebangs_noop_off_termux(tmp_path: Path):
    (tmp_path / "node_modules").mkdir()
    with patch("hermes_cli._early_recovery.subprocess.run") as run:
        fix_termux_node_shebangs(tmp_path)
    run.assert_not_called()


def test_fix_termux_node_shebangs_runs_fixer_for_discovered_bins(
    tmp_path: Path, monkeypatch
):
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    target = bin_dir / "tsc"
    target.write_bytes(b"#!/usr/bin/env node\nconsole.log(1)\n")
    noise = tmp_path / "node_modules" / "pkg" / "index.js"
    noise.parent.mkdir(parents=True)
    noise.write_bytes(b"#!/usr/bin/env node\n")

    monkeypatch.setattr(
        "hermes_cli._early_recovery._is_termux_env", lambda env=None: True
    )
    monkeypatch.setattr(
        "hermes_cli._early_recovery.shutil.which",
        lambda name: (
            "/usr/bin/termux-fix-shebang"
            if name == "termux-fix-shebang"
            else None
        ),
    )

    real_path = Path

    class _FakeEnvPath:
        def exists(self):
            return False

    def _path_factory(arg, *args, **kwargs):
        if str(arg) == "/usr/bin/env":
            return _FakeEnvPath()
        return real_path(arg, *args, **kwargs)

    monkeypatch.setattr("hermes_cli._early_recovery.Path", _path_factory)

    with patch("hermes_cli._early_recovery.subprocess.run") as run:
        fix_termux_node_shebangs(tmp_path)

    run.assert_called_once()
    args = run.call_args[0][0]
    assert args[0] == "/usr/bin/termux-fix-shebang"
    assert str(target) in args
    assert str(noise) not in args


def test_termux_uv_python_platform_aarch64():
    env = {"HOSTTYPE": "aarch64"}
    assert termux_uv_python_platform(env) == "aarch64-unknown-linux-gnu"


def test_termux_uv_python_platform_x86_64():
    env = {"HOSTTYPE": "x86_64"}
    assert termux_uv_python_platform(env) == "x86_64-unknown-linux-gnu"


def test_termux_uv_python_platform_unknown_arch_skips_bare_linux():
    assert termux_uv_python_platform({"HOSTTYPE": "riscv64"}) == ""


def test_termux_uv_python_platform_override():
    env = {"HOSTTYPE": "aarch64", "HERMES_UV_PYTHON_PLATFORM": "aarch64-linux-android"}
    assert termux_uv_python_platform(env) == "aarch64-linux-android"


def test_with_uv_termux_python_platform_injects_arch_flag():
    env = {
        "TERMUX_VERSION": "0.118.3",
        "PREFIX": "/data/data/com.termux/files/usr",
        "HOSTTYPE": "aarch64",
    }
    cmd = ["/usr/bin/uv", "pip", "install", "-e", ".[termux-all]"]
    assert with_uv_termux_python_platform(cmd, env) == [
        "/usr/bin/uv",
        "pip",
        "install",
        "--python-platform",
        "aarch64-unknown-linux-gnu",
        "-e",
        ".[termux-all]",
    ]


def test_with_uv_termux_python_platform_noop_unknown_arch():
    env = {
        "TERMUX_VERSION": "1",
        "HOSTTYPE": "riscv64",
    }
    cmd = ["uv", "pip", "install", "markupsafe"]
    assert with_uv_termux_python_platform(cmd, env) == cmd


def test_with_uv_termux_python_platform_noop_off_termux():
    cmd = ["uv", "pip", "install", "markupsafe"]
    assert with_uv_termux_python_platform(cmd, {}) == cmd


def test_with_uv_termux_python_platform_idempotent():
    env = {"TERMUX_VERSION": "1", "HOSTTYPE": "aarch64"}
    cmd = ["uv", "pip", "install", "--python-platform", "linux", "x"]
    assert with_uv_termux_python_platform(cmd, env) == cmd


def test_prefer_termux_bionic_path_puts_prefix_bin_first():
    env = {
        "TERMUX_VERSION": "0.118.3",
        "PREFIX": "/data/data/com.termux/files/usr",
        "PATH": "/data/data/com.termux/files/usr/glibc/bin:/opt/bin:/data/data/com.termux/files/usr/bin",
    }
    fixed = prefer_termux_bionic_path(env)
    assert fixed["PATH"].startswith("/data/data/com.termux/files/usr/bin:")
    assert "/data/data/com.termux/files/usr/glibc/bin" in fixed["PATH"]
    # Original env untouched.
    assert env["PATH"].startswith("/data/data/com.termux/files/usr/glibc/bin:")


def test_prefer_termux_bionic_path_noop_off_termux():
    env = {"PATH": "/usr/bin:/bin"}
    assert prefer_termux_bionic_path(env) == env


def test_install_repair_run_install_cmd_injects_termux_uv_platform(tmp_path: Path):
    """Delegation: ``_run_install_cmd`` applies PATH + --python-platform on Termux."""
    env = {
        "TERMUX_VERSION": "0.118.3",
        "PREFIX": "/data/data/com.termux/files/usr",
        "HOSTTYPE": "aarch64",
        "PATH": "/data/data/com.termux/files/usr/glibc/bin:/usr/bin",
    }
    cmd = ["uv", "pip", "install", "-e", ".[termux-all]"]
    captured: dict = {}

    def _fake_run(argv, cwd=None, check=None, env=None):
        captured["argv"] = list(argv)
        captured["env"] = dict(env) if env is not None else None
        return MagicMock(returncode=0)

    with (
        patch.object(install_repair, "_is_windows", return_value=False),
        patch.object(install_repair.subprocess, "run", side_effect=_fake_run),
    ):
        install_repair._run_install_cmd(cmd, env=env, root=tmp_path)

    assert captured["argv"] == [
        "uv",
        "pip",
        "install",
        "--python-platform",
        "aarch64-unknown-linux-gnu",
        "-e",
        ".[termux-all]",
    ]
    assert captured["env"]["PATH"].startswith(
        "/data/data/com.termux/files/usr/bin:"
    )

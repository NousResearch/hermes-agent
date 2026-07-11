"""Tests for ``hermes gui`` desktop launcher wiring."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main


def _ns(**kw):
    defaults = dict(
        skip_build=False,
        build_only=False,
        force_build=False,
        source=False,
        fake_boot=False,
        ignore_existing=False,
        hermes_root=None,
        cwd=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _make_desktop_tree(tmp_path: Path) -> Path:
    root = tmp_path / "hermes-agent"
    desktop_dir = root / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}", encoding="utf-8")
    return root


def _native_binary_bytes(platform_name: str, arch: str = "x64") -> bytes:
    if platform_name == "linux":
        data = bytearray(64)
        data[:4] = b"\x7fELF"
        data[4] = 2 if arch in {"x64", "arm64"} else 1
        data[5] = 1
        machine = {"ia32": 3, "x64": 62, "arm": 40, "arm64": 183}[arch]
        data[18:20] = machine.to_bytes(2, "little")
        return bytes(data)
    if platform_name == "win32":
        data = bytearray(128)
        data[:2] = b"MZ"
        data[0x3C:0x40] = (64).to_bytes(4, "little")
        data[64:68] = b"PE\0\0"
        machine = {"ia32": 0x14C, "x64": 0x8664, "arm": 0x1C4, "arm64": 0xAA64}[arch]
        data[68:70] = machine.to_bytes(2, "little")
        return bytes(data)
    if platform_name == "darwin":
        cpu = {"ia32": 7, "x64": 0x01000007, "arm": 12, "arm64": 0x0100000C}[arch]
        return bytes.fromhex("cffaedfe") + cpu.to_bytes(4, "little") + bytes(24)
    raise AssertionError(f"unsupported fixture platform: {platform_name}")


def _packaged_node_pty_payload(executable: Path, platform: str) -> Path:
    if platform == "darwin":
        resources_dir = executable.parent.parent / "Resources"
    else:
        resources_dir = executable.parent / "resources"
    return (
        resources_dir
        / "app.asar.unpacked"
        / "dist"
        / "node_modules"
        / "node-pty"
        / "build"
        / "Release"
        / "pty.node"
    )


def _make_packaged_executable(root: Path, monkeypatch, platform: str = "darwin") -> Path:
    monkeypatch.setattr(cli_main.sys, "platform", platform)
    desktop_dir = root / "apps" / "desktop"
    if platform == "darwin":
        exe = desktop_dir / "release" / "mac-arm64" / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    elif platform == "win32":
        exe = desktop_dir / "release" / "win-unpacked" / "Hermes.exe"
    else:
        exe = desktop_dir / "release" / "linux-unpacked" / "hermes"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    binding = _packaged_node_pty_payload(exe, platform)
    binding.parent.mkdir(parents=True)
    binding.write_bytes(_native_binary_bytes(platform))
    node_pty_root = binding.parents[2]
    (node_pty_root / "package.json").write_text('{"main":"lib/index.js"}', encoding="utf-8")
    (node_pty_root / "lib").mkdir()
    (node_pty_root / "lib" / "index.js").write_text("module.exports = {}", encoding="utf-8")
    if platform == "darwin":
        helper = binding.parent / "spawn-helper"
        helper.write_bytes(b"native helper")
        helper.chmod(0o755)
    return exe


def test_gui_installs_packages_and_launches_desktop_app(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch)

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_ok = subprocess.CompletedProcess(["npm", "run", "pack"], 0)
    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok) as mock_install, \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_ok, launch_ok]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0
    mock_install.assert_called_once_with("/usr/bin/npm", root, capture_output=False, env=None)
    assert mock_run.call_args_list[0].args[0] == ["/usr/bin/npm", "run", "pack"]
    assert mock_run.call_args_list[0].kwargs["cwd"] == desktop_dir
    assert mock_run.call_args_list[1].args[0] == [str(packaged_exe)]
    assert mock_run.call_args_list[1].kwargs["cwd"] == desktop_dir


def test_gui_preserves_nixos_python_for_native_desktop_build(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch)

    ok = subprocess.CompletedProcess([], 0)
    nixos_env = {"PYTHON": "/nix/store/python/bin/python3", "PATH": "/nix/store/bin"}

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._nixos_build_env", return_value=nixos_env), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=ok) as mock_install, \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main.subprocess.run", side_effect=[ok, ok]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0
    mock_install.assert_called_once_with("/usr/bin/npm", root, capture_output=False, env=nixos_env)
    build_call = mock_run.call_args_list[0]
    assert build_call.args[0] == ["/usr/bin/npm", "run", "pack"]
    assert build_call.kwargs["cwd"] == desktop_dir
    assert build_call.kwargs["env"]["PYTHON"] == nixos_env["PYTHON"]
    assert mock_run.call_args_list[1].args[0] == [str(packaged_exe)]


def test_gui_forwards_desktop_environment_overrides(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    hermes_root = tmp_path / "custom-hermes"
    cwd = tmp_path / "project"
    hermes_root.mkdir()
    cwd.mkdir()
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main.subprocess.run", side_effect=[ok, ok]) as mock_run, \
         pytest.raises(SystemExit):
        cli_main.cmd_gui(_ns(
            fake_boot=True,
            ignore_existing=True,
            hermes_root=str(hermes_root),
            cwd=str(cwd),
        ))

    launch_env = mock_run.call_args_list[1].kwargs["env"]
    assert launch_env["HERMES_DESKTOP_BOOT_FAKE"] == "1"
    assert launch_env["HERMES_DESKTOP_IGNORE_EXISTING"] == "1"
    assert launch_env["HERMES_DESKTOP_HERMES_ROOT"] == str(hermes_root)
    assert launch_env["HERMES_DESKTOP_CWD"] == str(cwd)


def test_gui_exits_when_npm_missing(tmp_path, monkeypatch, capsys):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    with patch("hermes_cli.main.shutil.which", return_value=None), \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    assert "npm was not found" in capsys.readouterr().out


def test_gui_skip_build_requires_existing_packaged_app(tmp_path, monkeypatch, capsys):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")

    with pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(skip_build=True))

    assert exc.value.code == 1
    assert "no packaged desktop app" in capsys.readouterr().out


def test_gui_skip_build_launches_existing_packaged_app_without_npm(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch)

    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main.shutil.which", return_value=None), \
         patch("hermes_cli.main._run_npm_install_deterministic") as mock_install, \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(skip_build=True))

    assert exc.value.code == 0
    mock_install.assert_not_called()
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == [str(packaged_exe)]


def test_gui_skip_build_rejects_incomplete_native_payload(tmp_path, monkeypatch, capsys):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")
    _packaged_node_pty_payload(packaged_exe, "linux").unlink()

    with patch("hermes_cli.main.subprocess.run") as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(skip_build=True))

    assert exc.value.code == 1
    mock_run.assert_not_called()
    output = capsys.readouterr().out
    assert "native payload is missing or empty" in output
    assert "hermes desktop --force-build" in output


def test_gui_linux_configures_sandbox_before_launch(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")
    sandbox = packaged_exe.parent / "chrome-sandbox"
    sandbox.write_text("", encoding="utf-8")
    sandbox.chmod(0o755)
    ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/sudo"), \
         patch("hermes_cli.main.subprocess.run", return_value=ok) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(skip_build=True))

    assert exc.value.code == 0
    assert mock_run.call_args_list[0].args[0] == ["/usr/bin/sudo", "chown", "root:root", str(sandbox)]
    assert mock_run.call_args_list[1].args[0] == ["/usr/bin/sudo", "chmod", "4755", str(sandbox)]
    assert mock_run.call_args_list[2].args[0] == [str(packaged_exe)]


def test_gui_linux_rejects_symlink_sandbox(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")
    # Point chrome-sandbox at an unrelated file via symlink
    target = tmp_path / "dangerous"
    target.write_text("pwned", encoding="utf-8")
    sandbox = packaged_exe.parent / "chrome-sandbox"
    sandbox.symlink_to(target)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/sudo"), \
         patch("hermes_cli.main.subprocess.run") as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(skip_build=True))

    assert exc.value.code == 1
    # Must NOT have called sudo chown/chmod on the symlink target
    for call in mock_run.call_args_list:
        assert "chown" not in call.args[0]
        assert "chmod" not in call.args[0]


def test_gui_linux_skips_fixup_when_already_configured(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")
    sandbox = packaged_exe.parent / "chrome-sandbox"
    sandbox.write_text("", encoding="utf-8")
    # Simulate root-owned 4755 — lstat().st_uid==0 and mode==0o4755
    # We can't actually chown to root in tests, so mock lstat to return
    # the expected values directly.
    import stat as stat_mod
    fake_stat = type("s", (), {"st_uid": 0, "st_mode": 0o4755 | stat_mod.S_IFREG})()
    sandbox_lstat_orig = type(sandbox).lstat
    monkeypatch.setattr(type(sandbox), "lstat", lambda self: fake_stat)

    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/sudo"), \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(skip_build=True))

    assert exc.value.code == 0
    # Only the launch call — no sudo chown/chmod
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == [str(packaged_exe)]


def test_gui_linux_falls_back_to_no_sandbox_when_userns_is_restricted(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")
    sandbox = packaged_exe.parent / "chrome-sandbox"
    sandbox.write_text("", encoding="utf-8")

    launch_ok = subprocess.CompletedProcess([str(packaged_exe), "--no-sandbox"], 0)

    with patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=False), \
         patch("hermes_cli.main._desktop_linux_needs_no_sandbox", return_value=True), \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(skip_build=True))

    assert exc.value.code == 0
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0] == [str(packaged_exe), "--no-sandbox"]


def test_gui_linux_exits_when_sandbox_fixup_fails_without_safe_fallback(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch, platform="linux")

    with patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=False), \
         patch("hermes_cli.main._desktop_linux_needs_no_sandbox", return_value=False), \
         patch("hermes_cli.main.subprocess.run") as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(skip_build=True))

    assert exc.value.code == 1
    mock_run.assert_not_called()


def test_gui_source_mode_uses_renderer_build_and_electron(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    build_ok = subprocess.CompletedProcess(["npm", "run", "build"], 0)
    launch_ok = subprocess.CompletedProcess(["npm", "exec", "--", "electron", "."], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main.subprocess.run", side_effect=[build_ok, launch_ok]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(source=True))

    assert exc.value.code == 0
    assert mock_run.call_args_list[0].args[0] == ["/usr/bin/npm", "run", "build"]
    assert mock_run.call_args_list[0].kwargs["cwd"] == desktop_dir
    assert mock_run.call_args_list[1].args[0] == ["/usr/bin/npm", "exec", "--", "electron", "."]
    assert mock_run.call_args_list[1].kwargs["cwd"] == desktop_dir


@pytest.mark.parametrize(
    "argv",
    [
        ["hermes", "gui"],
        ["hermes", "-m", "gpt5", "gui"],
    ],
)
def test_gui_is_known_builtin_for_plugin_gating(argv):
    with patch.object(sys, "argv", argv):
        assert cli_main._plugin_cli_discovery_needed() is False


# ── Content-hash stamp tests ──────────────────────────────────────────


def test_desktop_build_stamp_skips_build_when_up_to_date(tmp_path, monkeypatch):
    """When the stamp matches and the artifact exists, build is skipped entirely."""
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    launch_ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main._desktop_build_needed", return_value=False), \
         patch("hermes_cli.main._run_npm_install_deterministic") as mock_install, \
         patch("hermes_cli.main.subprocess.run", return_value=launch_ok) as mock_run, \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0
    mock_install.assert_not_called()
    mock_run.assert_called_once()  # only the launch call, no build


def test_desktop_force_build_overrides_stamp(tmp_path, monkeypatch):
    """--force-build forces a rebuild even when the stamp says up-to-date."""
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch)

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_ok = subprocess.CompletedProcess(["npm", "run", "pack"], 0)
    launch_ok = subprocess.CompletedProcess([], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok) as mock_install, \
         patch("hermes_cli.main._desktop_build_needed", return_value=False), \
         patch("hermes_cli.main._write_desktop_build_stamp") as mock_stamp, \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_ok, launch_ok]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(force_build=True))

    assert exc.value.code == 0
    mock_install.assert_called_once()
    mock_stamp.assert_called_once()
    # pack + launch = 2 calls
    assert mock_run.call_count == 2


def test_gui_rejects_successful_pack_without_native_payload(tmp_path, monkeypatch, capsys):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    packaged_exe = desktop_dir / "release" / "linux-unpacked" / "hermes"

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_ok = subprocess.CompletedProcess(["npm", "run", "pack"], 0)

    def build_without_native_payload(*args, **kwargs):
        packaged_exe.parent.mkdir(parents=True, exist_ok=True)
        packaged_exe.write_text("", encoding="utf-8")
        return pack_ok

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_build_needed", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp") as mock_stamp, \
         patch("hermes_cli.main.subprocess.run", side_effect=build_without_native_payload), \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns(build_only=True))

    assert exc.value.code == 1
    mock_stamp.assert_not_called()
    assert "without a usable node-pty native payload" in capsys.readouterr().out


def test_compute_desktop_content_hash_stable(tmp_path, monkeypatch):
    """_compute_desktop_content_hash returns the same digest for identical trees."""
    root = _make_desktop_tree(tmp_path)
    (root / "apps" / "desktop" / "main.js").write_text("console.log('hi')", encoding="utf-8")
    (root / "package.json").write_text('{"name":"hermes"}', encoding="utf-8")
    (root / "package-lock.json").write_text('{}', encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    h1 = cli_main._compute_desktop_content_hash(root)
    h2 = cli_main._compute_desktop_content_hash(root)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_compute_desktop_content_hash_changes_on_edit(tmp_path, monkeypatch):
    """Editing a file under apps/desktop/ changes the hash."""
    root = _make_desktop_tree(tmp_path)
    (root / "apps" / "desktop" / "main.js").write_text("v1", encoding="utf-8")
    (root / "package.json").write_text("{}", encoding="utf-8")
    (root / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    h1 = cli_main._compute_desktop_content_hash(root)
    (root / "apps" / "desktop" / "main.js").write_text("v2", encoding="utf-8")
    h2 = cli_main._compute_desktop_content_hash(root)
    assert h1 != h2


def test_desktop_build_needed_detects_missing_artifact(tmp_path, monkeypatch):
    """Even with a valid stamp, missing artifact means build is needed."""
    root = _make_desktop_tree(tmp_path)
    (root / "package.json").write_text("{}", encoding="utf-8")
    (root / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    # Write a stamp that matches current content
    cli_main._write_desktop_build_stamp(root, source_mode=False)
    # No packaged executable exists → build needed
    assert cli_main._desktop_build_needed(
        root / "apps" / "desktop", root, source_mode=False
    ) is True


@pytest.mark.parametrize("payload_bytes", [None, b"", b"not a native binary"])
def test_desktop_build_needed_detects_incomplete_native_payload(
    tmp_path, monkeypatch, payload_bytes
):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")
    binding = _packaged_node_pty_payload(packaged_exe, "linux")
    if payload_bytes is None:
        binding.unlink()
    else:
        binding.write_bytes(payload_bytes)
    cli_main._write_desktop_build_stamp(root, source_mode=False)

    assert cli_main._desktop_build_needed(
        desktop_dir, root, source_mode=False
    ) is True


def test_desktop_build_needed_rejects_wrong_architecture_payload(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    desktop_dir = root / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")
    _packaged_node_pty_payload(packaged_exe, "linux").write_bytes(
        _native_binary_bytes("linux", "arm64")
    )
    cli_main._write_desktop_build_stamp(root, source_mode=False)

    assert cli_main._desktop_build_needed(desktop_dir, root, source_mode=False) is True


def test_desktop_packaged_runtime_accepts_debug_fallback(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")
    release_binding = _packaged_node_pty_payload(packaged_exe, "linux")
    debug_binding = release_binding.parents[1] / "Debug" / "pty.node"
    debug_binding.parent.mkdir()
    release_binding.rename(debug_binding)

    assert cli_main._desktop_packaged_runtime_complete(packaged_exe) is True


def test_desktop_packaged_runtime_requires_package_and_js_files(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")
    node_pty_root = _packaged_node_pty_payload(packaged_exe, "linux").parents[2]
    (node_pty_root / "lib" / "index.js").unlink()

    assert cli_main._desktop_packaged_runtime_complete(packaged_exe) is False


def test_desktop_packaged_runtime_requires_executable_darwin_helper(tmp_path, monkeypatch):
    root = _make_desktop_tree(tmp_path)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="darwin")
    helper = _packaged_node_pty_payload(packaged_exe, "darwin").parent / "spawn-helper"
    helper.chmod(0o644)

    assert cli_main._desktop_packaged_runtime_complete(packaged_exe) is False


def test_desktop_build_stamp_round_trip(tmp_path, monkeypatch):
    """Write stamp, then _desktop_build_needed returns False when artifact exists."""
    root = _make_desktop_tree(tmp_path)
    (root / "package.json").write_text("{}", encoding="utf-8")
    (root / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    # Create the artifact so the "artifact exists" check passes
    _make_packaged_executable(root, monkeypatch)
    # Write stamp
    cli_main._write_desktop_build_stamp(root, source_mode=False)
    # Build should NOT be needed
    assert cli_main._desktop_build_needed(
        root / "apps" / "desktop", root, source_mode=False
    ) is False


def test_compute_desktop_content_hash_works_without_gitignore(tmp_path, monkeypatch):
    """When no .gitignore exists, _compute_desktop_content_hash still works (matches everything)."""
    root = _make_desktop_tree(tmp_path)
    (root / "apps" / "desktop" / "main.js").write_text("v1", encoding="utf-8")
    (root / "package.json").write_text("{}", encoding="utf-8")
    (root / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    # No .gitignore → pathspec matches nothing → all files hashed
    h = cli_main._compute_desktop_content_hash(root)
    assert len(h) == 64  # valid sha256 hex

    # Edit a file → hash changes
    (root / "apps" / "desktop" / "main.js").write_text("v2", encoding="utf-8")
    h2 = cli_main._compute_desktop_content_hash(root)
    assert h != h2


def test_compute_desktop_content_hash_respects_gitignore(tmp_path, monkeypatch):
    """Files matched by .gitignore are excluded from the hash."""
    root = _make_desktop_tree(tmp_path)
    (root / "apps" / "desktop" / "main.js").write_text("hello", encoding="utf-8")
    (root / "apps" / "desktop" / "secrets.env").write_text("API_KEY=xxx", encoding="utf-8")
    (root / "package.json").write_text("{}", encoding="utf-8")
    (root / "package-lock.json").write_text("{}", encoding="utf-8")
    (root / ".gitignore").write_text("*.env\n", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)

    # Reset cached spec
    cli_main._DESKTOP_STAMP_SPEC = None

    h1 = cli_main._compute_desktop_content_hash(root)

    # Change the .env file (ignored) — hash should NOT change
    (root / "apps" / "desktop" / "secrets.env").write_text("API_KEY=yyy", encoding="utf-8")
    cli_main._DESKTOP_STAMP_SPEC = None  # reset since gitignore hasn't changed
    h2 = cli_main._compute_desktop_content_hash(root)
    assert h1 == h2, "changing an ignored file should not change the hash"

    # Change the .js file (not ignored) — hash SHOULD change
    (root / "apps" / "desktop" / "main.js").write_text("world", encoding="utf-8")
    cli_main._DESKTOP_STAMP_SPEC = None
    h3 = cli_main._compute_desktop_content_hash(root)
    assert h1 != h3, "changing a tracked file should change the hash"


# ── Electron build-cache recovery tests ───────────────────────────────


def _write_zip(path: Path) -> None:
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("electron", "fake binary payload")


def test_purge_electron_build_cache_clears_all_zips_and_unpacked_dir(tmp_path, monkeypatch):
    """Purge is unconditional: it removes every electron-*.zip (regardless of
    whether stdlib zipfile thinks it's corrupt) plus the half-written unpacked
    dir, because @electron/get's own SHASUM check on re-download is the real
    validator — not a self-rolled one."""
    cache = tmp_path / "electron-cache"
    # A "clean" zip and a prepended-junk zip — the latter is the real-world
    # corruption that zipfile.testzip() silently passes (it reads from the
    # end-of-central-directory backward), which is why we don't gate on it.
    clean = cache / "electron-v40.9.3-linux-x64.zip"
    prepended = cache / "hashdir" / "electron-v40.9.3-linux-x64.zip"
    _write_zip(clean)
    _write_zip(prepended)
    prepended.write_bytes(b"\x00" * 4096 + prepended.read_bytes())

    desktop_dir = tmp_path / "apps" / "desktop"
    unpacked = desktop_dir / "release" / "linux-unpacked"
    unpacked.mkdir(parents=True)
    (unpacked / "LICENSE.electron.txt").write_text("x", encoding="utf-8")
    (unpacked / "resources.pak").write_text("x", encoding="utf-8")

    monkeypatch.setattr(cli_main, "_electron_download_cache_dirs", lambda: [cache])

    removed = cli_main._purge_electron_build_cache(desktop_dir)

    assert clean in removed
    assert prepended in removed
    assert unpacked in removed
    assert not clean.exists()
    assert not prepended.exists()
    assert not unpacked.exists()


def test_purge_electron_build_cache_empty_when_nothing_present(tmp_path, monkeypatch):
    """No cached zips and no unpacked dir → nothing removed, so the caller
    knows a retry is pointless."""
    cache = tmp_path / "electron-cache"
    cache.mkdir()
    desktop_dir = tmp_path / "apps" / "desktop"
    monkeypatch.setattr(cli_main, "_electron_download_cache_dirs", lambda: [cache])

    assert cli_main._purge_electron_build_cache(desktop_dir) == []


def test_gui_retries_pack_once_after_purging_build_cache(tmp_path, monkeypatch):
    """First pack fails with NO packaged executable (corrupt-download shape),
    purge clears the cache, second pack succeeds and produces the exe, launch."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    # Executable is ABSENT at build-failure time — that is the corrupt-download
    # signature the cache purge + retry exist for (#40187). Only the successful
    # retry produces it (via the side_effect below).
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    packaged_exe = root / "apps" / "desktop" / "release" / "linux-unpacked" / "hermes"

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_fail = subprocess.CompletedProcess(["npm", "run", "pack"], 1)
    pack_ok = subprocess.CompletedProcess(["npm", "run", "pack"], 0)
    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    _calls = {"n": 0}

    def run_side_effect(*args, **kwargs):
        _calls["n"] += 1
        if _calls["n"] == 1:
            return pack_fail
        if _calls["n"] == 2:
            # Successful retry materializes the packaged executable.
            _make_packaged_executable(root, monkeypatch, platform="linux")
            return pack_ok
        return launch_ok

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._purge_electron_build_cache", return_value=[Path("/c/electron.zip")]) as mock_purge, \
         patch("hermes_cli.main._electron_dist_ok", return_value=False), \
         patch("hermes_cli.main._redownload_electron_dist", return_value=True), \
         patch("hermes_cli.main.subprocess.run", side_effect=run_side_effect) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0
    mock_purge.assert_called_once()
    # pack(fail) → repair succeeds → pack(ok) → launch = 3 subprocess.run calls
    assert mock_run.call_count == 3
    assert mock_run.call_args_list[0].args[0] == ["/usr/bin/npm", "run", "pack"]
    assert mock_run.call_args_list[1].args[0] == ["/usr/bin/npm", "run", "pack"]
    assert mock_run.call_args_list[2].args[0] == [str(packaged_exe)]


def test_gui_redownloads_electron_via_mirror_then_repacks(tmp_path, monkeypatch, capsys):
    """Purge clears nothing and the pinned electronDist (#38673) is missing →
    the mirror fallback must drive electron's own downloader (NOT another pack,
    which never downloads Electron) and only then retry pack (#47266)."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    # No packaged executable: the corrupt-download recovery must run.
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    monkeypatch.delenv("ELECTRON_MIRROR", raising=False)

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_fail = subprocess.CompletedProcess(["npm", "run", "pack"], 1)
    nixos_env = {"PYTHON": "/nix/store/python/bin/python3", "PATH": "/nix/store/bin"}

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._nixos_build_env", return_value=nixos_env), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._purge_electron_build_cache", return_value=[]), \
         patch("hermes_cli.main._electron_dist_ok", return_value=False), \
         patch("hermes_cli.main._redownload_electron_dist", side_effect=[False, True]) as mock_dl, \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_fail, pack_fail]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    # initial pack + mirror pack = 2 npm calls. The first-retry pack is skipped
    # because the canonical-source re-download (no mirror) failed, so there was
    # never a binary to build against.
    assert mock_run.call_count == 2
    # First re-download attempt is canonical (no mirror); the second drives the
    # public mirror.
    assert mock_dl.call_args_list[0].kwargs.get("mirror") is None
    assert mock_dl.call_args_list[1].kwargs["mirror"]
    # Only the mirror-driven pack carries ELECTRON_MIRROR, and it must retain
    # the NixOS Python selected for node-gyp.
    assert "ELECTRON_MIRROR" not in (mock_run.call_args_list[0].kwargs.get("env") or {})
    assert mock_run.call_args_list[1].kwargs["env"]["ELECTRON_MIRROR"]
    assert mock_run.call_args_list[1].kwargs["env"]["PYTHON"] == nixos_env["PYTHON"]
    assert "Desktop GUI build failed" in capsys.readouterr().out


def test_gui_retries_pack_under_mirror_even_when_prefetch_blocked(tmp_path, monkeypatch, capsys):
    """When electron's own downloader can't fetch the binary (even via the
    mirror), still retry pack under ELECTRON_MIRROR: the build resolves
    electronDist dynamically and lets electron-builder fetch Electron itself
    via @electron/get, which honors the mirror. That retry is no longer
    pointless (it was, back when electronDist was a static path)."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    # No packaged executable: the corrupt-download recovery must run.
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    monkeypatch.delenv("ELECTRON_MIRROR", raising=False)

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_fail = subprocess.CompletedProcess(["npm", "run", "pack"], 1)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._purge_electron_build_cache", return_value=[]), \
         patch("hermes_cli.main._electron_dist_ok", return_value=False), \
         patch("hermes_cli.main._redownload_electron_dist", return_value=False), \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_fail, pack_fail]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    # Initial pack + mirror-driven pack = 2; the mirror retry runs even though
    # the pre-fetch failed, so electron-builder gets a shot at downloading.
    assert mock_run.call_count == 2
    assert "ELECTRON_MIRROR" not in (mock_run.call_args_list[0].kwargs.get("env") or {})
    assert mock_run.call_args_list[1].kwargs["env"]["ELECTRON_MIRROR"]
    assert "Desktop GUI build failed" in capsys.readouterr().out


def test_gui_install_failure_self_heals_electron_and_continues(tmp_path, monkeypatch, capsys):
    """npm ci failing on electron's blocked binary download must NOT abort the
    install: with the electron package staged, repopulate its dist and continue
    to the build instead of sys.exit-ing before pack ever runs (#47266/#48021)."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    packaged_exe = _make_packaged_executable(root, monkeypatch, platform="linux")
    # electron package staged on disk (postinstall download was the casualty).
    (root / "apps" / "desktop" / "node_modules" / "electron").mkdir(parents=True)
    (root / "apps" / "desktop" / "node_modules" / "electron" / "package.json").write_text("{}", encoding="utf-8")
    (root / "apps" / "desktop" / "node_modules" / "electron" / "install.js").write_text("", encoding="utf-8")

    install_fail = subprocess.CompletedProcess(["npm", "ci"], 1)
    pack_ok = subprocess.CompletedProcess(["npm", "run", "pack"], 0)
    launch_ok = subprocess.CompletedProcess([str(packaged_exe)], 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_fail), \
         patch("hermes_cli.main._desktop_linux_sandbox_fixup", return_value=True), \
         patch("hermes_cli.main._write_desktop_build_stamp"), \
         patch("hermes_cli.main._electron_dist_ok", return_value=False), \
         patch("hermes_cli.main._try_redownload_electron_dist", return_value=True) as mock_dl, \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_ok, launch_ok]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 0
    mock_dl.assert_called()  # tried to repopulate the dist
    # pack + launch ran — the install failure did NOT abort the build.
    assert mock_run.call_count == 2
    assert "repopulated" in capsys.readouterr().out.lower()


def test_gui_install_failure_hard_fails_when_electron_not_staged(tmp_path, monkeypatch, capsys):
    """A dependency-install failure where electron never even staged is a genuine
    error (not a blocked binary download) — hard-fail with guidance, don't try to
    self-heal a tree that isn't there."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch, platform="linux")

    install_fail = subprocess.CompletedProcess(["npm", "ci"], 1)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_fail), \
         patch("hermes_cli.main.subprocess.run") as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    mock_run.assert_not_called()  # build never started
    assert "Desktop dependency install failed" in capsys.readouterr().out


def test_gui_install_failure_hard_fails_when_electron_dist_exists(tmp_path, monkeypatch, capsys):
    """If npm install fails but Electron dist is already present, don't classify
    it as the blocked-download shape; fail fast as a generic install error."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    _make_packaged_executable(root, monkeypatch, platform="linux")
    electron_dir = root / "apps" / "desktop" / "node_modules" / "electron"
    electron_dir.mkdir(parents=True)
    (electron_dir / "package.json").write_text("{}", encoding="utf-8")
    (electron_dir / "install.js").write_text("", encoding="utf-8")

    install_fail = subprocess.CompletedProcess(["npm", "ci"], 1)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_fail), \
         patch("hermes_cli.main._electron_dist_ok", return_value=True), \
         patch("hermes_cli.main.subprocess.run") as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    mock_run.assert_not_called()
    assert "Desktop dependency install failed" in capsys.readouterr().out


def test_gui_does_not_retry_after_packaged_executable_exists(tmp_path, monkeypatch, capsys):
    """A build that already produced a packaged executable did NOT fail from the
    Electron-download problem the cache purge + mirror retries exist to repair.

    Regression for #40187: a late failure such as macOS code signing leaves
    Hermes.app/Contents/MacOS/Hermes in place. Re-downloading Electron can't
    repair a signing failure, so the destructive purge + slow mirror retry must
    be skipped — we fail directly instead of grinding through an identical retry.
    """
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    # Executable EXISTS at failure time → late failure, not a corrupt download.
    _make_packaged_executable(root, monkeypatch, platform="darwin")
    monkeypatch.delenv("ELECTRON_MIRROR", raising=False)

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_fail = subprocess.CompletedProcess(["npm", "run", "pack"], 1)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._purge_electron_build_cache", return_value=[Path("/c/electron.zip")]) as mock_purge, \
         patch("hermes_cli.main._redownload_electron_dist", return_value=True) as mock_dl, \
         patch("hermes_cli.main.subprocess.run", return_value=pack_fail) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    # Neither destructive recovery runs, and there is exactly ONE pack attempt.
    mock_purge.assert_not_called()
    mock_dl.assert_not_called()
    assert mock_run.call_count == 1
    assert "Desktop GUI build failed" in capsys.readouterr().out


def test_gui_does_not_override_user_electron_mirror(tmp_path, monkeypatch, capsys):
    """A user-pinned ELECTRON_MIRROR is respected: no extra mirror fallback
    attempt (and we never swap in our default mirror)."""
    root = _make_desktop_tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    # No packaged executable: the build failure is the download-class the
    # mirror fallback handles (and we assert the user's pin is respected).
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    monkeypatch.setenv("ELECTRON_MIRROR", "https://mirror.example/electron/")

    install_ok = subprocess.CompletedProcess(["npm", "ci"], 0)
    pack_fail = subprocess.CompletedProcess(["npm", "run", "pack"], 1)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/npm"), \
         patch("hermes_cli.main._run_npm_install_deterministic", return_value=install_ok), \
         patch("hermes_cli.main._desktop_macos_relaunchable_fixup"), \
         patch("hermes_cli.main._purge_electron_build_cache", return_value=[]) as mock_purge, \
         patch("hermes_cli.main.subprocess.run", side_effect=[pack_fail]) as mock_run, \
         pytest.raises(SystemExit) as exc:
        cli_main.cmd_gui(_ns())

    assert exc.value.code == 1
    mock_purge.assert_called_once()
    assert mock_run.call_count == 1
    assert mock_run.call_args_list[0].kwargs["env"]["ELECTRON_MIRROR"] == "https://mirror.example/electron/"
    assert "Desktop GUI build failed" in capsys.readouterr().out


# ── electronDist (re)download helper tests (#47266) ───────────────────


@pytest.mark.parametrize(
    "platform,rel",
    [
        ("linux", "dist/electron"),
        ("win32", "dist/electron.exe"),
        ("darwin", "dist/Electron.app/Contents/MacOS/Electron"),
    ],
)
def test_electron_dist_ok_per_platform(tmp_path, monkeypatch, platform, rel):
    monkeypatch.setattr(cli_main.sys, "platform", platform)
    electron = tmp_path / "node_modules" / "electron"
    # A dist dir that exists but lacks the binary is NOT ok (partial extraction).
    (electron / "dist").mkdir(parents=True)
    assert cli_main._electron_dist_ok(tmp_path) is False

    binp = electron / rel
    binp.parent.mkdir(parents=True, exist_ok=True)
    binp.write_text("", encoding="utf-8")
    assert cli_main._electron_dist_ok(tmp_path) is True


def test_electron_dir_prefers_workspace_local_package(tmp_path):
    """npm may nest electron under apps/desktop; resolve there over the root hoist."""
    root_electron = tmp_path / "node_modules" / "electron"
    local_electron = tmp_path / "apps" / "desktop" / "node_modules" / "electron"
    root_electron.mkdir(parents=True)
    local_electron.mkdir(parents=True)

    assert cli_main._electron_dir(tmp_path) == local_electron


def test_electron_dir_falls_back_to_root_hoist(tmp_path):
    """When npm hoists electron to the repo root, resolve there."""
    root_electron = tmp_path / "node_modules" / "electron"
    root_electron.mkdir(parents=True)

    assert cli_main._electron_dir(tmp_path) == root_electron


def test_electron_dist_ok_finds_workspace_local_binary(tmp_path, monkeypatch):
    """A nested apps/desktop electron with a valid binary counts as ok."""
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    binp = tmp_path / "apps" / "desktop" / "node_modules" / "electron" / "dist" / "electron"
    binp.parent.mkdir(parents=True)
    binp.write_text("", encoding="utf-8")
    assert cli_main._electron_dist_ok(tmp_path) is True


def test_redownload_electron_dist_noop_when_present(tmp_path, monkeypatch):
    """Already-healthy dist → no download, so an unrelated build failure can't
    trigger a needless ~200 MB refetch."""
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    binp = tmp_path / "node_modules" / "electron" / "dist" / "electron"
    binp.parent.mkdir(parents=True)
    binp.write_text("", encoding="utf-8")

    with patch("hermes_cli.main.subprocess.run") as mock_run:
        assert cli_main._redownload_electron_dist(tmp_path, {}) is True
    mock_run.assert_not_called()


def test_redownload_electron_dist_missing_installer(tmp_path, monkeypatch):
    """No electron/install.js (deps never installed) → nothing to run."""
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    (tmp_path / "node_modules" / "electron").mkdir(parents=True)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/node"), \
         patch("hermes_cli.main.subprocess.run") as mock_run:
        assert cli_main._redownload_electron_dist(tmp_path, {}) is False
    mock_run.assert_not_called()


def test_redownload_electron_dist_runs_installer_with_mirror(tmp_path, monkeypatch):
    """Missing dist → wipe any partial dist + version marker, run electron's own
    install.js with ELECTRON_MIRROR injected, and report success on the binary."""
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    electron = tmp_path / "node_modules" / "electron"
    electron.mkdir(parents=True)
    (electron / "install.js").write_text("// stub", encoding="utf-8")
    # A stale partial dist + version marker that MUST be cleared first, otherwise
    # electron's install.js short-circuits on path.txt and never re-downloads.
    (electron / "dist").mkdir()
    (electron / "dist" / "leftover").write_text("junk", encoding="utf-8")
    (electron / "path.txt").write_text("electron", encoding="utf-8")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        captured["cwd"] = kwargs.get("cwd")
        # simulate electron's install.js producing the dist binary
        binp = electron / "dist" / "electron"
        binp.parent.mkdir(parents=True, exist_ok=True)
        binp.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/node"), \
         patch("hermes_cli.main.subprocess.run", side_effect=fake_run):
        ok = cli_main._redownload_electron_dist(
            tmp_path, {"PATH": "/x"}, mirror="https://mirror.example/electron/"
        )

    assert ok is True
    assert captured["cmd"] == ["/usr/bin/node", str(electron / "install.js")]
    assert captured["cwd"] == str(electron)
    assert captured["env"]["ELECTRON_MIRROR"] == "https://mirror.example/electron/"
    # The partial dir + marker were dropped before the re-download.
    assert not (electron / "dist" / "leftover").exists()
    assert not (electron / "path.txt").exists()


def test_redownload_electron_dist_returns_false_when_download_fails(tmp_path, monkeypatch):
    """install.js ran but produced no binary (still blocked) → False, so the
    caller skips a doomed pack."""
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    electron = tmp_path / "node_modules" / "electron"
    electron.mkdir(parents=True)
    (electron / "install.js").write_text("// stub", encoding="utf-8")

    with patch("hermes_cli.main.shutil.which", return_value="/usr/bin/node"), \
         patch("hermes_cli.main.subprocess.run",
               return_value=subprocess.CompletedProcess(["node"], 1)):
        assert cli_main._redownload_electron_dist(tmp_path, {}) is False


class _FakeProc:
    """Minimal psutil.Process stand-in for the lock-breaker tests."""

    def __init__(self, pid: int, exe: str | None):
        self.pid = pid
        self.info = {"pid": pid, "exe": exe}
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_stop_desktop_build_lock_noop_off_windows(tmp_path, monkeypatch):
    """POSIX can unlink a running binary, so the helper is a no-op there."""
    desktop_dir = tmp_path / "apps" / "desktop"
    exe = desktop_dir / "release" / "linux-unpacked" / "hermes"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli_main.sys, "platform", "linux")

    proc = _FakeProc(4321, str(exe))
    with patch("psutil.process_iter", return_value=[proc]) as it:
        assert cli_main._stop_desktop_processes_locking_build(desktop_dir) == []
    it.assert_not_called()
    assert proc.terminated is False


def test_stop_desktop_build_lock_terminates_only_release_procs(tmp_path, monkeypatch):
    desktop_dir = tmp_path / "apps" / "desktop"
    release = desktop_dir / "release" / "win-unpacked"
    release.mkdir(parents=True)
    locker_exe = release / "Hermes.exe"
    locker_exe.write_text("", encoding="utf-8")
    other_exe = tmp_path / "elsewhere" / "Hermes.exe"
    other_exe.parent.mkdir(parents=True)
    other_exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    monkeypatch.setattr(cli_main.os, "getpid", lambda: 999)

    locker = _FakeProc(101, str(locker_exe))
    unrelated = _FakeProc(102, str(other_exe))
    selfish = _FakeProc(999, str(locker_exe))  # our own PID — never killed
    no_exe = _FakeProc(103, None)

    captured = {}

    def _wait(procs, timeout=None):
        captured["waited"] = list(procs)
        return procs, []

    with patch("psutil.process_iter", return_value=[locker, unrelated, selfish, no_exe]), \
         patch("psutil.wait_procs", side_effect=_wait):
        stopped = cli_main._stop_desktop_processes_locking_build(desktop_dir)

    assert stopped == [101]
    assert locker.terminated is True
    assert unrelated.terminated is False
    assert selfish.terminated is False
    assert captured["waited"] == [locker]


def test_stop_desktop_build_lock_no_release_dir(tmp_path, monkeypatch):
    desktop_dir = tmp_path / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    monkeypatch.setattr(cli_main.sys, "platform", "win32")
    with patch("psutil.process_iter") as it:
        assert cli_main._stop_desktop_processes_locking_build(desktop_dir) == []
    it.assert_not_called()


def test_force_adhoc_signing_disables_discovery_on_local_packaged_rebuild(monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    env = {}
    assert cli_main._force_adhoc_macos_signing(env, source_mode=False) is True
    assert env["CSC_IDENTITY_AUTO_DISCOVERY"] == "false"


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_force_adhoc_signing_noop_off_macos(monkeypatch, platform):
    monkeypatch.setattr(cli_main.sys, "platform", platform)
    env = {}
    assert cli_main._force_adhoc_macos_signing(env, source_mode=False) is False
    assert "CSC_IDENTITY_AUTO_DISCOVERY" not in env


def test_force_adhoc_signing_noop_for_source_mode(monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    env = {}
    assert cli_main._force_adhoc_macos_signing(env, source_mode=True) is False
    assert "CSC_IDENTITY_AUTO_DISCOVERY" not in env


@pytest.mark.parametrize("key", ["CSC_LINK", "APPLE_SIGNING_IDENTITY"])
def test_force_adhoc_signing_preserves_real_identity(monkeypatch, key):
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    env = {key: "secret"}
    assert cli_main._force_adhoc_macos_signing(env, source_mode=False) is False
    assert "CSC_IDENTITY_AUTO_DISCOVERY" not in env


def test_force_adhoc_signing_respects_explicit_caller_flag(monkeypatch):
    monkeypatch.setattr(cli_main.sys, "platform", "darwin")
    env = {"CSC_IDENTITY_AUTO_DISCOVERY": "true"}
    assert cli_main._force_adhoc_macos_signing(env, source_mode=False) is False
    assert env["CSC_IDENTITY_AUTO_DISCOVERY"] == "true"


# --- desktop.* launch options (config.yaml) -------------------------------


def test_desktop_launch_options_defaults_when_no_config():
    with patch("hermes_cli.config.load_config", return_value={}):
        flags, gpu = cli_main._desktop_launch_options()
    assert flags == []
    assert gpu == "auto"


def test_desktop_launch_options_reads_flags_list():
    cfg = {"desktop": {"electron_flags": ["--ozone-platform=x11", "--disable-gpu"]}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        flags, gpu = cli_main._desktop_launch_options()
    assert flags == ["--ozone-platform=x11", "--disable-gpu"]
    assert gpu == "auto"


def test_desktop_launch_options_splits_flag_string():
    cfg = {"desktop": {"electron_flags": "--ozone-platform=x11 --disable-gpu"}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        flags, _ = cli_main._desktop_launch_options()
    assert flags == ["--ozone-platform=x11", "--disable-gpu"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        (True, "1"),
        (False, "0"),
        ("true", "1"),
        ("off", "0"),
        ("auto", "auto"),
        ("garbage", "auto"),
    ],
)
def test_desktop_launch_options_normalizes_disable_gpu(raw, expected):
    cfg = {"desktop": {"disable_gpu": raw}}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        _, gpu = cli_main._desktop_launch_options()
    assert gpu == expected


def test_desktop_launch_options_survives_config_error():
    with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
        flags, gpu = cli_main._desktop_launch_options()
    assert flags == []
    assert gpu == "auto"

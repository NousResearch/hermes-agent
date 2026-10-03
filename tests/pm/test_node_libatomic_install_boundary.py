"""Regression tests for #124926: libatomic repair belongs to install, not verify."""

from pathlib import Path

import pytest

from pm.packages import Nodejs


_LOADER = (
    "node: error while loading shared libraries: libatomic.so.1: "
    "cannot open shared object file: No such file or directory"
)


def _node_script(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.platforms("posix")
def test_node_verify_remains_read_only_when_libatomic_is_missing(tmp_path, monkeypatch):
    """A diagnostic verify, including pm doctor, must never install host packages."""
    node = tmp_path / "bin" / "node"
    _node_script(node, f'#!/bin/sh\necho "{_LOADER}" >&2\nexit 127\n')
    monkeypatch.setattr("pm.packages.current_target", lambda: "linux-x64")

    def forbidden():
        raise AssertionError("verify() attempted host mutation")

    monkeypatch.setattr("pm.libatomic.try_install_libatomic", forbidden)
    reason = Nodejs().verify(tmp_path, "linux-x64")

    assert "libatomic.so.1" in reason
    assert "exited 127" in reason


@pytest.mark.platforms("posix")
def test_staged_repair_installs_libatomic_then_reprobes(tmp_path, monkeypatch):
    node = tmp_path / "bin" / "node"
    _node_script(node, f'#!/bin/sh\necho "{_LOADER}" >&2\nexit 127\n')
    monkeypatch.setattr("pm.packages.current_target", lambda: "linux-x64")

    calls = {"n": 0}

    def install():
        calls["n"] += 1
        _node_script(node, "#!/bin/sh\necho v26.7.0\n")
        return True, "install with sudo dnf install -y libatomic"

    monkeypatch.setattr("pm.libatomic.try_install_libatomic", install)
    package = Nodejs()
    first = package.verify(tmp_path, "linux-x64")
    repaired = package.repair_staged_verification(tmp_path, "linux-x64", first)

    assert calls["n"] == 1
    assert repaired == ""


def test_cross_target_staging_never_installs_host_libatomic(tmp_path, monkeypatch):
    monkeypatch.setattr("pm.packages.current_target", lambda: "linux-x64")

    def forbidden():
        raise AssertionError("cross-target stage attempted host mutation")

    monkeypatch.setattr("pm.libatomic.try_install_libatomic", forbidden)
    reason = Nodejs().repair_staged_verification(
        tmp_path,
        "linux-arm64",
        f"bin/node --version exited 127: {_LOADER}",
    )
    assert "libatomic.so.1" in reason


@pytest.mark.platforms("posix")
def test_successful_libatomic_install_does_not_mask_a_new_probe_error(tmp_path, monkeypatch):
    node = tmp_path / "bin" / "node"
    _node_script(node, f'#!/bin/sh\necho "{_LOADER}" >&2\nexit 127\n')
    monkeypatch.setattr("pm.packages.current_target", lambda: "linux-x64")

    def install():
        _node_script(node, "#!/bin/sh\necho different-failure >&2\nexit 23\n")
        return True, "install with sudo dnf install -y libatomic"

    monkeypatch.setattr("pm.libatomic.try_install_libatomic", install)
    package = Nodejs()
    first = package.verify(tmp_path, "linux-x64")
    repaired = package.repair_staged_verification(tmp_path, "linux-x64", first)

    assert "different-failure" in repaired
    assert "libatomic" not in repaired


def test_almalinux_prefers_dnf_package(monkeypatch):
    from pm import libatomic

    monkeypatch.setattr(libatomic, "_release_tokens", lambda: {"almalinux", "rhel"})
    monkeypatch.setattr(
        libatomic.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"dnf", "apt-get"} else None,
    )
    assert libatomic._host_install_command() == ("dnf", "install", "-y", "libatomic")


def test_nonroot_sudo_is_always_noninteractive(monkeypatch):
    from pm import libatomic

    monkeypatch.setattr(libatomic, "_is_root", lambda: False)
    monkeypatch.setattr(libatomic.shutil, "which", lambda name: "/usr/bin/sudo" if name == "sudo" else None)

    argv, attempt, remedy = libatomic._command_plan(("dnf", "install", "-y", "libatomic"))

    assert argv == ["/usr/bin/sudo", "-n", "dnf", "install", "-y", "libatomic"]
    assert attempt == "sudo -n dnf install -y libatomic"
    assert remedy == "sudo dnf install -y libatomic"


def test_missing_sudo_remedy_never_advertises_sudo(monkeypatch):
    from pm import libatomic

    monkeypatch.setattr(libatomic, "_is_root", lambda: False)
    monkeypatch.setattr(libatomic, "_host_install_command", lambda: ("dnf", "install", "-y", "libatomic"))
    monkeypatch.setattr(libatomic.shutil, "which", lambda _name: None)

    attempted, remedy = libatomic.try_install_libatomic()

    assert attempted is False
    assert "sudo" not in remedy
    assert "as root: dnf install -y libatomic" in remedy


def test_auto_repair_never_reads_a_tty(monkeypatch):
    from pm import libatomic

    monkeypatch.setattr(libatomic, "_is_root", lambda: False)
    monkeypatch.setattr(libatomic, "_host_install_command", lambda: ("dnf", "install", "-y", "libatomic"))
    monkeypatch.setattr(libatomic.shutil, "which", lambda name: "/usr/bin/sudo" if name == "sudo" else None)

    captured = {}

    def run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return type("Result", (), {"returncode": 1})()

    monkeypatch.setattr(libatomic.subprocess, "run", run)

    attempted, _remedy = libatomic.try_install_libatomic()

    assert attempted is False
    assert captured["argv"][:2] == ["/usr/bin/sudo", "-n"]
    assert captured["stdin"] is libatomic.subprocess.DEVNULL

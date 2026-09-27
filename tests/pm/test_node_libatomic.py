"""Official Node's libatomic.so.1 dependency is a host package, not a bad pin."""

from __future__ import annotations

from pathlib import Path

import pytest

from pm.libatomic import (
    install_command,
    parse_os_release,
    preferred_manager,
    repair_node_libatomic_failure,
    should_repair_node_libatomic,
    try_install_libatomic,
)
from pm.packages import Nodejs

_LOADER = (
    "bin/node: error while loading shared libraries: libatomic.so.1: "
    "cannot open shared object file: No such file or directory"
)


def _script(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def test_parse_os_release_strips_quotes() -> None:
    release = parse_os_release(
        'ID="almalinux"\nID_LIKE="rhel centos fedora"\n# comment\nNOTHING\n'
    )
    assert release == {"ID": "almalinux", "ID_LIKE": "rhel centos fedora"}


@pytest.mark.parametrize(
    ("text", "available", "manager"),
    [
        ('ID="almalinux"\nID_LIKE="rhel centos fedora"\n', ["apt-get", "dnf"], "dnf"),
        ("ID=ubuntu\nID_LIKE=debian\n", ["apt-get", "dnf"], "apt-get"),
        ("ID=fedora\n", ["yum"], "yum"),
        ("ID=opensuse-tumbleweed\nID_LIKE=\"opensuse suse\"\n", ["zypper"], "zypper"),
        ("ID=arch\n", ["pacman"], "pacman"),
        ("ID=alpine\n", ["apk"], "apk"),
        ("ID=almalinux\n", ["apt-get"], "apt-get"),
        ("ID=\n", [], None),
    ],
)
def test_preferred_manager_follows_the_distro_family(text, available, manager) -> None:
    assert preferred_manager(parse_os_release(text), available) == manager


def test_install_command_names_the_rpm_package_and_sudo() -> None:
    assert install_command("dnf", root=False) == "sudo dnf install -y libatomic"
    assert install_command("dnf", root=True) == "dnf install -y libatomic"
    assert install_command("apt-get", root=False) == "sudo apt-get install -y libatomic1"


def test_repair_is_only_the_native_glibc_probe() -> None:
    reason = f"/tmp/node --version exited 127: {_LOADER}"
    assert should_repair_node_libatomic(target="linux-x64", host="linux-x64", reason=reason)
    assert not should_repair_node_libatomic(target="linux-arm64", host="linux-x64", reason=reason)
    assert not should_repair_node_libatomic(
        target="linux-arm64-bionic", host="linux-arm64-bionic", reason=reason
    )
    assert not should_repair_node_libatomic(
        target="linux-x64", host="linux-x64", reason="node --version exited 1"
    )


def test_successful_install_retries_the_probe_once(monkeypatch) -> None:
    monkeypatch.setattr("pm.libatomic.try_install_libatomic", lambda: True)
    calls = {"n": 0}

    def retry() -> str:
        calls["n"] += 1
        return ""

    reason = repair_node_libatomic_failure(
        f"bin/node --version exited 127: {_LOADER}",
        target="linux-x64",
        host="linux-x64",
        retry=retry,
    )
    assert calls["n"] == 1
    assert reason == ""


def test_failed_install_keeps_the_probe_error_and_names_the_command(monkeypatch) -> None:
    monkeypatch.setattr("pm.libatomic.try_install_libatomic", lambda: False)
    monkeypatch.setattr(
        "pm.libatomic.remediation_for_host",
        lambda: "install it with `sudo dnf install -y libatomic`",
    )
    calls = {"n": 0}

    def retry() -> str:
        calls["n"] += 1
        return "still missing libatomic.so.1"

    reason = repair_node_libatomic_failure(
        f"bin/node --version exited 127: {_LOADER}",
        target="linux-x64",
        host="linux-x64",
        retry=retry,
    )
    assert calls["n"] == 0
    assert reason.endswith("— install it with `sudo dnf install -y libatomic`")
    assert "exited 127" in reason


@pytest.mark.platforms("posix")
def test_verify_installs_libatomic_then_accepts_node(tmp_path: Path, monkeypatch) -> None:
    node = tmp_path / "bin" / "node"
    _script(node, f'#!/bin/sh\necho "{_LOADER}" >&2\nexit 127\n')

    def install_and_fix() -> bool:
        _script(node, "#!/bin/sh\nexit 0\n")
        return True

    monkeypatch.setattr("pm.libatomic.try_install_libatomic", install_and_fix)
    monkeypatch.setattr("pm.packages.current_target", lambda: "linux-x64")
    assert Nodejs().verify(tmp_path, "linux-x64") == ""


@pytest.mark.platforms("posix")
def test_verify_names_the_dnf_command_when_install_cannot_run(
    tmp_path: Path, monkeypatch
) -> None:
    node = tmp_path / "bin" / "node"
    _script(node, f'#!/bin/sh\necho "{_LOADER}" >&2\nexit 127\n')
    probes = {"n": 0}
    real_verify = Nodejs.verify

    def counting_install() -> bool:
        probes["n"] += 1
        return False

    monkeypatch.setattr("pm.libatomic.try_install_libatomic", counting_install)
    monkeypatch.setattr(
        "pm.libatomic.remediation_for_host",
        lambda: "official Node links libatomic.so.1, which is not installed; "
        "install it with `sudo dnf install -y libatomic` and rerun `hermes update`",
    )
    monkeypatch.setattr("pm.packages.current_target", lambda: "linux-x64")
    reason = real_verify(Nodejs(), tmp_path, "linux-x64")
    assert probes["n"] == 1
    assert "sudo dnf install -y libatomic" in reason
    assert "libatomic.so.1" in reason
    assert "exited 127" in reason


def test_try_install_uses_passwordless_dnf_on_alma(monkeypatch) -> None:
    checks = {"n": 0}

    def present() -> bool:
        checks["n"] += 1
        return checks["n"] > 1

    ran: dict[str, list[str]] = {}
    monkeypatch.setattr("pm.libatomic.libatomic_present", present)
    monkeypatch.setattr(
        "pm.libatomic.read_os_release",
        lambda: {"ID": "almalinux", "ID_LIKE": "rhel centos fedora"},
    )
    monkeypatch.setattr("pm.libatomic.available_managers", lambda: ["dnf", "apt-get"])
    monkeypatch.setattr("pm.libatomic._is_root", lambda: False)
    monkeypatch.setattr("pm.libatomic._sudo_noninteractive_ok", lambda: True)
    monkeypatch.setattr("pm.libatomic.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("pm.libatomic._run_install", lambda argv: ran.__setitem__("argv", argv))
    assert try_install_libatomic() is True
    assert ran["argv"] == ["sudo", "-n", "dnf", "install", "-y", "libatomic"]


def test_try_install_prompts_on_a_tty_when_sudo_needs_a_password(monkeypatch) -> None:
    checks = {"n": 0}

    def present() -> bool:
        checks["n"] += 1
        return checks["n"] > 1

    ran: dict[str, list[str]] = {}
    monkeypatch.setattr("pm.libatomic.libatomic_present", present)
    monkeypatch.setattr("pm.libatomic.read_os_release", lambda: {"ID": "almalinux", "ID_LIKE": "rhel"})
    monkeypatch.setattr("pm.libatomic.available_managers", lambda: ["dnf"])
    monkeypatch.setattr("pm.libatomic._is_root", lambda: False)
    monkeypatch.setattr("pm.libatomic._sudo_noninteractive_ok", lambda: False)
    monkeypatch.setattr("pm.libatomic._stdin_is_tty", lambda: True)
    monkeypatch.setattr("pm.libatomic.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("pm.libatomic._run_install", lambda argv: ran.__setitem__("argv", argv))
    assert try_install_libatomic() is True
    assert ran["argv"] == ["sudo", "dnf", "install", "-y", "libatomic"]


def test_try_install_does_not_wait_for_a_password_without_a_tty(monkeypatch) -> None:
    monkeypatch.setattr("pm.libatomic.libatomic_present", lambda: False)
    monkeypatch.setattr("pm.libatomic.read_os_release", lambda: {"ID": "ubuntu", "ID_LIKE": "debian"})
    monkeypatch.setattr("pm.libatomic.available_managers", lambda: ["apt-get"])
    monkeypatch.setattr("pm.libatomic._is_root", lambda: False)
    monkeypatch.setattr("pm.libatomic._sudo_noninteractive_ok", lambda: False)
    monkeypatch.setattr("pm.libatomic._stdin_is_tty", lambda: False)
    monkeypatch.setattr("pm.libatomic.shutil.which", lambda name: f"/usr/bin/{name}")

    def refuse(argv: list[str]) -> None:
        raise AssertionError(argv)

    monkeypatch.setattr("pm.libatomic._run_install", refuse)
    assert try_install_libatomic() is False


def test_try_install_skips_the_package_manager_when_the_library_loads(monkeypatch) -> None:
    monkeypatch.setattr("pm.libatomic.libatomic_present", lambda: True)

    def refuse(argv: list[str]) -> None:
        raise AssertionError(argv)

    monkeypatch.setattr("pm.libatomic._run_install", refuse)
    assert try_install_libatomic() is True


@pytest.mark.platforms("posix")
def test_cross_target_verify_does_not_install(tmp_path: Path, monkeypatch) -> None:
    node = tmp_path / "bin" / "node"
    _script(node, "#!/bin/sh\nexit 0\n")
    called = {"n": 0}
    monkeypatch.setattr(
        "pm.libatomic.try_install_libatomic",
        lambda: called.__setitem__("n", called["n"] + 1) or True,
    )
    monkeypatch.setattr("pm.packages.current_target", lambda: "linux-x64")
    Nodejs().verify(tmp_path, "linux-arm64")
    assert called["n"] == 0

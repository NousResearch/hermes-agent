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
        ("ID=almalinux\n", ["apt-get"], None),
        ("ID=custom\n", ["apt-get", "dnf"], None),
        ("ID=custom\n", ["dnf"], "dnf"),
        ("ID=custom\n", ["dnf", "yum"], "dnf"),
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


@pytest.fixture
def node_store(tmp_path, monkeypatch):
    import hashlib
    import io
    import tarfile

    from pm import paths
    from pm.lock import Lockfile
    from pm.store import Store, current_target

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    lock = Lockfile(tmp_path / "lock.json")
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock.path)
    ready = tmp_path / "libatomic-ready"
    script = (
        f'#!/bin/sh\n[ -f "{ready}" ] && exit 0\n'
        f'echo "{_LOADER}" >&2\nexit 127\n'
    ).encode()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as payload:
        member = tarfile.TarInfo("node-v1.0/bin/node")
        member.size, member.mode = len(script), 0o755
        payload.addfile(member, io.BytesIO(script))
    data = archive.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    target = current_target()
    lock.set_pin("node", "1.0", {target: {
        "url": "https://example.invalid/node.tar.gz", "sha256": digest,
    }})
    lock.save()
    store = Store(paths.store_root())
    cached = store.entry(f"fetch-{digest}")
    cached.mkdir(parents=True)
    (cached / "node.tar.gz").write_bytes(data)
    monkeypatch.setattr("pm.libatomic.libatomic_present", ready.is_file)
    monkeypatch.setattr("pm.libatomic.read_os_release", lambda: {"ID": "almalinux"})
    monkeypatch.setattr("pm.libatomic.available_managers", lambda: ["dnf", "apt-get"])
    monkeypatch.setattr("pm.libatomic._is_root", lambda: True)
    return store, target, ready, script, digest


@pytest.mark.platforms("linux")
def test_verify_and_doctor_never_install_libatomic(node_store, monkeypatch, capsys):
    from pm import paths
    from pm.cli import cmd_doctor
    from pm.lock import Facts
    from pm.store import tree_digest

    store, target, _, script, digest = node_store
    package = Nodejs()
    entry = store.entry(package.store_entry("1.0", target))
    _script(entry / "bin/node", script.decode())
    facts = Facts(paths.facts_path())
    facts.record("node", "1.0", entry.name, package.env(entry, target), store.root,
                 target=target, artifacts=[digest], digest=tree_digest(entry))
    before = paths.facts_path().read_bytes()
    attempts = []
    monkeypatch.setattr("pm.libatomic._run_install", attempts.append)

    assert "libatomic.so.1" in package.verify(entry, target)
    assert cmd_doctor(None) == 1
    assert "installed but failed verification" in capsys.readouterr().out
    assert attempts == []
    assert paths.facts_path().read_bytes() == before


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_native_install_repairs_libatomic_or_keeps_actionable_error(
    node_store, monkeypatch, repair_succeeds
):
    import importlib

    from pm.package import InstallError

    install = importlib.import_module("pm.install")
    store, target, ready, _, _ = node_store
    attempts = []

    def install_library(argv):
        attempts.append(argv)
        if repair_succeeds:
            ready.touch()

    monkeypatch.setattr("pm.libatomic._run_install", install_library)
    facts = install._facts()
    if repair_succeeds:
        entry = install._install(Nodejs(), install._lockfile(), facts, store, target)
        assert Nodejs().verify(entry, target) == ""
        assert facts.get("node")["entry"] == entry.name
    else:
        with pytest.raises(InstallError, match="dnf install -y libatomic") as error:
            install._install(Nodejs(), install._lockfile(), facts, store, target)
        assert "exited 127" in str(error.value)
        assert facts.get("node") is None
    assert attempts == [["dnf", "install", "-y", "libatomic"]]


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

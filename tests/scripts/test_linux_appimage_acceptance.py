"""Receipts for the packaged x64 acceptance lanes; no release build or namespaces needed."""

import errno
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

try:
    import fcntl
except ImportError:
    fcntl = None

import pytest

from scripts.bundles import appimage_checks
from scripts.bundles.appimage_checks import check_execs, check_image, shipped_binary, trace_command, wait_window

pytestmark = pytest.mark.platforms("linux")


def test_digest_and_format_bind_both_lanes_to_packaged_x64(tmp_path):
    image = tmp_path / "Hermes.AppImage"
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    header[8:11] = b"AI\x02"
    header[18:20] = b"\x3e\x00"
    image.write_bytes(header)
    image.chmod(0o700)
    check_image(image, hashlib.sha256(header).hexdigest())
    with pytest.raises(ValueError, match="SHA-256"):
        check_image(image, "a" * 64)
    header[8:11] = bytes(3)
    image.write_bytes(header)
    with pytest.raises(ValueError, match="AppImage"):
        check_image(image, hashlib.sha256(header).hexdigest())


def test_worker_main_verifies_image_before_extraction(tmp_path, monkeypatch):
    bad = tmp_path / "not-an-appimage"
    bad.write_bytes(b"not an elf appimage")
    monkeypatch.setattr(appimage_checks.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(appimage_checks.socket, "if_nameindex", lambda: [(1, "lo")])
    extracted = []
    monkeypatch.setattr(appimage_checks.subprocess, "run", lambda *a, **kw: extracted.append(a))
    with pytest.raises(ValueError, match="type 2 AppImage"):
        appimage_checks.main("22.04", hashlib.sha256(bad.read_bytes()).hexdigest(), image=bad)
    assert not extracted


def _make_fake_ioctl(flags_by_name, ipv4_names=()):
    def fake_ioctl(_fd, request, arg):
        name = arg[:16].split(b"\0", 1)[0].decode("ascii")
        if request == appimage_checks._SIOCGIFFLAGS:
            if name not in flags_by_name:
                raise OSError(errno.ENODEV, "No such device")
            return struct.pack("16sH22x", name.encode("ascii"), flags_by_name[name])
        if request == appimage_checks._SIOCGIFADDR:
            if name in ipv4_names:
                return arg
            raise OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address")
        raise AssertionError(f"unexpected ioctl: {request:#x}")

    return fake_ioctl


def _write_proc_net(directory: Path, *, route: str = "", if_inet6: str = "", ipv6_route: str = "") -> None:
    (directory / "route").write_text(
        "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n" + route,
        encoding="ascii",
    )
    (directory / "if_inet6").write_text(
        "00000000000000000000000000000001 01 80 10 80       lo\n" + if_inet6,
        encoding="ascii",
    )
    (directory / "ipv6_route").write_text(
        "00000000000000000000000000000001 80 00000000000000000000000000000000 00 "
        "00000000000000000000000000000000 00000000 00000002 00000000 80200001       lo\n" + ipv6_route,
        encoding="ascii",
    )


def test_worker_main_allows_down_unaddressed_kernel_tunnels(tmp_path, monkeypatch):
    bad = tmp_path / "not-an-appimage"
    bad.write_bytes(b"not an elf appimage")
    monkeypatch.setattr(appimage_checks.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        appimage_checks.socket,
        "if_nameindex",
        lambda: [(1, "lo"), (2, "tunl0"), (3, "sit0"), (4, "ip6tnl0"), (5, "gre0")],
    )
    monkeypatch.setattr(
        fcntl,
        "ioctl",
        _make_fake_ioctl({"lo": 0x9, "tunl0": 0x80, "sit0": 0x80, "ip6tnl0": 0x80, "gre0": 0x80}, {"lo"}),
    )
    extracted = []
    monkeypatch.setattr(appimage_checks.subprocess, "run", lambda *a, **kw: extracted.append(a))
    with pytest.raises(ValueError, match="type 2 AppImage"):
        appimage_checks.main("22.04", hashlib.sha256(bad.read_bytes()).hexdigest(), image=bad)
    assert not extracted


def test_sandbox_check_refuses_root_active_addressed_routed_and_unreadable_network(tmp_path, monkeypatch):
    monkeypatch.setattr(appimage_checks.os, "geteuid", lambda: 0)
    assert "root" in (appimage_checks._sandbox_problem(tmp_path) or "")

    monkeypatch.setattr(appimage_checks.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(appimage_checks.socket, "if_nameindex", lambda: [(2, "tunl0")])
    assert "loopback" in (appimage_checks._sandbox_problem(tmp_path) or "")

    monkeypatch.setattr(appimage_checks.socket, "if_nameindex", lambda: [(1, "lo"), (2, "eth0")])
    monkeypatch.setattr(fcntl, "ioctl", _make_fake_ioctl({"lo": 0x8, "eth0": 0x0}, {"lo"}))
    assert "loopback" in (appimage_checks._sandbox_problem(tmp_path) or "")

    _write_proc_net(tmp_path)
    monkeypatch.setattr(fcntl, "ioctl", _make_fake_ioctl({"lo": 0x9, "eth0": 0x1043}, {"lo"}))
    assert "eth0" in (appimage_checks._sandbox_problem(tmp_path) or "")

    monkeypatch.setattr(fcntl, "ioctl", _make_fake_ioctl({"lo": 0x9, "eth0": 0x1002}, {"lo", "eth0"}))
    assert "IPv4" in (appimage_checks._sandbox_problem(tmp_path) or "")

    monkeypatch.setattr(fcntl, "ioctl", _make_fake_ioctl({"lo": 0x9, "eth0": 0x1002}, {"lo"}))
    _write_proc_net(tmp_path, route="eth0\t00000000\t0100000A\t0003\t0\t0\t0\t00000000\t0\t0\t0\n")
    assert "IPv4 routes" in (appimage_checks._sandbox_problem(tmp_path) or "")

    _write_proc_net(tmp_path, if_inet6="fd000000000000000000000000000001 02 40 00 c0 eth0\n")
    monkeypatch.setattr(appimage_checks, "_ipv6_available", lambda: True)
    assert "IPv6 addresses" in (appimage_checks._sandbox_problem(tmp_path) or "")

    _write_proc_net(
        tmp_path,
        ipv6_route="fd000000000000000000000000000000 40 00000000000000000000000000000000 00 "
        "00000000000000000000000000000000 00000100 00000001 00000000 00000001 eth0\n",
    )
    assert "IPv6 routes" in (appimage_checks._sandbox_problem(tmp_path) or "")

    _write_proc_net(tmp_path)
    assert appimage_checks._sandbox_problem(tmp_path) is None

    (tmp_path / "if_inet6").unlink()
    (tmp_path / "ipv6_route").unlink()
    assert "cannot verify" in (appimage_checks._sandbox_problem(tmp_path) or "")
    monkeypatch.setattr(appimage_checks, "_ipv6_available", lambda: False)
    assert appimage_checks._sandbox_problem(tmp_path) is None

    (tmp_path / "route").unlink()
    assert "cannot verify" in (appimage_checks._sandbox_problem(tmp_path) or "")

    def fail_ioctl(_fd, _request, _arg):
        raise PermissionError("blocked")

    monkeypatch.setattr(fcntl, "ioctl", fail_ioctl)
    assert "cannot verify" in (appimage_checks._sandbox_problem(tmp_path) or "")


def _engine(backend):
    return Path(f"/work/payload/tools/llamacpp-{backend}-123-linux-x64/llama-server")


def _exec(binary, rc=0):
    return f'execve("{binary}", ["{binary}", "--version"], 0x10 /* 2 vars */) = {rc}\n'


def test_both_distinct_native_execs_are_required(tmp_path):
    trace = tmp_path / "probe.41"
    required = {_engine("cuda"), _engine("hip")}
    trace.write_text(_exec(_engine("cuda")) + _exec(_engine("hip"), -1))
    with pytest.raises(ValueError, match="hip"):
        check_execs(tmp_path, "probe", required, floor=False)
    trace.write_text(_exec(_engine("cuda")) + _exec(_engine("hip")))
    check_execs(tmp_path, "probe", required, floor=False)
    trace.write_text(_exec(_engine("cuda")) + _exec(_engine("hip")).replace("--version", "--help"))
    with pytest.raises(ValueError, match="hip"):
        check_execs(tmp_path, "probe", required, floor=False)


@pytest.mark.parametrize("attempt", [
    _exec(_engine("cuda"), -1), _exec("./llama-server", -1),
    'execveat(3</work/payload/tools/llamacpp-hip-123-linux-x64/llama-server>, "", ["server"], 0x0, AT_EMPTY_PATH) = -1 EACCES\n',
])
def test_floor_refuses_attempt_even_if_app_swallows_error(tmp_path, attempt):
    trace = tmp_path / "probe.77"
    cpu = _engine("cpu")
    trace.write_text(_exec(cpu) + 'execve("/bin/echo", ["echo", "llamacpp-hip-123-linux-x64"], 0x0) = 0\n')
    check_execs(tmp_path, "probe", {cpu}, floor=True)
    (tmp_path / "probe.78").write_text(attempt)
    with pytest.raises(ValueError, match="execution"):
        check_execs(tmp_path, "probe", {cpu}, floor=True)


def test_desktop_that_dropped_the_chromium_sandbox_is_refused(tmp_path):
    app = "/tmp/appimage_extracted_1/Hermes Agent"
    (tmp_path / "desktop.9").write_text(
        f'execve("{app}", ["{app}", "--disable-setuid-sandbox"], 0x10 /* 2 vars */) = 0\n'
        f'execve("{app}", ["{app}", "--no-sandbox"], 0x10 /* 2 vars */) = -1 ENOENT (No such file or directory)\n')
    check_execs(tmp_path, "desktop", set(), floor=True)
    (tmp_path / "desktop.10").write_text(
        f'execve("{app}", ["{app}", "--type=zygote", "--no-sandbox"], 0x10 /* 2 vars */) = 0\n')
    with pytest.raises(ValueError, match="without its sandbox"):
        check_execs(tmp_path, "desktop", set(), floor=True)


def test_no_trace_is_not_proof_no_engine_ran(tmp_path):
    with pytest.raises(ValueError, match="trace"):
        check_execs(tmp_path, "desktop", set(), floor=True)


def test_shipped_engine_cannot_be_missing_or_link_to_another_engine(tmp_path):
    name = "llamacpp-cuda"
    entry = tmp_path / f"{name}-123-linux-x64"
    entry.mkdir()
    facts = {name: {"entry": entry.name, "target": "linux-x64", "version": "123"}}
    with pytest.raises(FileNotFoundError):
        shipped_binary(tmp_path, facts, name, "llama-server")
    (entry / "llama-server").symlink_to(Path(sys.executable))
    with pytest.raises(ValueError, match="escapes"):
        shipped_binary(tmp_path, facts, name, "llama-server")
    (entry / "llama-server").unlink()
    entry.rmdir()
    other = tmp_path / "llamacpp-cpu-123-linux-x64"
    other.mkdir()
    shutil.copy2("/usr/bin/true", other / "llama-server")
    entry.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="redirected"):
        shipped_binary(tmp_path, facts, name, "llama-server")


@pytest.mark.skipif(not Path("/usr/bin/strace").is_file(), reason="optional system strace")
def test_trace_of_real_pm_probe_and_failed_gpu_exec(tmp_path):
    cpu = tmp_path / "llamacpp-cpu-123-linux-x64" / "llama-server"
    cpu.parent.mkdir()
    shutil.copy2("/usr/bin/true", cpu)
    code = ("import sys, pm; from pathlib import Path; "
            "reason = pm.get_package('llamacpp-cpu').verify(Path(sys.argv[1]), pm.current_target()); "
            "sys.exit(reason or 0)")
    subprocess.run([*trace_command(tmp_path, "probe"), sys.executable, "-c", code, cpu.parent], check=True)
    check_execs(tmp_path, "probe", {cpu}, floor=True)
    subprocess.run([*trace_command(tmp_path, "desktop"), sys.executable, "-c",
                    "import subprocess,sys\ntry: subprocess.run([sys.argv[1]])\nexcept FileNotFoundError: pass",
                    _engine("hip")], check=True)
    with pytest.raises(ValueError, match="engine execution"):
        check_execs(tmp_path, "desktop", set(), floor=True)


@pytest.mark.parametrize("byte", ["r", "x", ""])
def test_main_window_receipt_is_required(byte):
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "rb") as ready:
        proc = subprocess.Popen([sys.executable, "-c", "import os,sys; os.write(int(sys.argv[1]), sys.argv[2].encode())",
                                 str(write_fd), byte], pass_fds=(write_fd,))
        os.close(write_fd)
        try:
            if byte == "r":
                wait_window(ready, 5)
            else:
                with pytest.raises(ValueError, match="window"):
                    wait_window(ready, 5)
        finally:
            proc.wait(timeout=5)

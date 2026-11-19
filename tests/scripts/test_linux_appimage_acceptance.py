"""Receipts for the packaged x64 acceptance lanes; no release build or namespaces needed."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.bundles.linux_appimage_acceptance import check_image
from scripts.bundles.appimage_checks import check_execs, shipped_binary, trace_command, wait_window

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

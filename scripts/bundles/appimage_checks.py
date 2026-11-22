"""Acceptance worker for scripts/ci/linux_x64_userland.sh accept. Run only inside its offline sandbox."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import socket
import struct
import subprocess
import sys
import time
from urllib.request import Request, urlopen

try:
    import fcntl
except ImportError:
    fcntl = None


def check_image(image: Path, sha256: str) -> None:
    with image.open("rb") as stream:
        header = stream.read(64)
        if (header[:6] != b"\x7fELF\x02\x01" or header[8:11] != b"AI\x02"
                or header[18:20] != b"\x3e\x00"):
            raise ValueError("expected a packaged x86-64 type 2 AppImage")
        digest = hashlib.sha256(header)
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise ValueError("AppImage SHA-256 does not match the producer")


_TRACE = re.compile(r'^execve(?:at\(([^,]+), |\()"([^"\n]*)", (.*)')
_GPU = re.compile(r'(?:^|/)llamacpp-(?:cuda|hip)-[^/]+(?:/|$)')
_ALLOWED = re.compile(r'/llamacpp-(?:cpu|vulkan)-[^/]+-linux-x64/llama-server$')


def trace_command(directory: Path, phase: str) -> list[str]:
    return ["/usr/bin/strace", "-I", "2", "-ff", "-yy", "-s", "4096", "-e", "trace=execve,execveat",
            "-o", str(directory / phase)]


def check_execs(directory: Path, phase: str, required: set[Path], *, floor: bool) -> None:
    missing = {str(binary) for binary in required}
    observed = False
    for trace in directory.glob(f"{phase}.[0-9]*"):
        with trace.open(encoding="ascii", errors="backslashreplace") as stream:
            for line in stream:
                if not line.startswith(("execve(", "execveat(")):
                    continue
                match = _TRACE.match(line)
                if not match:
                    raise ValueError(f"{phase}: unreadable execution trace")
                fd, exe, rest = match.groups()
                if fd and not exe.startswith("/"):
                    parent = re.search(r'<(/[^>]*)>$', fd)
                    if not parent:
                        raise ValueError(f"{phase}: unresolved execveat in execution trace")
                    exe = str(Path(parent[1]) / exe)
                observed = True
                if '"--no-sandbox"' in rest.split("], ", 1)[0] and rest.rstrip().endswith(" = 0"):
                    raise ValueError(f"{phase}: Chromium ran without its sandbox: {exe}")
                if floor and (_GPU.search(exe) or (Path(exe).name == "llama-server" and not _ALLOWED.search(exe))):
                    raise ValueError(f"{phase}: incompatible or unidentified engine execution: {exe}")
                if re.match(r'^\["[^"]*", "--version"\], ', rest) and rest.rstrip().endswith(" = 0"):
                    missing.discard(exe)
    if not observed:
        raise ValueError(f"{phase}: missing execution trace")
    if missing:
        raise ValueError(f"{phase}: no successful --version exec for {', '.join(sorted(missing))}")


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if Path(relative).is_absolute() or not path.is_relative_to(root) or path == root:
        raise ValueError("runtime path escapes the packaged payload")
    return path


def shipped_binary(tools: Path, facts: dict, name: str, relative: str) -> Path:
    row = facts[name]
    if row["target"] != "linux-x64" or row["entry"] != f"{name}-{row['version']}-linux-x64":
        raise ValueError(f"invalid packaged {name} fact")
    entry = contained(tools, row["entry"])
    if entry != tools / row["entry"]:
        raise ValueError(f"redirected packaged {name} entry")
    binary = contained(entry, relative)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"missing executable for packaged {name}")
    return binary


_ENGINE_CHECK = """\
import sys, site
from pathlib import Path
repo, site_dir, tools, *paths = sys.argv[1:]
sys.path[:0] = [repo, site_dir]
site.addsitedir(site_dir)
import pm
from pm import paths as pm_paths
from hermes_cli.local_runtime.binaries import installed_engine
if pm_paths.repo_root() != Path(repo) or pm_paths.store_root() != Path(tools) or pm.current_target() != 'linux-x64':
    raise SystemExit('PM did not resolve the packaged linux-x64 runtime')
for path in map(Path, paths):
    backend = path.parent.name.split('-')[1]
    engine = installed_engine('auto' if backend == 'cpu' else backend, allow_outdated=False)
    if engine is None or engine.backend != backend or engine.binary != path:
        raise SystemExit(f'packaged runtime did not select the pinned {backend} engine')
    reason = pm.get_package('llamacpp-' + backend).verify(path.parent, pm.current_target())
    if reason:
        raise SystemExit(f'{backend} failed its PM executable check: {reason}')
"""


def check_payload(work: Path, payload: Path, floor: bool) -> None:
    manifest = json.loads((payload / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest["target"] != "linux-x64":
        raise ValueError("AppImage does not contain a linux-x64 payload")
    runtime = manifest["runtime"]
    python, repo, site, tools = (contained(payload, runtime[key]) for key in
                                ("storePython", "repoDir", "sitePackages", "toolsDir"))
    facts = json.loads((tools / "facts.json").read_text(encoding="utf-8-sig"))["packages"]
    engines = {key: shipped_binary(tools, facts, f"llamacpp-{key}", "llama-server") for key in ("cpu", "cuda", "hip")}
    if floor:
        node = shipped_binary(tools, facts, "node", "bin/node")
        subprocess.run([*trace_command(work, "smoke"), "bash", "/smoke-payload.sh", payload],
                       env={**os.environ, "PATH": f"{node.parent}:{os.environ['PATH']}"}, check=True, timeout=780)
        check_execs(work, "smoke", set(), floor=True)
    selected = [engines["cpu"]] if floor else [engines["cuda"], engines["hip"]]
    subprocess.run([*trace_command(work, "engines"), python, "-I", "-B", "-S", "-c", _ENGINE_CHECK,
                    repo, site, tools, *selected], check=True, timeout=180)
    check_execs(work, "engines", set(selected), floor=floor)
    if floor:
        subprocess.run([*trace_command(work, "inference"), sys.executable, "-I", __file__,
                        "cpu-completion", engines["cpu"]], check=True, timeout=90)
        check_execs(work, "inference", set(), floor=True)


def cpu_completion(server: Path, model: Path) -> None:
    """Start the selected CPU server and require a real generation over loopback."""
    if hashlib.sha256(model.read_bytes()).hexdigest() != '270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d':
        raise ValueError("wrong CPU smoke model digest")
    # A private network namespace per acceptance run: this port has no host listener.
    base = 'http://127.0.0.1:18080'
    proc = subprocess.Popen([server, '-m', model, '--host', '127.0.0.1', '--port', '18080',
                             '--n-gpu-layers', '0', '--threads', '1', '--ctx-size', '128', '--parallel', '1'],
                            cwd=server.parent, stdin=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 45
        while proc.poll() is None and time.monotonic() < deadline:
            try:
                with urlopen(base + '/health', timeout=1) as response:
                    if json.load(response).get('status') == 'ok':
                        break
            except OSError:
                pass  # Refused during startup; HTTP 503 while loading the model.
            time.sleep(0.15)
        else:
            raise RuntimeError(f"packaged CPU server never became ready (exit={proc.poll()})")
        body = json.dumps({'prompt': 'Once upon a time', 'n_predict': 3, 'temperature': 0, 'ignore_eos': True}).encode()
        request = Request(base + '/completion', data=body, headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=30) as response:
            reply = json.load(response)
        if reply.get('tokens_predicted') != 3 or not isinstance(reply.get('content'), str) or not reply['content']:
            raise ValueError("packaged CPU engine did not generate a completion")
        print('Offline CPU completion: 3 tokens from the packaged engine', flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def wait_window(ready, timeout: float) -> None:
    if not select.select([ready], [], [], timeout)[0] or ready.read(1) != b"r":
        raise ValueError("desktop did not signal main window reveal")


def check_desktop(work: Path, image: Path) -> None:
    if subprocess.run(["unshare", "-Ur", "true"], stderr=subprocess.DEVNULL, check=False, timeout=30).returncode:
        raise ValueError("this container refuses the user namespace Chromium's sandbox needs")
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "rb") as ready:
        with os.fdopen(write_fd, "wb") as writer:
            proc = subprocess.Popen([*trace_command(work, "desktop"), image, "--appimage-extract-and-run",
                                     "--disable-setuid-sandbox"], pass_fds=(writer.fileno(),),
                                    env={**os.environ, "HERMES_DESKTOP_READY_FD": str(writer.fileno())},
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_window(ready, 240)
            log = Path(os.environ["HERMES_HOME"]) / "logs" / "desktop.log"
            deadline = time.monotonic() + 180
            while not (log.is_file() and b"[backend] Hermes gateway is ready" in log.read_bytes()):
                if proc.poll() is not None or time.monotonic() >= deadline:
                    raise ValueError("desktop did not report a ready local gateway")
                time.sleep(0.25)
        finally:
            # The enclosing private PID namespace disposes of the app and its detached gateway.
            proc.terminate()
            proc.wait(timeout=15)
    check_execs(work, "desktop", set(), floor=True)


_SIOCGIFFLAGS = 0x8913
_SIOCGIFADDR = 0x8915
_IFF_UP = 0x1
_IFF_LOOPBACK = 0x8


def _ifreq(name: str) -> bytes:
    raw = name.encode("ascii")
    if not raw or len(raw) > 15:
        raise ValueError(f"invalid interface name: {name!r}")
    return struct.pack("16s24x", raw)


def _iface_flags(sock: socket.socket, name: str) -> int:
    if fcntl is None:
        raise OSError("fcntl is unavailable")
    packed = fcntl.ioctl(sock.fileno(), _SIOCGIFFLAGS, _ifreq(name))
    return struct.unpack_from("H", packed, 16)[0]


def _iface_has_ipv4(sock: socket.socket, name: str) -> bool:
    if fcntl is None:
        raise OSError("fcntl is unavailable")
    try:
        fcntl.ioctl(sock.fileno(), _SIOCGIFADDR, _ifreq(name))
    except OSError as exc:
        if exc.errno == errno.EADDRNOTAVAIL:
            return False
        raise
    return True


def _ipv6_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM):
            return True
    except OSError:
        return False


def _proc_net_ifaces(path: Path, column: int, *, skip_header: bool = False) -> set[str]:
    lines = path.read_text(encoding="ascii").splitlines()
    if skip_header:
        lines = lines[1:]
    return {fields[column] for line in lines if len(fields := line.split()) > column}


def _non_loopback_problem(sock: socket.socket, extra: list[str], proc_net: Path) -> str | None:
    # Kernels with tunnel drivers add down, unaddressed fallback devices (tunl0, sit0, ...) to --network none.
    active = [name for name in extra if _iface_flags(sock, name) & _IFF_UP]
    if active:
        return f"non-loopback interfaces are up: {', '.join(active)}"
    addressed_v4 = [name for name in extra if _iface_has_ipv4(sock, name)]
    if addressed_v4:
        return f"non-loopback interfaces have IPv4 addresses: {', '.join(addressed_v4)}"
    extra_set = set(extra)
    routed_v4 = sorted(extra_set & _proc_net_ifaces(proc_net / "route", 0, skip_header=True))
    if routed_v4:
        return f"non-loopback IPv4 routes: {', '.join(routed_v4)}"
    if _ipv6_available():
        addressed_v6 = sorted(extra_set & _proc_net_ifaces(proc_net / "if_inet6", 5))
        if addressed_v6:
            return f"non-loopback interfaces have IPv6 addresses: {', '.join(addressed_v6)}"
        routed_v6 = sorted(extra_set & _proc_net_ifaces(proc_net / "ipv6_route", 9))
        if routed_v6:
            return f"non-loopback IPv6 routes: {', '.join(routed_v6)}"
    return None


def _sandbox_problem(proc_net: Path = Path("/proc/self/net")) -> str | None:
    if not hasattr(os, "geteuid") or fcntl is None or os.geteuid() == 0:
        return "running as root or unsupported platform"
    try:
        names = {name for _, name in socket.if_nameindex()}
        if "lo" not in names:
            return f"missing loopback interface ({sorted(names)})"
        extra = sorted(names - {"lo"})
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            lo_flags = _iface_flags(sock, "lo")
            if not (lo_flags & _IFF_UP) or not (lo_flags & _IFF_LOOPBACK):
                return f"loopback interface is not up (flags=0x{lo_flags:x})"
            if extra:
                return _non_loopback_problem(sock, extra, proc_net)
    except (OSError, ValueError) as exc:
        return f"cannot verify network isolation: {exc}"
    return None


def main(ubuntu: str, sha256: str, *, image: Path = Path("/image.AppImage")) -> None:
    problem = _sandbox_problem()
    if problem is not None:
        raise SystemExit(f"refusing to run outside the offline non-root sandbox ({problem})")
    work = Path.cwd()
    check_image(image, sha256)
    subprocess.run([image, "--appimage-extract"], stdout=subprocess.DEVNULL, check=True, timeout=300)
    root = work / "squashfs-root"
    check_payload(work, contained(root, "resources/agent-payload"), ubuntu == "22.04")
    if ubuntu == "22.04":
        shutil.rmtree(root)
        check_desktop(work, image)


if __name__ == "__main__":
    if sys.argv[1] == 'cpu-completion':
        cpu_completion(Path(sys.argv[2]), Path('/stories260K.gguf'))
    else:
        main(sys.argv[1], sys.argv[2])

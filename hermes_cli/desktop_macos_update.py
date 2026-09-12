"""Private macOS handoff: validate, exchange, observe adoption, retain rollback.

Reader evidence covers the updater's effective UID and libproc-exposed vnode
regions. Other users, unexposed VM submaps/pagers, and launches after the final
scan are outside that observation. This is not a launch lock or crash recovery.
"""
from __future__ import annotations

import ast
import argparse
import ctypes
import datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass


def command(*args, timeout=30):
    result = subprocess.run([str(a) for a in args], capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{Path(args[0]).name} failed ({result.returncode})")
    return result.stdout.decode("utf-8", errors="strict").strip()


def expectation(root):
    root = Path(root)
    sha = command("git", "-C", root, "rev-parse", "HEAD^{commit}")
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or sha == "0" * 40:
        raise ValueError("source has no real commit")
    if command("git", "-C", root, "status", "--porcelain", "-uno"):
        raise ValueError("tracked source differs from expected commit")
    package = json.loads((root / "apps/desktop/package.json").read_text(encoding="utf-8"))
    version = None
    for node in ast.parse((root / "hermes_cli/__init__.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
            version = ast.literal_eval(node.value)
    if not isinstance(version, str) or not version:
        raise ValueError("missing expected backend version")
    return dict(sha=sha, version=package["version"], identity=package["build"]["appId"],
                arch=platform.machine(), backend_version=version)


@dataclass(frozen=True)
class Fingerprint:
    tree: str
    executable: str
    archive: str
    stamp: str


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(app):
    app = Path(app).resolve(strict=True)
    tree = hashlib.sha256()
    for path in sorted(app.rglob("*")):
        info = path.lstat()
        tree.update(json.dumps([str(path.relative_to(app)), stat.S_IMODE(info.st_mode)]).encode())
        if path.is_symlink():
            if not path.resolve(strict=True).is_relative_to(app):
                raise ValueError("bundle symlink escapes its tree")
            tree.update(b"L" + os.fsencode(os.readlink(path)))
        elif stat.S_ISREG(info.st_mode):
            tree.update(b"F" + file_hash(path).encode())
        elif stat.S_ISDIR(info.st_mode):
            tree.update(b"D")
        else:
            raise ValueError("unsupported bundle file type")
    resources = app / "Contents/Resources"
    return Fingerprint(tree.hexdigest(), file_hash(app / "Contents/MacOS/Hermes"),
                       file_hash(resources / "app.asar"), file_hash(resources / "install-stamp.json"))


def verify_signature(app):
    command("/usr/bin/codesign", "--verify", "--deep", "--strict", app)


def validate(app, expected):
    app = Path(app)
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    if (info.get("CFBundleIdentifier") != expected["identity"]
            or info.get("CFBundleExecutable") != "Hermes"
            or info.get("CFBundleShortVersionString") != expected["version"]):
        raise ValueError("candidate bundle identity or version mismatch")
    exe = app / "Contents/MacOS/Hermes"
    if not os.access(exe, os.X_OK) or expected["arch"] not in command("/usr/bin/lipo", "-archs", exe).split():
        raise ValueError("candidate architecture mismatch")
    stamp = json.loads((app / "Contents/Resources/install-stamp.json").read_text(encoding="utf-8"))
    if (type(stamp.get("schemaVersion")) is not int or stamp["schemaVersion"] != 1
            or stamp.get("commit") != expected["sha"] or stamp.get("dirty") is not False
            or stamp.get("source") not in ("ci", "local")):
        raise ValueError("candidate stamp is stale, dirty, unsupported or fallback")
    built = datetime.datetime.fromisoformat(stamp["builtAt"].replace("Z", "+00:00"))
    now = datetime.datetime.now(datetime.timezone.utc)
    if built.tzinfo is None or built.year < 2000 or built > now + datetime.timedelta(seconds=60):
        raise ValueError("invalid build timestamp")
    verify_signature(app)
    return fingerprint(app)


def select_candidate(root, expected):
    candidates = []
    for app in sorted((Path(root) / "apps/desktop/release").glob("mac*/Hermes.app")):
        # A mismatched/stale directory must not win by ordering or modification time.
        try:
            validate(app, expected)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError):
            continue
        candidates.append(app)
    if len(candidates) != 1:
        raise ValueError("missing or ambiguous compatible candidate")
    return candidates[0]


def exchange(left, right):
    left, right = Path(left), Path(right)
    if left.stat().st_dev != right.stat().st_dev:
        raise ValueError("atomic exchange requires the same volume")
    libc = ctypes.CDLL(None, use_errno=True)
    swap = libc.renamex_np
    swap.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    swap.restype = ctypes.c_int
    # RENAME_SWAP | RENAME_NOFOLLOW_ANY. No destructive two-rename fallback.
    if swap(os.fsencode(left), os.fsencode(right), 2 | 16):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def preflight(root, target):
    root, target = Path(root).resolve(strict=True), Path(target)
    if target.is_symlink() or any(p.is_symlink() for p in target.parents):
        raise ValueError("symlinked publication target")
    target = target.resolve(strict=True)
    if (target.suffix != ".app" or not target.is_dir()
            or target.is_relative_to(root) or root.is_relative_to(target)):
        raise ValueError("unsupported or overlapping source/target layout")
    with tempfile.TemporaryDirectory(prefix=".hermes-swap-probe-", dir=target.parent) as directory:
        left, right = Path(directory) / "left", Path(directory) / "right"
        left.mkdir()
        right.mkdir()
        exchange(left, right)
    return target


def native():
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    lib.proc_listpids.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
    lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
    lib.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    return lib


@dataclass(frozen=True)
class Process:
    pid: int
    start: tuple
    parent: int
    executable: Path | None


def process_record(lib, pid):
    # Darwin proc_bsdinfo ABI, shared by supported arm64/x86_64 macOS.
    data = ctypes.create_string_buffer(136)
    ctypes.set_errno(0)
    count = lib.proc_pidinfo(pid, 3, 0, data, len(data))
    if count == 0 and ctypes.get_errno() == errno.ESRCH:
        return None
    if count != len(data):
        raise RuntimeError("incomplete process identity scan")
    status, = struct.unpack_from("=I", data, 4)
    found, parent, uid = struct.unpack_from("=III", data, 12)
    if found != pid or uid != os.geteuid():  # windows-footgun: ok — private Darwin observer
        raise RuntimeError("process identity or credentials changed")
    if status == 5:  # SZOMB: its address space has already exited.
        return None
    start = struct.unpack_from("=QQ", data, 120)
    path = ctypes.create_string_buffer(4096)
    ctypes.set_errno(0)
    count = lib.proc_pidpath(pid, path, len(path))
    if count == 0 and ctypes.get_errno() == errno.ENOENT:
        # A removed executable (e.g. another app's completed update) still has
        # vnode-backed mappings. Observe those; never treat this as an exited PID.
        return Process(pid, start, parent, None)
    if count <= 0 or count >= len(path):
        raise RuntimeError("executable identity unavailable")
    return Process(pid, start, parent, Path(os.fsdecode(path.value)))


def process_table():
    lib = native()
    for _ in range(3):
        needed = lib.proc_listpids(4, os.geteuid(), None, 0)  # windows-footgun: ok — Darwin PROC_UID_ONLY (effective UID)
        if needed <= 0 or needed > 4_000_000:
            raise RuntimeError("process enumeration failed")
        data = ctypes.create_string_buffer(needed + 4096)
        count = lib.proc_listpids(4, os.geteuid(), data, len(data))  # windows-footgun: ok — private Darwin observer
        if count == len(data):
            continue
        if count <= 0 or count > len(data) or count % 4:
            raise RuntimeError("malformed process enumeration")
        rows = {}
        for pid in struct.unpack_from(f"={count // 4}I", data):
            if pid:
                row = process_record(lib, pid)
                if row is not None:
                    rows[pid] = row
        return rows
    raise RuntimeError("process enumeration overflow")


@dataclass
class Observation:
    state: str
    readers: dict
    processes: dict
    reason: str = ""


def mapped_files(lib, process, deadline):
    address = 0
    files = {}
    for _ in range(100_000):
        if time.monotonic() > deadline:
            raise RuntimeError("reader scan timed out")
        data = ctypes.create_string_buffer(1272)  # proc_regionwithpathinfo
        ctypes.set_errno(0)
        count = lib.proc_pidinfo(process.pid, 8, address, data, len(data))
        error = ctypes.get_errno()
        if count == 0 and error in (errno.EINVAL, errno.ESRCH):
            current = process_record(lib, process.pid)
            if current is None:
                return {}
            if error == errno.EINVAL and current.start == process.start:
                if current.executable is None and not files:
                    raise RuntimeError("neither executable nor mapped identity available")
                return files  # Flavor 8's end-of-map; flavor 22 conflates other errors.
            raise RuntimeError("process changed during reader scan")
        if count != len(data):
            raise RuntimeError("incomplete or denied mapped-file scan")
        start, size = struct.unpack_from("=QQ", data, 80)
        tag, = struct.unpack_from("=I", data, 32)
        device, = struct.unpack_from("=I", data, 96)
        inode, = struct.unpack_from("=Q", data, 104)
        path = bytes(data[248:1272]).split(b"\0", 1)[0]
        if not size and start > address and tag == 31 and not inode and not path:
            # Darwin can return a zero-length hint when the cursor is in a gap.
            # Requery its exact address; skip no bytes and require forward progress.
            address = start
            continue
        if not size or start + size <= address:
            raise RuntimeError("malformed mapped-file region")
        if inode:
            if not path:
                raise RuntimeError("mapped vnode path unavailable")
            files[(device, inode)] = Path(os.fsdecode(path))
        # A zero vnode covers anonymous memory and kernel-unexposed submaps/pagers;
        # it is explicitly outside this observer's file-identity guarantee.
        address = start + size
    raise RuntimeError("mapped-file scan exceeded region limit")


def observe(bundles):
    bundles = [Path(p) for p in bundles]
    readers = {p: set() for p in bundles}
    rows = {}
    try:
        identities = {}
        for bundle in bundles:
            identities[bundle] = {(s.st_dev, s.st_ino) for p in bundle.rglob("*")
                                  if p.is_file() for s in [p.stat()]}
        lib, rows = native(), process_table()
        deadline = time.monotonic() + 30
        for pid, row in rows.items():
            files = mapped_files(lib, row, deadline)
            current = process_record(lib, pid)
            if current is None:
                continue
            if current.start != row.start or current.executable != row.executable:
                raise RuntimeError("process changed during reader scan")
            for bundle in bundles:
                if (files.keys() & identities[bundle]
                        or any(path.is_relative_to(bundle) for path in files.values())
                        or (row.executable is not None and row.executable.is_relative_to(bundle))):
                    readers[bundle].add(pid)
        missing_paths = sum(row.executable is None for row in rows.values())
        note = f"{missing_paths} removed executable paths observed through mapped identities" if missing_paths else ""
        return Observation("present" if any(readers.values()) else "none", readers, rows, note)
    except (OSError, RuntimeError, ValueError) as exc:
        return Observation("unknown", readers, rows, str(exc))


def scan_readers(bundles):
    # A PID may exec between enumeration and inspection. Discard that entire
    # unknown observation and retry from scratch, at most three times. Never
    # retry presence, permission errors, malformed data, or other uncertainty.
    for _ in range(3):
        result = observe(bundles)
        if result.state != "unknown" or result.reason != "process changed during reader scan":
            return result
    return result


class LogCursor:
    def __init__(self, path):
        self.path = Path(path)
        self.identity = None
        self.prefix = b""
        self.last_size = 0
        if self.path.exists():
            with self.path.open("rb") as stream:
                info = os.fstat(stream.fileno())
                self.identity = (info.st_dev, info.st_ino)
                self.prefix = stream.read()
                self.last_size = len(self.prefix)

    def read(self):
        if not self.path.exists() and self.identity is None:
            return ""
        with self.path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            identity = (info.st_dev, info.st_ino)
            if self.identity is None:
                self.identity = identity
            if identity != self.identity or info.st_size < self.last_size:
                raise RuntimeError("Desktop log rotated or truncated")
            data = stream.read()
        if not data.startswith(self.prefix):
            raise RuntimeError("Desktop log history changed")
        self.last_size = len(data)
        return data[len(self.prefix):].decode("utf-8", errors="strict")


def clear_marker(marker, owner):
    try:
        lines = Path(marker).read_text(encoding="utf-8").splitlines()
        if lines and lines[0].strip() == str(owner):
            Path(marker).unlink()
    except FileNotFoundError:
        return


def launch_app(app):
    command("/usr/bin/open", app)


def descendants(rows, main):
    tree = {main.pid: main}
    for _ in range(len(rows)):
        added = {pid: row for pid, row in rows.items() if row.parent in tree and pid not in tree}
        if not added:
            return tree
        tree.update(added)
    raise RuntimeError("cyclic process ancestry")


def readiness(text, tree, expected, not_before=0):
    events = []
    for line in text.splitlines():
        match = re.fullmatch(r"\[([^\]]+)\] \[hermes\] (.*)", line)
        if not match:
            continue
        try:
            stamp = datetime.datetime.fromisoformat(match[1].replace("Z", "+00:00"))
            if stamp.tzinfo is None or not not_before <= stamp.timestamp() <= time.time() + 1:
                continue
        except ValueError:
            continue
        events.append(match[2])
    remote = "[boot] Remote Hermes backend is ready" in events
    local = "[boot] Hermes backend is ready. Finalizing desktop startup" in events
    ports = {match[1] for line in events
             if (match := re.fullmatch(r"HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)", line))}
    if remote and (local or ports) or len(ports) > 1:
        raise RuntimeError("ambiguous fresh readiness evidence")
    if remote:
        return ("remote",)
    if not local or len(ports) != 1:
        return None
    port = int(next(iter(ports)))
    if not 1 <= port <= 65535:
        raise RuntimeError("invalid announced port")
    import psutil
    listeners = []
    for pid, row in tree.items():
        try:
            for connection in psutil.Process(pid).net_connections(kind="tcp"):
                if (connection.status == psutil.CONN_LISTEN and connection.laddr.port == port
                        and connection.laddr.ip in ("127.0.0.1", "::1")):
                    listeners.append((pid, row.start))
        except psutil.NoSuchProcess:
            return None
        except psutil.AccessDenied as exc:
            raise RuntimeError("backend listener identity unavailable") from exc
    if len(listeners) != 1:
        return None
    # No proxy or credentials; local health endpoint is public. Its socket must
    # belong to this launch's descendant, not an unrelated healthy server.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/api/health", timeout=0.5) as response:
            body = json.loads(response.read(65536))
            if response.status != 200 or body.get("version") != expected["backend_version"]:
                return None
    except (OSError, ValueError):
        return None
    return ("local", port, *listeners[0])


def adopt(app, expected, log, retained=None, *, startup=90, stability=10, owned=None):
    app = Path(app)
    owned = {} if owned is None else owned
    bundles = [app] + ([retained] if retained else [])
    before = scan_readers(bundles)
    if before.state != "none":
        raise RuntimeError("launch refused: bundle readers " + before.state)
    hashes, cursor = validate(app, expected), LogCursor(log)
    launched_at = time.time()
    launch_app(app)
    deadline, stable_since, identity, evidence = time.monotonic() + startup, None, None, None
    while time.monotonic() < deadline or stable_since is not None:
        tick = time.monotonic()
        rows = process_table()
        mains = [p for p in rows.values() if p.executable == app / "Contents/MacOS/Hermes"
                 and p.start[0] + p.start[1] / 1_000_000 >= launched_at]
        if len(mains) > 1:
            raise RuntimeError("ambiguous canonical Desktop launch")
        if not mains:
            if identity is not None:
                raise RuntimeError("Desktop died during adoption")
        else:
            main = mains[0]
            if identity is not None and (main.pid, main.start) != identity:
                raise RuntimeError("Desktop process identity changed")
            identity = (main.pid, main.start)
            tree = descendants(rows, main)
            owned.update(tree)
            renderers = [p for p in tree.values() if p.pid != main.pid and p.executable is not None
                         and p.executable == app / "Contents/Frameworks/Hermes Helper (Renderer).app/Contents/MacOS/Hermes Helper (Renderer)"]
            fresh = readiness(cursor.read(), {pid: p for pid, p in tree.items() if pid != main.pid}, expected, launched_at)
            signature = (fresh, frozenset((p.pid, p.start, p.executable) for p in renderers))
            if fresh and renderers:
                if stable_since is None:
                    stable_since, evidence = tick, signature
                elif signature != evidence:
                    raise RuntimeError("backend or renderer changed during stability")
                if tick - stable_since >= stability:
                    after = scan_readers(bundles)
                    if (after.state == "unknown" or (retained and after.readers[retained])
                            or not after.readers[app].issubset(tree) or main.pid not in after.readers[app]):
                        raise RuntimeError(f"late or ambiguous bundle readers: {after.state}; {after.reason}")
                    if validate(app, expected) != hashes:
                        raise RuntimeError("installed candidate changed during adoption")
                    final = process_table()
                    if any(final.get(pid) != p for pid, p in tree.items()):
                        raise RuntimeError("process tree changed at adoption boundary")
                    cursor.read()
                    return main
            elif stable_since is not None:
                raise RuntimeError("readiness or renderer lost during stability")
        # Never sample less frequently than once per second during stability.
        elapsed = time.monotonic() - tick
        if stable_since is not None and elapsed > 1:
            raise RuntimeError("could not sustain one-second stability sampling")
        interval = 0.8 if stable_since is not None else min(0.8, deadline - time.monotonic())
        time.sleep(max(0, interval - elapsed))
    raise RuntimeError("Desktop did not provide fresh adoption evidence within startup deadline")


@dataclass
class Outcome:
    code: int
    message: str


def require_absent(bundles):
    observed = scan_readers(bundles)
    if observed.state != "none":
        raise RuntimeError(f"bundle readers {observed.state}: {observed.reason}")


def stop_owned(owned):
    lib = native()
    for pid, original in reversed(list(owned.items())):
        current = process_record(lib, pid)
        if current is None:
            continue
        if current.start != original.start or current.executable != original.executable:
            raise RuntimeError("candidate process identity uncertain; not signaling it")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if all(process_record(lib, pid) is None for pid in owned):
            return
        time.sleep(0.5)
    raise RuntimeError("candidate processes did not exit within 30 seconds")


def recover(target, retained, old_hashes, old_expected, log, owned):
    try:
        stop_owned(owned)
        require_absent([target, retained])
        if validate(retained, old_expected) != old_hashes:
            raise RuntimeError("retained old bundle changed")
        # Final scan after signature/hashing, immediately before restoration.
        require_absent([target, retained])
        exchange(target, retained)
        if validate(target, old_expected) != old_hashes:
            raise RuntimeError("restored bundle verification failed")
    except Exception as exc:
        return Outcome(7, f"Source/backend advanced; shell recovery uncertain. Manual recovery required: {exc}. Both bundle paths retained: {target}, {retained}")
    try:
        adopt(target, old_expected, log, retained)
        return Outcome(6, f"Source/backend advanced; previous shell restored and reopened. Candidate retained at {retained}.")
    except Exception as exc:
        return Outcome(7, f"Source/backend advanced; previous shell restored, reopening unverified. Manual recovery required: {exc}. Candidate retained at {retained}.")


def publish_candidate(root, target, owner):
    target = preflight(root, target)
    expected = expectation(root)
    candidate = select_candidate(root, expected)
    if candidate.resolve().is_relative_to(target) or target.is_relative_to(candidate.resolve()):
        raise ValueError("overlapping candidate and publication target")
    old_info = plistlib.loads((target / "Contents/Info.plist").read_bytes())
    old_stamp = json.loads((target / "Contents/Resources/install-stamp.json").read_text(encoding="utf-8"))
    old_expected = {**expected, "version": old_info["CFBundleShortVersionString"], "sha": old_stamp["commit"]}
    old_hashes = validate(target, old_expected)
    source_hashes = validate(candidate, expected)
    require_absent([target])
    directory = Path(tempfile.mkdtemp(prefix=".hermes-exchange-", dir=target.parent))
    retained = directory / "Hermes.app"
    # Never sweep prior staging/rollback paths. Even partial copies remain diagnosable.
    command("/usr/bin/ditto", candidate, retained, timeout=180)
    if validate(retained, expected) != source_hashes or validate(candidate, expected) != source_hashes:
        raise RuntimeError("candidate copy/hash mismatch; old shell preserved")
    if validate(target, old_expected) != old_hashes:
        raise RuntimeError("old shell changed while staging")
    require_absent([target, retained])
    old_inode = target.stat().st_ino
    candidate_inode = retained.stat().st_ino
    owned = {}
    log = Path(root).parent / "logs/desktop.log"
    try:
        exchange(target, retained)
        if validate(target, expected) != source_hashes or fingerprint(retained) != old_hashes:
            raise RuntimeError("post-exchange bundle identity mismatch")
        require_absent([target, retained])
        clear_marker(Path(root).parent / ".hermes-update-in-progress", owner)
        adopt(target, expected, log, retained, owned=owned)
        return Outcome(0, f"Updated Desktop adopted. Previous shell retained at {retained}.")
    except Exception as exc:
        # A signal can arrive after the native swap but before its wrapper returns.
        # Determine orientation from directory identity, never from a success flag.
        try:
            if target.stat().st_ino == old_inode and retained.stat().st_ino == candidate_inode:
                return Outcome(5, f"Source/backend advanced; previous shell preserved. Exchange refused: {exc}")
            if target.stat().st_ino != candidate_inode or retained.stat().st_ino != old_inode:
                raise RuntimeError("bundle directory identities changed")
        except Exception as identity_error:
            return Outcome(7, f"Source/backend advanced; publication state uncertain. Manual recovery required: {identity_error}. Both paths retained: {target}, {retained}")
        restored = recover(target, retained, old_hashes, old_expected, log, owned)
        restored.message = f"Candidate adoption failed ({exc}). {restored.message}"
        return restored


def complete(root, target, branch, owner, update_code, message):
    root, target = Path(root), Path(target)
    outcome = Outcome(update_code or 5, message)
    try:
        if update_code:
            outcome.message = f"Underlying update failed ({update_code}); shell preserved; source/backend may have advanced. {message}"
        else:
            outcome = publish_candidate(root, target, owner)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        outcome = Outcome(5, f"Source/backend advanced; previous shell preserved. Publication refused: {exc}")
    try:
        clear_marker(root.parent / ".hermes-update-in-progress", owner)
        result = dict(ok=outcome.code == 0, exit_code=outcome.code, manual=outcome.code != 0,
                      message=outcome.message, branch=branch, finished_at=int(time.time()))
        path = root.parent / ".hermes-update-result.json"
        # Existing result protocol; a failed write cannot become a successful event.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root.parent,
                                         prefix=".hermes-update-result-", delete=False) as stream:
            json.dump(result, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stream.name, path)
    except (OSError, IndexError) as exc:
        return Outcome(9, f"{outcome.message} Terminal result could not be recorded ({type(exc).__name__}); manual follow-up required.")
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "complete"])
    parser.add_argument("root", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--owner", type=int, default=0)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--update-code", type=int, default=0)
    parser.add_argument("--message", default="")
    args = parser.parse_args()
    if sys.platform != "darwin":
        return 5

    def interrupted(signum, frame):
        raise RuntimeError(f"controlled updater interruption ({signum})")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):  # windows-footgun: ok — Darwin-only entrypoint
        signal.signal(signum, interrupted)
    if args.action == "preflight":
        try:
            preflight(args.root, args.target)
            return 0
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Publication refused before update: {exc}")
            return 5
    outcome = complete(args.root, args.target, args.branch, args.owner, args.update_code, args.message)
    print(outcome.message)
    return outcome.code


if __name__ == "__main__":
    sys.exit(main())

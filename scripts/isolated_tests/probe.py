"""Canary probe source executed INSIDE the container by the human proof.

Kept as a string so the host never executes it; its sha256 is part of the
attestation binding. It prints one JSON line of check results.
"""

from __future__ import annotations

import hashlib

from isolated_tests.policy import (
    CONTAINER_UID,
    MOUNT_INVENTORY_EXCLUDED_POINTS,
    MOUNT_INVENTORY_EXCLUDED_TREES,
    SRC_MOUNT,
)

# Mount inventory (B3). Its own constant so the regression tests exec exactly
# this code against fixture mountinfo text; PROBE_SOURCE embeds it verbatim.
# mountinfo_rw() parses /proc/self/mountinfo per proc(5): field 5 is the mount
# point (octal-escaped), field 6 the per-mount options, then zero or more
# optional fields up to a lone "-", then fstype, source and super options. Any
# line that does not fit that shape, or an empty inventory, returns None and the
# verdict fails closed. mount_inventory_verdict(parsed, required, allowed) passes
# only when every REQUIRED point (the TMPFS mounts) is rw and the rw set (after
# the policy exclusions) is a subset of required | allowed. ALLOWED (Docker's
# residual mounts) is an upper bound: a residual mount may be rw, ro or absent,
# since ro/absent is stricter than rw (Docker Desktop mounts /etc/hosts,
# /etc/hostname and /etc/resolv.conf ro under --read-only). Any other rw mount
# fails. Missing required and unexpected rw points are named in the record.
# dev_create_denied() is the /dev proof: as the
# container uid, creating a new file in /dev must fail with EACCES/EPERM/EROFS.
MOUNT_INVENTORY_HELPER = (
    "_MI_EXCLUDED_TREES = %r\n_MI_EXCLUDED_POINTS = %r\n_MI_PROBE_UID = %d\n"
    % (tuple(MOUNT_INVENTORY_EXCLUDED_TREES), tuple(MOUNT_INVENTORY_EXCLUDED_POINTS),
       CONTAINER_UID)
) + r'''
import errno as _mi_errno
import re as _mi_re
def _mi_unescape(s):
    return _mi_re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), s)
def _mi_excluded(mp):
    return mp in _MI_EXCLUDED_POINTS or any(
        mp == t or mp.startswith(t + "/") for t in _MI_EXCLUDED_TREES)
def _mi_fields(line):
    f = line.split()
    if len(f) < 10 or "-" not in f[6:]:
        return None
    sep = f.index("-", 6)
    if len(f) - sep != 4 or not (f[0].isdigit() and f[1].isdigit()):
        return None
    mm = f[2].split(":")
    if len(mm) != 2 or not (mm[0].isdigit() and mm[1].isdigit()):
        return None
    mp, opts = _mi_unescape(f[4]), f[5].split(",")
    if not mp.startswith("/") or ("rw" in opts) == ("ro" in opts):
        return None
    return mp, "rw" in opts
def mountinfo_rw(lines):
    rw, excluded, seen = set(), set(), 0
    for line in lines:
        got = _mi_fields(line)
        if got is None:
            return None
        seen += 1
        mp, is_rw = got
        if _mi_excluded(mp):
            excluded.add(mp)
        elif is_rw:
            rw.add(mp)
    if not seen:
        return None
    return rw, excluded
def mount_inventory_verdict(parsed, required, allowed):
    required, allowed = set(required), set(allowed)
    if parsed is None or not required:
        return False, {"parse_ok": parsed is not None, "missing": sorted(required),
                       "unexpected": [], "excluded": [], "rw_mounts": []}
    rw, excluded = parsed
    missing, unexpected = sorted(required - rw), sorted(rw - required - allowed)
    return not missing and not unexpected, {
        "parse_ok": True, "missing": missing, "unexpected": unexpected,
        "excluded": sorted(excluded), "rw_mounts": sorted(rw)}
def dev_create_denied(fs, path, uid):
    if uid != _MI_PROBE_UID or not path.startswith("/dev/") or path.count("/") != 2:
        return False
    flags = fs.O_WRONLY | fs.O_CREAT | fs.O_EXCL | getattr(fs, "O_NOFOLLOW", 0)
    try:
        fd = fs.open(path, flags, 0o600)
    except OSError as e:
        return e.errno in (_mi_errno.EACCES, _mi_errno.EPERM, _mi_errno.EROFS)
    try:
        fs.close(fd)
    finally:
        try: fs.unlink(path)
        except OSError: pass
    return False
'''

# Per-mount tmpfs size proof (B1). Kept as its own constant so the regression
# tests exec exactly this code against a fake filesystem; PROBE_SOURCE embeds
# it verbatim, so there is one implementation. Only ENOSPC from write() counts
# as the mount's size limit: EFBIG (RLIMIT_FSIZE), EACCES, EMFILE, any other
# error, a zero-byte write, a close error, reaching the byte budget, or
# unverified cleanup of the files this helper created is a failure. A short
# write (0 < k < requested) is how Linux fills the last free bytes of a tmpfs
# that is not chunk-aligned (e.g. after the 1-byte canary): its bytes are
# counted but it is never a pass by itself; the very next write, in this file
# or the next one, must fail with ENOSPC, and any other outcome is a failure.
TMPFS_FILL_HELPER = r'''
import errno as _fill_errno
_FILL_CHUNK = 1048576
_FILL_SLACK_CHUNKS = 2
def _fill_one(fs, fd, buf, per_file, budget, written, pending):
    # pending: the previous write (possibly in the previous file, when it hit
    # that file's last chunk slot) was short, so this write must be ENOSPC.
    n = 0
    while n + len(buf) <= per_file and written < budget:
        m = min(len(buf), budget - written)
        try:
            k = fs.write(fd, buf if m == len(buf) else buf[:m])
        except OSError as e:
            if e.errno == _fill_errno.ENOSPC:
                return True, written, pending
            raise
        if k == 0:
            raise OSError(_fill_errno.EIO, "zero-byte write")
        if pending:
            raise OSError(_fill_errno.EIO, "write succeeded after a short write")
        n += k; written += k
        pending = k < m
    return False, written, pending
def _fill_cleanup(fs, d, files, made):
    ok = True
    for p in reversed(files):
        try: fs.unlink(p)
        except OSError: ok = False
    if made:
        try: fs.rmdir(d)
        except OSError: ok = False
    left = [p for p in files if fs.path.lexists(p)]
    if made and fs.path.lexists(d):
        left.append(d)
    return ok and not left
def tmpfs_fill(fs, mount, capacity, file_cap, tag):
    if (not mount.startswith("/") or "/" in tag or tag in ("", ".", "..")
            or capacity <= 0 or file_cap <= _FILL_CHUNK):
        return False
    per_file = (file_cap - 1) // _FILL_CHUNK * _FILL_CHUNK
    budget = capacity + _FILL_SLACK_CHUNKS * _FILL_CHUNK
    d = mount.rstrip("/") + "/" + tag
    flags = fs.O_WRONLY | fs.O_CREAT | fs.O_EXCL | getattr(fs, "O_NOFOLLOW", 0)
    buf = bytes(_FILL_CHUNK)
    files = []; made = False; full = False; cleaned = False
    try:
        fs.mkdir(d, 0o700); made = True
        written = 0; pending = False
        while not full and written < budget:
            p = d + "/f%d" % len(files)
            fd = fs.open(p, flags, 0o600); files.append(p)
            try:
                full, written, pending = _fill_one(
                    fs, fd, buf, per_file, budget, written, pending)
            finally:
                fs.close(fd)
    except OSError:
        full = False
    finally:
        cleaned = _fill_cleanup(fs, d, files, made)
    return full and cleaned
def tmpfs_fill_all(fs, tm, lim, tag):
    res = {}
    for path, key in sorted(tm.items()):
        res[path] = tmpfs_fill(fs, path, lim[key], lim["fsize_bytes"], tag)
    return res, bool(res) and all(res.values())
'''

# Env and network verdicts, pure so the regression tests exec exactly this code
# against fixture values; PROBE_SOURCE embeds it verbatim.
#
# env_exact_verdict: the non-ISOTEST_ env keys must EQUAL the expected set, with
# one exception: PWD, which /bin/sh -c (dash) exports from the container
# workdir. PWD is accepted only when its value is exactly SRC_MOUNT, the
# --workdir docker_argv.run_argv passes (both read policy.SRC_MOUNT). Any other
# extra or missing key, or PWD with another value, fails.
#
# net_denied_verdict: Docker Desktop's LinuxKit kernel creates fallback tunnel
# devices (gre0, sit0, tunl0, ...) in every network namespace, so --network=none
# shows more than "lo" in /sys/class/net. The container runs with
# --cap-drop=ALL, so it cannot bring an interface up or add a route
# (CAP_NET_ADMIN). The verdict passes only when ALL hold: connect() to a public
# address was refused; "lo" is present; every interface other than lo has
# IFF_UP (0x1) clear in its flags; no flags read failed (None); and neither
# /proc/net/route (after its header; None = unreadable) nor
# /proc/net/ipv6_route (no header; None = unreadable, a missing file is passed
# in as []) has a row for an interface other than lo.
ENV_NET_HELPER = "_EN_PWD = %r\n" % SRC_MOUNT + r'''
_IFF_UP = 0x1
def env_exact_verdict(env, want):
    want = set(want)
    have = {k for k in env if not k.startswith("ISOTEST_")}
    if "PWD" in have and "PWD" not in want:
        if env["PWD"] != _EN_PWD:
            return False
        have.discard("PWD")
    return have == want
def _en_route_ifaces(lines, iface_col):
    out = set()
    for line in lines:
        f = line.split()
        if f:
            out.add(f[iface_col])
    return out
def net_denied_verdict(connect_denied, flags, route, route6):
    if connect_denied is not True or "lo" not in flags:
        return False
    if any(v is None for v in flags.values()):
        return False
    if any(v & _IFF_UP for n, v in flags.items() if n != "lo"):
        return False
    if route is None or route6 is None:
        return False
    ifaces = _en_route_ifaces(route[1:], 0) | _en_route_ifaces(route6, -1)
    return ifaces <= {"lo"}
'''

_PROBE_HEAD = r'''
import json, os, resource, socket
r = {}
def denied(fn):
    try:
        fn(); return False
    except OSError:
        return True
def w(p):
    with open(p, "w") as f: f.write("x")
r["scratch_write_allowed"] = not denied(lambda: w("/scratch/tmp/canary"))
r["source_write_denied"] = denied(lambda: w("/src/.canary"))
r["rootfs_write_denied"] = denied(lambda: w("/etc/.canary"))
host = os.environ.get("ISOTEST_HOST_CANARY", "")
r["host_canary_absent"] = bool(host) and not os.path.exists(host)
r["host_home_absent"] = not any(os.path.exists(p) for p in ("/Users", "/home/" + os.environ.get("ISOTEST_HOST_USER", "x")))
r["docker_socket_absent"] = not any(os.path.exists(p) for p in ("/var/run/docker.sock", "/run/docker.sock"))
def net():
    s = socket.socket(); s.settimeout(3); s.connect(("1.1.1.1", 443))
def net_flags():
    out = {}
    try: names = sorted(os.listdir("/sys/class/net"))
    except OSError: return out
    for n in names:
        p = "/sys/class/net/" + n
        if not os.path.isdir(p): continue  # e.g. the bonding_masters control file
        try:
            with open(p + "/flags") as f: out[n] = int(f.read().strip(), 16)
        except (OSError, ValueError): out[n] = None
    return out
def lines_of(p, missing):
    try:
        with open(p) as f: return f.read().splitlines()
    except FileNotFoundError: return missing
    except OSError: return None
net_connect_denied = denied(net)
net_if = net_flags()
net_route = lines_of("/proc/net/route", None)
net_route6 = lines_of("/proc/net/ipv6_route", [])
r["network_denied"] = net_denied_verdict(net_connect_denied, net_if, net_route, net_route6)
print("ISOTEST_RECORD " + json.dumps({"net": {
    "connect_denied": net_connect_denied,
    "if_flags": {n: (None if v is None else "0x%x" % v) for n, v in net_if.items()},
    "route_rows": None if net_route is None else net_route[1:],
    "route6_rows": net_route6}}, sort_keys=True))
r["nonroot"] = os.getuid() != 0
st = open("/proc/self/status").read()
r["no_caps"] = "CapEff:\t0000000000000000" in st
r["no_new_privs"] = "NoNewPrivs:\t1" in st
def cg(name):
    try: return open("/sys/fs/cgroup/" + name).read().split()
    except OSError: return None
lim = json.loads(os.environ["ISOTEST_LIMITS"])
r["memory_limit"] = cg("memory.max") == [str(lim["memory_bytes"])]
r["swap_limit"] = cg("memory.swap.max") == [str(lim["memory_swap_bytes"] - lim["memory_bytes"])]
r["pids_limit"] = cg("pids.max") == [str(lim["pids"])]
r["cgroup_v2"] = os.path.exists("/sys/fs/cgroup/cgroup.controllers")
r["cpu_limit"] = (cg("cpu.max") or [""])[0] == str(int(float(lim["cpus"]) * int((cg("cpu.max") or ["0", "100000"])[1])))
r["nofile_limit"] = resource.getrlimit(resource.RLIMIT_NOFILE) == (lim["nofile"], lim["nofile"])
r["fsize_limit"] = resource.getrlimit(resource.RLIMIT_FSIZE) == (lim["fsize_bytes"], lim["fsize_bytes"])
tm = json.loads(os.environ["ISOTEST_TMPFS"])
tot = 0
for path, key in tm.items():
    sv = os.statvfs(path); tot += sv.f_blocks * sv.f_frsize
r["tmpfs_aggregate_bounded"] = tot <= lim["scratch_tmpfs_bytes"] + lim["var_tmp_tmpfs_bytes"] + lim["tmp_tmpfs_bytes"]
residual = json.loads(os.environ["ISOTEST_RESIDUAL"])
with open("/proc/self/mountinfo") as mi:
    parsed = mountinfo_rw(mi.read().splitlines())
r["mount_inventory_expected"], inv = mount_inventory_verdict(parsed, set(tm), set(residual))
rw = set(inv["rw_mounts"])
rec = {pth: pth in rw for pth in residual}
print("ISOTEST_RECORD " + json.dumps({"residual_rw": rec, "rw_mounts": sorted(rw),
                                      "mount_inventory": inv}, sort_keys=True))
r["dev_write_denied"] = dev_create_denied(os, "/dev/.isotest-dev-%d" % os.getpid(), os.getuid())
r["seccomp_filtered"] = "Seccomp:\t2" in st
r["env_exact"] = env_exact_verdict(os.environ, json.loads(os.environ["ISOTEST_EXPECT_ENV"]))
socks = []
for root in ("/run", "/var/run", "/tmp", "/var/tmp", "/scratch", "/dev"):
    for dp, dn, fn in os.walk(root):
        if dp.count("/") > 4: dn[:] = []
        for n in fn:
            try:
                import stat as _s
                if _s.S_ISSOCK(os.lstat(os.path.join(dp, n)).st_mode): socks.append(os.path.join(dp, n))
            except OSError: pass
r["no_unix_sockets"] = not socks
def loop():
    a = socket.socket(); a.bind(("127.0.0.1", 0)); a.listen(1)
    b = socket.socket(); b.settimeout(3); b.connect(a.getsockname()); b.close(); a.close()
r["loopback_allowed"] = not denied(loop)
import hashlib as _h
known = json.loads(os.environ["ISOTEST_SRC_HASHES"])
r["source_hash_known"] = bool(known) and all(_h.sha256(open("/src/" + k, "rb").read()).hexdigest() == v for k, v in known.items())
def big():
    with open("/scratch/tmp/fsize", "wb") as f:
        f.write(b"\0" * (lim["fsize_bytes"] + 4096))
import signal as _sg
_sg.signal(_sg.SIGXFSZ, _sg.SIG_IGN)
r["fsize_enforced"] = denied(big)
try: os.unlink("/scratch/tmp/fsize")
except OSError: pass
'''

_PROBE_TAIL = r'''
tag = ".isotest-fill-%d" % os.getpid()
fills, r["tmpfs_full_enforced"] = tmpfs_fill_all(os, tm, lim, tag)
print("ISOTEST_RECORD " + json.dumps({"tmpfs_fill": fills}, sort_keys=True))
r["tmpfs_cleaned"] = (not os.path.exists("/scratch/tmp/fsize")
                      and not any(os.path.lexists(p.rstrip("/") + "/" + tag) for p in tm))
print("ISOTEST_PROBE " + json.dumps(r, sort_keys=True))
'''

PROBE_SOURCE = MOUNT_INVENTORY_HELPER + ENV_NET_HELPER + _PROBE_HEAD + TMPFS_FILL_HELPER + _PROBE_TAIL

PROBE_SHA256 = hashlib.sha256(PROBE_SOURCE.encode()).hexdigest()


def parse_probe(stdout: bytes) -> dict[str, bool]:
    """Return the probe's checks; an absent or malformed line is a failure."""
    import json

    for line in stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("ISOTEST_PROBE "):
            doc = json.loads(line[len("ISOTEST_PROBE "):])
            if isinstance(doc, dict) and all(isinstance(v, bool) for v in doc.values()):
                return doc
    return {}


REQUIRED_CHECKS = (
    "scratch_write_allowed", "source_write_denied", "rootfs_write_denied",
    "host_canary_absent", "host_home_absent", "docker_socket_absent",
    "network_denied", "nonroot", "no_caps", "no_new_privs", "memory_limit",
    "swap_limit", "pids_limit", "cpu_limit", "nofile_limit", "fsize_limit",
    "cgroup_v2", "tmpfs_aggregate_bounded", "mount_inventory_expected", "seccomp_filtered",
    "env_exact", "no_unix_sockets", "loopback_allowed", "source_hash_known",
    "fsize_enforced", "tmpfs_full_enforced", "tmpfs_cleaned", "dev_write_denied",
)

# The wall-clock kill is proven by a second, separate canary container with a
# short wall whose cleanup status must be "removed" and whose result is timed_out.
WALL_CANARY_SECONDS = 5
WALL_CANARY_SCRIPT = "exec sleep 600"


def output_digest(stdout: bytes) -> str:
    """sha256 of the probe's ISOTEST_* lines, bound into the attestation."""
    lines = [ln for ln in stdout.decode("utf-8", "replace").splitlines() if ln.startswith("ISOTEST_")]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def all_passed(results: dict[str, bool]) -> bool:
    return all(results.get(k) is True for k in REQUIRED_CHECKS)

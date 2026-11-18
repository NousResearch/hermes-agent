"""Mount inventory, env and network verdicts of the canary probe.

Mount (B3, P8): every TMPFS mount must be rw; Docker's residual mounts may be rw,
ro or absent; any other rw mount fails. Env (P8): the non-ISOTEST_ keys must equal
the expected set, plus PWD only when it is exactly SRC_MOUNT. Network (P8):
connect refused, lo present, every other interface down (IFF_UP clear), no
non-lo route. The helpers under test are the same code the in-container probe
runs (``probe.MOUNT_INVENTORY_HELPER`` and ``probe.ENV_NET_HELPER`` are embedded
verbatim in ``PROBE_SOURCE``); here they run against fixture text, including the
Docker Desktop diagnostic capture that the R7 probe refused. Not run yet.
"""

from __future__ import annotations

import ast
import errno

import pytest

from isolated_tests import docker_argv, policy, probe

REQUIRED = set(policy.TMPFS)
ALLOWED = set(policy.DOCKER_RESIDUAL_RW)
EXPECTED = REQUIRED | ALLOWED


@pytest.fixture(scope="module")
def ns():
    space: dict = {}
    exec(probe.MOUNT_INVENTORY_HELPER, space)  # the exact code PROBE_SOURCE runs in the container
    return space


def _line(n, mp, opts, *, optional=("shared:1",), fstype="tmpfs", source="tmpfs", sup="rw"):
    return " ".join([str(n), "1", "0:%d" % n, "/", mp, opts, *optional, "-", fstype, source, sup])


def _mountinfo(points, *, extra=(), ro_points=()):
    lines = [_line(1, "/", "ro,relatime", fstype="overlay", source="overlay")]
    lines += [_line(i + 2, mp, "rw,nosuid,nodev") for i, mp in enumerate(sorted(points))]
    lines += [_line(i + 30, mp, "ro,relatime") for i, mp in enumerate(sorted(ro_points))]
    lines += [
        _line(50, "/proc", "rw,nosuid", fstype="proc", source="proc"),
        _line(51, "/proc/sys", "ro,nosuid", fstype="proc", source="proc"),
        _line(52, "/sys", "ro,nosuid", fstype="sysfs", source="sysfs"),
        _line(53, "/sys/fs/cgroup", "ro,nosuid", fstype="cgroup2", source="cgroup"),
        _line(54, "/dev", "rw,nosuid", optional=(), source="tmpfs"),
        _line(55, "/dev/pts", "rw,nosuid", fstype="devpts", source="devpts"),
        _line(56, "/src", "ro,relatime", optional=("master:3", "shared:7"), fstype="virtiofs",
              source="mount0"),
    ]
    return lines + list(extra)


def _verdict(ns, lines, required=REQUIRED, allowed=ALLOWED):
    return ns["mount_inventory_verdict"](ns["mountinfo_rw"](lines), required, allowed)


def test_exact_match_passes_and_records_exclusions(ns):
    ok, rec = _verdict(ns, _mountinfo(EXPECTED))
    assert ok is True
    assert rec["parse_ok"] is True and rec["missing"] == [] and rec["unexpected"] == []
    assert set(rec["rw_mounts"]) == EXPECTED
    assert set(rec["excluded"]) == {"/proc", "/proc/sys", "/sys", "/sys/fs/cgroup", "/dev",
                                    "/dev/pts"}


@pytest.mark.parametrize("tmpfs", sorted(policy.TMPFS))
def test_missing_tmpfs_mount_fails(ns, tmpfs):
    ok, rec = _verdict(ns, _mountinfo(EXPECTED - {tmpfs}))
    assert ok is False
    assert rec["missing"] == [tmpfs] and rec["unexpected"] == []


@pytest.mark.parametrize("tmpfs", sorted(policy.TMPFS))
def test_read_only_tmpfs_mount_fails(ns, tmpfs):
    ok, rec = _verdict(ns, _mountinfo(EXPECTED - {tmpfs}, ro_points={tmpfs}))
    assert ok is False
    assert rec["missing"] == [tmpfs]


def test_residual_mounts_read_only_or_absent_pass(ns):
    ro = {"/etc/hosts", "/etc/hostname", "/etc/resolv.conf"}
    ok, rec = _verdict(ns, _mountinfo(EXPECTED - ro, ro_points=ro))
    assert ok is True and rec["missing"] == [] and rec["unexpected"] == []
    assert set(rec["rw_mounts"]) == EXPECTED - ro
    assert _verdict(ns, _mountinfo(REQUIRED))[0] is True


def test_other_rw_mount_fails_even_with_residual_read_only(ns):
    ro = {"/etc/hosts", "/etc/hostname", "/etc/resolv.conf"}
    ok, rec = _verdict(ns, _mountinfo(EXPECTED - ro | {"/data"}, ro_points=ro))
    assert ok is False and rec["unexpected"] == ["/data"]


def test_unexpected_rw_mount_fails(ns):
    ok, rec = _verdict(ns, _mountinfo(EXPECTED | {"/srv/leak"}))
    assert ok is False
    assert rec["unexpected"] == ["/srv/leak"] and rec["missing"] == []


def test_exclusion_is_exact_point_or_subtree_only(ns):
    # /devices is not /dev, /system is not /sys: lookalikes stay in the inventory.
    ok, rec = _verdict(ns, _mountinfo(EXPECTED | {"/devices", "/system"}))
    assert ok is False
    assert rec["unexpected"] == ["/devices", "/system"]
    assert "/devices" not in rec["excluded"] and "/system" not in rec["excluded"]


def test_escaped_mount_point_is_decoded(ns):
    extra = [_line(60, "/srv/a\\040b", "rw")]
    ok, rec = _verdict(ns, _mountinfo(EXPECTED, extra=extra))
    assert ok is False and rec["unexpected"] == ["/srv/a b"]


@pytest.mark.parametrize("bad", [
    "",                                                        # blank line
    "1 1 0:1 / /x rw shared:1 tmpfs tmpfs rw",                 # no separator
    "1 1 0:1 / /x rw - tmpfs tmpfs",                           # too few fields after "-"
    "1 1 0:1 / /x rw - tmpfs tmpfs rw extra",                  # too many fields after "-"
    "x 1 0:1 / /x rw - tmpfs tmpfs rw",                        # non-numeric mount id
    "1 1 01 / /x rw - tmpfs tmpfs rw",                         # bad major:minor
    "1 1 0:1 / relative rw - tmpfs tmpfs rw",                  # mount point not absolute
    "1 1 0:1 / /x nosuid - tmpfs tmpfs rw",                    # neither rw nor ro
    "1 1 0:1 / /x rw,ro - tmpfs tmpfs rw",                     # both rw and ro
])
def test_malformed_line_fails_closed(ns, bad):
    assert ns["mountinfo_rw"](_mountinfo(EXPECTED, extra=[bad])) is None
    ok, rec = _verdict(ns, _mountinfo(EXPECTED, extra=[bad]))
    assert ok is False and rec["parse_ok"] is False
    assert set(rec["missing"]) == REQUIRED


def test_empty_inventory_or_requirement_fails_closed(ns):
    assert ns["mountinfo_rw"]([]) is None
    assert _verdict(ns, [])[0] is False
    assert _verdict(ns, _mountinfo(set()), required=set(), allowed=set())[0] is False
    assert _verdict(ns, _mountinfo(ALLOWED), required=set())[0] is False


# /proc/self/mountinfo captured in a Docker Desktop (LinuxKit) canary container
# started with the runner's run_argv flags; the R7 probe refused it because
# /etc/hosts, /etc/hostname and /etc/resolv.conf are ro. The host user path of
# the /src bind source is replaced; nothing else is edited.
DOCKER_DESKTOP_MOUNTINFO = (
    "1107 978 0:261 / / ro,relatime - overlay overlay rw,lowerdir=/var/lib/desktop-containerd/daemon/io.containerd.snapshotter.v1.overlayfs/snapshots/4051/fs:/var/lib/desktop-containerd/daemon/io.containerd.snapshotter.v1.overlayfs/snapshots/4046/fs:/var/lib/desktop-containerd/daemon/io.containerd.snapshotter.v1.overlayfs/snapshots/4045/fs:/var/lib/desktop-containerd/daemon/io.containerd.snapshotter.v1.overlayfs/snapshots/4044/fs,upperdir=/var/lib/desktop-containerd/daemon/io.containerd.snapshotter.v1.overlayfs/snapshots/4052/fs,workdir=/var/lib/desktop-containerd/daemon/io.containerd.snapshotter.v1.overlayfs/snapshots/4052/work,nouserxattr",
    "1109 1107 0:308 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw",
    "1110 1107 0:309 / /dev rw,nosuid - tmpfs tmpfs rw,size=65536k,mode=755",
    "1111 1110 0:310 / /dev/pts rw,nosuid,noexec,relatime - devpts devpts rw,gid=5,mode=620,ptmxmode=666",
    "1112 1107 0:311 / /sys ro,nosuid,nodev,noexec,relatime - sysfs sysfs ro",
    "1113 1112 0:41 / /sys/fs/cgroup ro,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw",
    "1114 1110 0:306 / /dev/mqueue rw,nosuid,nodev,noexec,relatime - mqueue mqueue rw",
    "1115 1110 0:312 / /dev/shm rw,nosuid,nodev,noexec,relatime - tmpfs shm rw,size=65536k",
    "1116 1107 0:26 /usr/libexec/docker/docker-init /usr/sbin/docker-init ro,relatime - overlay overlay rw,lowerdir=/,upperdir=/run/rootfs.upper,workdir=/run/rootfs.workdir,uuid=on,nouserxattr",
    "1117 1107 0:313 / /scratch rw,nosuid,nodev,noexec,relatime - tmpfs tmpfs rw,size=262144k",
    "1118 1107 0:46 /user/isotest/snapshot-ae2a81ccc00e-323b755a /src ro,nosuid,nodev,relatime - fakeowner /run/host_mark/Users rw,fakeowner",
    "1119 1107 0:314 / /tmp rw,nosuid,nodev,noexec,relatime - tmpfs tmpfs rw,size=65536k",
    "1120 1107 254:1 /docker/containers/9c7cef58ac947f0dac79cfcc08a982ac88ae6966c18a49828389f982c96bca63/hostname /etc/hostname ro,relatime - ext4 /dev/vda1 rw,discard,noauto_da_alloc",
    "1121 1107 254:1 /docker/containers/9c7cef58ac947f0dac79cfcc08a982ac88ae6966c18a49828389f982c96bca63/hosts /etc/hosts ro,relatime - ext4 /dev/vda1 rw,discard,noauto_da_alloc",
    "1122 1107 254:1 /docker/containers/9c7cef58ac947f0dac79cfcc08a982ac88ae6966c18a49828389f982c96bca63/resolv.conf /etc/resolv.conf ro,relatime - ext4 /dev/vda1 rw,discard,noauto_da_alloc",
    "1123 1107 0:315 / /var/tmp rw,nosuid,nodev,noexec,relatime - tmpfs tmpfs rw,size=196608k",
    "979 1109 0:308 /bus /proc/bus ro,nosuid,nodev,noexec,relatime - proc proc rw",
    "980 1109 0:308 /fs /proc/fs ro,nosuid,nodev,noexec,relatime - proc proc rw",
    "981 1109 0:308 /irq /proc/irq ro,nosuid,nodev,noexec,relatime - proc proc rw",
    "988 1109 0:308 /sys /proc/sys ro,nosuid,nodev,noexec,relatime - proc proc rw",
    "989 1109 0:308 /sysrq-trigger /proc/sysrq-trigger ro,nosuid,nodev,noexec,relatime - proc proc rw",
    "990 1109 0:309 /null /proc/interrupts rw,nosuid - tmpfs tmpfs rw,size=65536k,mode=755",
    "991 1109 0:309 /null /proc/kcore rw,nosuid - tmpfs tmpfs rw,size=65536k,mode=755",
    "1003 1109 0:309 /null /proc/keys rw,nosuid - tmpfs tmpfs rw,size=65536k,mode=755",
    "1004 1109 0:316 / /proc/scsi ro,relatime - tmpfs tmpfs ro,size=4k,nr_inodes=1",
    "1005 1109 0:309 /null /proc/timer_list rw,nosuid - tmpfs tmpfs rw,size=65536k,mode=755",
    "1006 1112 0:316 / /sys/firmware ro,relatime - tmpfs tmpfs ro,size=4k,nr_inodes=1",
)


def test_docker_desktop_capture_passes(ns):
    ok, rec = _verdict(ns, list(DOCKER_DESKTOP_MOUNTINFO))
    assert ok is True and rec["parse_ok"] is True
    assert rec["missing"] == [] and rec["unexpected"] == []
    assert set(rec["rw_mounts"]) == REQUIRED | {"/dev/shm", "/dev/mqueue"}


def test_docker_desktop_capture_with_extra_rw_mount_fails(ns):
    extra = "1200 1107 0:400 / /data rw,relatime - tmpfs tmpfs rw"
    ok, rec = _verdict(ns, [*DOCKER_DESKTOP_MOUNTINFO, extra])
    assert ok is False and rec["unexpected"] == ["/data"]


def test_docker_desktop_capture_with_tmpfs_read_only_fails(ns):
    lines = [ln.replace(" /var/tmp rw,", " /var/tmp ro,") for ln in DOCKER_DESKTOP_MOUNTINFO]
    ok, rec = _verdict(ns, lines)
    assert ok is False and rec["missing"] == ["/var/tmp"]


@pytest.fixture(scope="module")
def en():
    space: dict = {}
    exec(probe.ENV_NET_HELPER, space)  # the exact code PROBE_SOURCE runs in the container
    return space


WANT_ENV = sorted(e.split("=", 1)[0] for e in docker_argv.container_env("/usr/bin/python3"))


def _env(**extra):
    env = {k: "v" for k in WANT_ENV}
    env.update({"ISOTEST_LIMITS": "{}", "ISOTEST_EXPECT_ENV": "[]"})
    env.update(extra)
    return env


def test_env_pwd_equal_to_src_mount_passes(en):
    assert en["env_exact_verdict"](_env(PWD=policy.SRC_MOUNT), WANT_ENV) is True
    assert en["env_exact_verdict"](_env(), WANT_ENV) is True


@pytest.mark.parametrize("pwd", ["/other", "/src/", "/", ""])
def test_env_pwd_other_value_fails(en, pwd):
    assert en["env_exact_verdict"](_env(PWD=pwd), WANT_ENV) is False


def test_env_unknown_extra_or_missing_key_fails(en):
    assert en["env_exact_verdict"](_env(PWD=policy.SRC_MOUNT, LD_PRELOAD="x"), WANT_ENV) is False
    env = _env()
    del env["HOME"]
    assert en["env_exact_verdict"](env, WANT_ENV) is False


# Docker Desktop --network=none capture: LinuxKit's fallback tunnel devices are
# present and down, lo is up, bonding_masters (a file) is not an interface.
DESKTOP_IF_FLAGS = {"erspan0": 0x1002, "gre0": 0x80, "gretap0": 0x1002, "ip6_vti0": 0x80,
                    "ip6gre0": 0x80, "ip6tnl0": 0x80, "ip_vti0": 0x80, "lo": 0x9,
                    "sit0": 0x80, "tunl0": 0x80}
ROUTE_HEADER = "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT"
LO_ROUTE6 = ("00000000000000000000000000000001 80 00000000000000000000000000000000 00 "
             "00000000000000000000000000000000 00000000 00000001 00000000 00000001       lo")


def _net(en, *, denied=True, flags=None, route=None, route6=None):
    return en["net_denied_verdict"](
        denied, DESKTOP_IF_FLAGS if flags is None else flags,
        [ROUTE_HEADER] if route is None else route, [LO_ROUTE6] if route6 is None else route6)


def test_net_docker_desktop_capture_passes(en):
    assert _net(en) is True
    assert _net(en, route6=[]) is True  # /proc/net/ipv6_route absent


def test_net_non_lo_interface_up_fails(en):
    assert _net(en, flags={**DESKTOP_IF_FLAGS, "gre0": 0x81}) is False
    assert _net(en, flags={**DESKTOP_IF_FLAGS, "eth0": 0x1043}) is False


def test_net_non_lo_route_fails(en):
    row = "eth0\t00000000\t010011AC\t0003\t0\t0\t0\t00000000\t0\t0\t0"
    assert _net(en, route=[ROUTE_HEADER, row]) is False
    assert _net(en, route6=[LO_ROUTE6, LO_ROUTE6.replace("      lo", "    eth0")]) is False


def test_net_lo_missing_fails(en):
    flags = {k: v for k, v in DESKTOP_IF_FLAGS.items() if k != "lo"}
    assert _net(en, flags=flags) is False
    assert _net(en, flags={}) is False


def test_net_connect_succeeding_fails(en):
    assert _net(en, denied=False) is False


def test_net_unreadable_input_fails_closed(en):
    assert _net(en, flags={**DESKTOP_IF_FLAGS, "sit0": None}) is False
    assert en["net_denied_verdict"](True, DESKTOP_IF_FLAGS, None, [LO_ROUTE6]) is False
    assert en["net_denied_verdict"](True, DESKTOP_IF_FLAGS, [ROUTE_HEADER], None) is False


class FakeDevFS:
    O_WRONLY, O_CREAT, O_EXCL, O_NOFOLLOW = 1, 2, 4, 8

    def __init__(self, open_errno=None):
        self.open_errno = open_errno
        self.files: set[str] = set()
        self.opened: list[tuple[str, int]] = []

    def open(self, p, flags, mode):
        self.opened.append((p, flags))
        if self.open_errno:
            raise OSError(self.open_errno, p)
        self.files.add(p)
        return 3

    def close(self, fd):
        pass

    def unlink(self, p):
        self.files.discard(p)


@pytest.mark.parametrize("code", [errno.EACCES, errno.EPERM, errno.EROFS])
def test_dev_create_denied_by_permission_errno(ns, code):
    fs = FakeDevFS(open_errno=code)
    assert ns["dev_create_denied"](fs, "/dev/.isotest-dev-1", policy.CONTAINER_UID) is True
    (path, flags), = fs.opened
    assert path == "/dev/.isotest-dev-1"
    assert flags & fs.O_CREAT and flags & fs.O_EXCL and flags & fs.O_NOFOLLOW


@pytest.mark.parametrize("code", [errno.EEXIST, errno.ENOSPC, errno.EIO])
def test_dev_create_other_errno_fails(ns, code):
    fs = FakeDevFS(open_errno=code)
    assert ns["dev_create_denied"](fs, "/dev/.isotest-dev-1", policy.CONTAINER_UID) is False


def test_dev_create_succeeding_fails_and_cleans_up(ns):
    fs = FakeDevFS()
    assert ns["dev_create_denied"](fs, "/dev/.isotest-dev-1", policy.CONTAINER_UID) is False
    assert fs.files == set()


def test_dev_create_wrong_uid_or_path_fails_without_trying(ns):
    fs = FakeDevFS(open_errno=errno.EACCES)
    assert ns["dev_create_denied"](fs, "/dev/.isotest-dev-1", 0) is False
    assert ns["dev_create_denied"](fs, "/dev/shm/x", policy.CONTAINER_UID) is False
    assert ns["dev_create_denied"](fs, "/tmp/x", policy.CONTAINER_UID) is False
    assert fs.opened == []


def test_probe_source_embeds_helpers_and_requires_dev_check():
    assert probe.MOUNT_INVENTORY_HELPER in probe.PROBE_SOURCE
    assert probe.TMPFS_FILL_HELPER in probe.PROBE_SOURCE
    assert probe.ENV_NET_HELPER in probe.PROBE_SOURCE
    ast.parse(probe.PROBE_SOURCE)
    assert "dev_write_denied" in probe.REQUIRED_CHECKS
    assert "mount_inventory_expected" in probe.REQUIRED_CHECKS
    results = {k: True for k in probe.REQUIRED_CHECKS}
    assert probe.all_passed(results)
    assert not probe.all_passed({**results, "dev_write_denied": False})

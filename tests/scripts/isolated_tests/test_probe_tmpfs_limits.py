"""Per-mount tmpfs size proof (B1): only ENOSPC from a bounded fill counts.

The helper under test is the same code the in-container probe runs
(``probe.TMPFS_FILL_HELPER`` is embedded verbatim in ``PROBE_SOURCE``); here it
runs against a fake filesystem that enforces RLIMIT_FSIZE (EFBIG) and a tmpfs
size (ENOSPC) the way the kernel does: a write that does not fit writes what is
left and returns that count, and only the next write fails. Verdicts come from
observed errno, bytes, per-file sizes and leftover entries, not from stubbed
return values. Against this fake, the unaligned-usage cases are expected RED on
the R1 helper (which failed on any short write) and GREEN after B1-F1; the
previous all-or-nothing fake could not show that. Neither run has been made.
"""

from __future__ import annotations

import errno

import pytest

from isolated_tests import policy, probe

MIB = 1024 * 1024
FSIZE = policy.LIMITS["fsize_bytes"]
TAG = ".isotest-fill-1"


class Boom(Exception):
    pass


class _Path:
    def __init__(self, fs):
        self._fs = fs

    def lexists(self, p):
        return p in self._fs.files or p in self._fs.dirs


class FakeFS:
    O_WRONLY, O_CREAT, O_EXCL, O_NOFOLLOW = 1, 2, 4, 8

    def __init__(self, capacities, *, write_errno=None, short_after=None, close_errno=None,
                 open_errno=None, unlink_errno=None, raise_after=None, used=None,
                 short_once=False, zero_after=None):
        self.capacities = dict(capacities)  # mount -> bytes, None = size not enforced
        self.used = {m: 0 for m in self.capacities}
        self.used.update(used or {})
        self.created = []
        self.dirs = set(self.capacities)
        self.files = {}
        self.fds = {}
        self.attempted = 0
        self.max_file = 0
        self.write_errno, self.short_after = write_errno, short_after
        self.close_errno, self.open_errno = close_errno, open_errno
        self.unlink_errno, self.raise_after = unlink_errno, raise_after
        # Injected faults, not kernel behaviour: an artificial short write while
        # capacity remains (so the next write succeeds), and a zero-byte return.
        self.short_once, self.zero_after, self._shorted = short_once, zero_after, False
        self.path = _Path(self)

    def _mount(self, p):
        return max((m for m in self.capacities if p.startswith(m.rstrip("/") + "/")), key=len)

    def mkdir(self, p, mode):
        if p in self.dirs or p in self.files:
            raise FileExistsError(errno.EEXIST, p)
        self.dirs.add(p)

    def open(self, p, flags, mode):
        assert flags & self.O_EXCL and flags & self.O_CREAT
        if self.open_errno:
            raise OSError(self.open_errno, p)
        if p in self.files:
            raise FileExistsError(errno.EEXIST, p)
        self.files[p] = 0
        self.created.append(p)
        fd = len(self.fds) + 3
        self.fds[fd] = p
        return fd

    def write(self, fd, buf):
        p = self.fds[fd]
        self.attempted += len(buf)
        if self.raise_after is not None and self.attempted > self.raise_after:
            raise Boom()
        if self.write_errno:
            raise OSError(self.write_errno, p)
        # Kernel semantics for both limits: write what still fits and return the
        # count; the errno comes only once nothing fits. The helper keeps every
        # file below RLIMIT_FSIZE, so its EFBIG partial is reached only by the
        # old single-file shape test.
        room = FSIZE - self.files[p]
        if room <= 0:
            raise OSError(errno.EFBIG, p)
        m = self._mount(p)
        cap = self.capacities[m]
        free = len(buf) if cap is None else cap - self.used[m]
        if free <= 0:
            raise OSError(errno.ENOSPC, p)
        n = min(len(buf), room, free)
        if self.zero_after is not None and self.attempted > self.zero_after:
            n = 0
        elif self.short_after is not None and self.attempted > self.short_after and not self._shorted:
            n = min(n, len(buf) // 2)
            self._shorted = self.short_once
        self.files[p] += n
        self.used[m] += n
        self.max_file = max(self.max_file, self.files[p])
        return n

    def close(self, fd):
        self.fds.pop(fd)
        if self.close_errno:
            raise OSError(self.close_errno, "close")

    def unlink(self, p):
        if self.unlink_errno:
            raise OSError(self.unlink_errno, p)
        size = self.files.pop(p)
        self.used[self._mount(p)] -= size

    def rmdir(self, d):
        if any(f.startswith(d + "/") for f in self.files):
            raise OSError(errno.ENOTEMPTY, d)
        self.dirs.remove(d)


def _helper():
    ns: dict = {}
    exec(probe.TMPFS_FILL_HELPER, ns)  # the exact code PROBE_SOURCE runs in the container
    return ns


def _policy_mounts():
    return [(path, policy.LIMITS[key]) for path, key in policy.TMPFS.items()]


def _owned_left(fs):
    return [p for p in (*fs.files, *fs.dirs) if TAG in p]


def test_probe_runs_the_same_helper_that_is_tested():
    assert probe.TMPFS_FILL_HELPER in probe.PROBE_SOURCE


@pytest.mark.parametrize("mount,cap", _policy_mounts())
def test_enospc_at_mount_size_passes_with_files_below_fsize(mount, cap):
    fs = FakeFS({mount: cap})
    assert _helper()["tmpfs_fill"](fs, mount, cap, FSIZE, TAG) is True
    assert 0 < fs.max_file < FSIZE
    assert fs.attempted <= cap + 2 * MIB
    assert _owned_left(fs) == [] and fs.used[mount] == 0


@pytest.mark.parametrize("mount,cap", _policy_mounts())
def test_unenforced_size_stops_at_budget_and_fails(mount, cap):
    fs = FakeFS({mount: None})
    assert _helper()["tmpfs_fill"](fs, mount, cap, FSIZE, TAG) is False
    assert cap <= fs.attempted <= cap + 3 * MIB
    assert 0 < fs.max_file < FSIZE
    assert _owned_left(fs) == []


@pytest.mark.parametrize("err", [errno.EFBIG, errno.EACCES, errno.EMFILE, errno.EIO])
def test_errors_other_than_enospc_fail(err):
    fs = FakeFS({"/tmp": 64 * MIB}, write_errno=err)
    assert _helper()["tmpfs_fill"](fs, "/tmp", 64 * MIB, FSIZE, TAG) is False
    assert _owned_left(fs) == []


def test_old_single_file_shape_efbig_is_not_a_tmpfs_proof():
    # file cap above RLIMIT_FSIZE: the fake kernel answers EFBIG before the tmpfs fills
    fs = FakeFS({"/tmp": None})
    assert _helper()["tmpfs_fill"](fs, "/tmp", 64 * MIB, FSIZE + 4 * MIB, TAG) is False
    assert _owned_left(fs) == []


def test_enospc_on_open_is_not_a_size_proof():
    fs = FakeFS({"/tmp": 64 * MIB}, open_errno=errno.ENOSPC)
    assert _helper()["tmpfs_fill"](fs, "/tmp", 64 * MIB, FSIZE, TAG) is False
    assert _owned_left(fs) == []


@pytest.mark.parametrize("short_once", [False, True])
def test_short_write_is_not_a_pass_when_the_next_write_succeeds(short_once):
    # capacity remains after the short write: the next write succeeds (a second
    # short one, or a full one), so the short write proved nothing
    fs = FakeFS({"/tmp": 64 * MIB}, short_after=3 * MIB, short_once=short_once)
    assert _helper()["tmpfs_fill"](fs, "/tmp", 64 * MIB, FSIZE, TAG) is False
    assert fs.attempted == 5 * MIB
    assert _owned_left(fs) == []


def test_zero_byte_write_fails_without_looping():
    fs = FakeFS({"/tmp": 64 * MIB}, zero_after=3 * MIB)
    assert _helper()["tmpfs_fill"](fs, "/tmp", 64 * MIB, FSIZE, TAG) is False
    assert fs.attempted == 4 * MIB
    assert _owned_left(fs) == [] and fs.used["/tmp"] == 0


def test_unaligned_usage_short_write_then_enospc_passes():
    # the probe's 1-byte canary leaves one page used on /scratch before the fill
    cap = policy.LIMITS["scratch_tmpfs_bytes"]
    fs = FakeFS({"/scratch": cap}, used={"/scratch": 4096})
    assert _helper()["tmpfs_fill"](fs, "/scratch", cap, FSIZE, TAG) is True
    assert 0 < fs.max_file < FSIZE
    assert fs.attempted <= cap + 2 * MIB
    assert _owned_left(fs) == [] and fs.used["/scratch"] == 4096


def test_aggregate_true_when_every_policy_mount_has_a_page_already_used():
    fs = FakeFS({path: cap for path, cap in _policy_mounts()},
                used={path: 4096 for path in policy.TMPFS})
    per_mount, ok = _helper()["tmpfs_fill_all"](fs, dict(policy.TMPFS), dict(policy.LIMITS), TAG)
    assert ok is True and all(per_mount.values())
    assert _owned_left(fs) == [] and all(fs.used[p] == 4096 for p in policy.TMPFS)


def test_short_write_in_a_files_last_slot_then_enospc_in_the_next_file_passes():
    # 62 full chunks + a short 63rd ends f0 (63 MiB per file); f1's first write is ENOSPC
    fs = FakeFS({"/tmp": 64 * MIB}, used={"/tmp": MIB + 4096})
    assert _helper()["tmpfs_fill"](fs, "/tmp", 64 * MIB, FSIZE, TAG) is True
    assert len(fs.created) == 2 and fs.max_file == 63 * MIB - 4096
    assert _owned_left(fs) == [] and fs.used["/tmp"] == MIB + 4096


def test_short_write_in_a_files_last_slot_then_success_in_the_next_file_fails():
    # the short-write state crosses the file boundary: without it, f1 would fill
    # to the 128 MiB cap and report a false ENOSPC pass
    fs = FakeFS({"/tmp": 128 * MIB}, short_after=62 * MIB, short_once=True)
    assert _helper()["tmpfs_fill"](fs, "/tmp", 128 * MIB, FSIZE, TAG) is False
    assert len(fs.created) == 2 and fs.attempted == 64 * MIB
    assert _owned_left(fs) == [] and fs.used["/tmp"] == 0


def test_close_error_fails():
    fs = FakeFS({"/tmp": 64 * MIB}, close_errno=errno.EIO)
    assert _helper()["tmpfs_fill"](fs, "/tmp", 64 * MIB, FSIZE, TAG) is False
    assert _owned_left(fs) == []


def test_unverified_cleanup_fails_and_leftovers_stay_visible():
    fs = FakeFS({"/tmp": 64 * MIB}, unlink_errno=errno.EBUSY)
    assert _helper()["tmpfs_fill"](fs, "/tmp", 64 * MIB, FSIZE, TAG) is False
    assert _owned_left(fs)  # not hidden: the probe's tmpfs_cleaned sees the dir


def test_existing_directory_is_never_reused_or_removed():
    fs = FakeFS({"/tmp": 64 * MIB})
    fs.dirs.add("/tmp/" + TAG)
    fs.files["/tmp/" + TAG + "/keep"] = 0
    assert _helper()["tmpfs_fill"](fs, "/tmp", 64 * MIB, FSIZE, TAG) is False
    assert "/tmp/" + TAG in fs.dirs and "/tmp/" + TAG + "/keep" in fs.files


@pytest.mark.parametrize("tag", ["", ".", "..", "../x", "a/b"])
def test_escaping_tag_is_refused_without_touching_fs(tag):
    fs = FakeFS({"/tmp": 64 * MIB})
    assert _helper()["tmpfs_fill"](fs, "/tmp", 64 * MIB, FSIZE, tag) is False
    assert fs.attempted == 0 and fs.dirs == {"/tmp"}


def test_unexpected_exception_still_cleans_owned_files():
    fs = FakeFS({"/tmp": 64 * MIB}, raise_after=5 * MIB)
    with pytest.raises(Boom):
        _helper()["tmpfs_fill"](fs, "/tmp", 64 * MIB, FSIZE, TAG)
    assert _owned_left(fs) == [] and fs.used["/tmp"] == 0


def test_aggregate_true_only_when_every_policy_mount_hits_enospc():
    fs = FakeFS({path: cap for path, cap in _policy_mounts()})
    per_mount, ok = _helper()["tmpfs_fill_all"](fs, dict(policy.TMPFS), dict(policy.LIMITS), TAG)
    assert ok is True and set(per_mount) == set(policy.TMPFS) and all(per_mount.values())
    assert _owned_left(fs) == []


@pytest.mark.parametrize("broken", list(policy.TMPFS))
def test_one_unenforced_mount_fails_the_aggregate(broken):
    fs = FakeFS({path: (None if path == broken else cap) for path, cap in _policy_mounts()})
    per_mount, ok = _helper()["tmpfs_fill_all"](fs, dict(policy.TMPFS), dict(policy.LIMITS), TAG)
    assert ok is False and per_mount[broken] is False
    assert all(v for p, v in per_mount.items() if p != broken)
    assert _owned_left(fs) == []


def test_no_mounts_is_not_a_pass():
    assert _helper()["tmpfs_fill_all"](FakeFS({}), {}, dict(policy.LIMITS), TAG) == ({}, False)

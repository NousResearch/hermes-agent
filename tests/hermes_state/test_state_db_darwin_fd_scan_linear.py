"""The Darwin fd scan must not copy a process's whole fd listing once per fd.

``ctypes`` arrays hand back a fresh ``bytes`` copy of the entire buffer on every
``.raw`` access. Unpacking each 8-byte ``proc_fdinfo`` record from ``listing.raw``
therefore made ``_iter_darwin_fd_targets`` quadratic in a process's fd count. A
Virtualization.framework VM (Docker Desktop, Apple ``container``, UTM) can hold
~190k vnode descriptors: the scan then took ~70s, every pass expired on the 5s
budget, and the deleted-WAL guard failed open without ever finishing (#115583).

The assertion counts buffer copies rather than timing anything, so it is
deterministic. libproc is faked throughout, so it runs on every platform.
"""

from __future__ import annotations

import ctypes
import struct

import hermes_state_dbfile as dbfile
from hermes_state_dbfile import _DARWIN_FD_RECORD_SIZE, _DARWIN_PROC_FD_INFO_SIZE

N_FDS = 2_000


class _ManyFdsLibproc:
    """One pid holding ``N_FDS`` descriptors; per-fd lookups report non-vnodes."""

    def proc_pidinfo(self, pid, flavor, arg, buf, size):
        needed = N_FDS * _DARWIN_PROC_FD_INFO_SIZE
        if size <= needed:
            return size  # buffer too small: the caller doubles and retries
        payload = b"".join(struct.pack("<iI", fd, 1) for fd in range(N_FDS))
        ctypes.memmove(buf, payload, len(payload))
        return len(payload)

    def proc_pidfdinfo(self, pid, fd, flavor, record, size):
        return 0  # not a vnode: skipped, keeps the test about the listing alone


def _counting_create_string_buffer(listing_raw_reads):
    real = ctypes.create_string_buffer

    def create_string_buffer(init, size=None):
        buffer = real(init, size) if size is not None else real(init)
        if ctypes.sizeof(buffer) == _DARWIN_FD_RECORD_SIZE:
            return buffer  # per-fd record buffers are small; only the listing matters
        base = type(buffer)

        def raw(self):
            listing_raw_reads.append(ctypes.sizeof(self))
            return base.raw.__get__(self)

        counted = type("CountedListing", (base,), {})
        # Assigned after creation: the ctypes array metaclass installs its own
        # ``raw`` getset on every new c_char array type, shadowing a class-body one.
        counted.raw = property(raw)  # pyright: ignore[reportAttributeAccessIssue]
        copy = counted()
        ctypes.memmove(copy, buffer, ctypes.sizeof(buffer))
        return copy

    return create_string_buffer


def test_fd_listing_is_not_copied_per_descriptor(monkeypatch):
    listing_raw_reads: list[int] = []
    monkeypatch.setattr(dbfile, "_darwin_libproc", lambda: _ManyFdsLibproc())
    monkeypatch.setattr(dbfile, "_darwin_all_pids", lambda _lib: [4242])
    monkeypatch.setattr(
        ctypes, "create_string_buffer", _counting_create_string_buffer(listing_raw_reads)
    )

    assert list(dbfile._iter_darwin_fd_targets()) == []
    # Pre-fix this was N_FDS full-listing copies (one per descriptor).
    assert len(listing_raw_reads) <= 1, (
        f"fd listing copied {len(listing_raw_reads)} times for {N_FDS} fds: "
        "the scan is quadratic in a process's descriptor count"
    )

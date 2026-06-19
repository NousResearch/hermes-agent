"""Tests for cgroup-aware TUI V8 heap sizing.

V8 is not cgroup-aware: a flat ``--max-old-space-size=8192`` lets the heap grow
toward 8GB in a memory-limited container, so the cgroup OOM-killer SIGKILLs Node
before V8's own monitor fires — leaving the user with only a bare gateway
``stdin EOF`` and no breadcrumb. ``_resolve_tui_heap_mb`` reads the real cgroup
limit and sizes the cap below it so V8 exits gracefully instead.
"""

import builtins
import io
from unittest import mock

import hermes_cli.main as m

V2 = "/sys/fs/cgroup/memory.max"
V1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
GB = 1024 ** 3


def _fake_open(files: dict):
    """Return an open() shim serving cgroup paths from ``files`` (path->str)."""
    real_open = builtins.open

    def opener(path, *args, **kwargs):
        if path in (V2, V1):
            content = files.get(path)
            if content is None:
                raise FileNotFoundError(path)
            return io.StringIO(content)
        return real_open(path, *args, **kwargs)

    return opener


def _read(files: dict):
    with mock.patch.object(builtins, "open", _fake_open(files)):
        return m._read_cgroup_memory_limit()


class TestReadCgroupMemoryLimit:
    def test_v2_max_is_unlimited(self):
        assert _read({V2: "max"}) is None

    def test_v2_numeric_limit(self):
        assert _read({V2: str(4 * GB)}) == 4 * GB

    def test_v1_unlimited_sentinel_is_none(self):
        # cgroup v1 reports "unlimited" as a near-INT64 huge value.
        assert _read({V1: "9223372036854771712"}) is None

    def test_v1_numeric_limit_when_no_v2(self):
        assert _read({V1: str(2 * GB)}) == 2 * GB

    def test_no_files_present(self):
        assert _read({}) is None

    def test_empty_v2_falls_through_to_v1(self):
        # A blank v2 file must NOT be mistaken for "unlimited" — fall to v1.
        assert _read({V2: "", V1: str(3 * GB)}) == 3 * GB

    def test_v2_wins_over_v1(self):
        assert _read({V2: str(6 * GB), V1: str(2 * GB)}) == 6 * GB

    def test_zero_is_skipped(self):
        assert _read({V2: "0"}) is None

    def test_petabyte_plus_treated_as_unlimited(self):
        assert _read({V2: str(1 << 51)}) is None


class TestResolveTuiHeapMb:
    def _resolve(self, limit_bytes):
        with mock.patch.object(m, "_read_cgroup_memory_limit", return_value=limit_bytes):
            return m._resolve_tui_heap_mb()

    def test_unconstrained_uses_default(self):
        assert self._resolve(None) == 8192

    def test_large_container_clamps_to_default(self):
        # 16GB -> 75% = 12288 >= 8192 -> clamp to 8192.
        assert self._resolve(16 * GB) == 8192

    def test_4gb_container_75_percent(self):
        assert self._resolve(4 * GB) == 3072

    def test_3gb_container_above_floor(self):
        assert self._resolve(3 * GB) == 2304

    def test_2gb_container_at_floor(self):
        assert self._resolve(2 * GB) == 1536

    def test_tiny_container_honors_limit_below_floor(self):
        # 1GB -> 75% = 768; honored even though below the 1536 floor, because a
        # graceful V8 exit beats a silent cgroup SIGKILL.
        assert self._resolve(1 * GB) == 768

    def test_never_exceeds_default(self):
        assert self._resolve(64 * GB) == 8192


class TestNodeOptionsTokenMerge:
    """The _launch_tui token-merge block must add the sized cap unless the user
    already supplied one, and must preserve unrelated NODE_OPTIONS flags."""

    def _merge(self, node_options, limit_bytes):
        with mock.patch.object(m, "_read_cgroup_memory_limit", return_value=limit_bytes):
            tokens = node_options.split()
            if not any(t.startswith("--max-old-space-size=") for t in tokens):
                tokens.append(f"--max-old-space-size={m._resolve_tui_heap_mb()}")
            return " ".join(tokens)

    def test_unconstrained_empty(self):
        assert self._merge("", None) == "--max-old-space-size=8192"

    def test_constrained_container(self):
        assert self._merge("", 4 * GB) == "--max-old-space-size=3072"

    def test_user_override_respected(self):
        assert self._merge("--max-old-space-size=12288", 2 * GB) == "--max-old-space-size=12288"

    def test_preserves_other_flags(self):
        assert self._merge("--enable-source-maps", 4 * GB) == "--enable-source-maps --max-old-space-size=3072"

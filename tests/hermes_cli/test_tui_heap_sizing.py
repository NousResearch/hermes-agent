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

from hermes_cli import main_tui_launch

V2 = "/sys/fs/cgroup/memory.max"
V1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

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
        return main_tui_launch._read_cgroup_memory_limit()

class TestReadCgroupMemoryLimit:
    def test_v2_max_is_unlimited(self):
        assert _read({V2: "max"}) is None

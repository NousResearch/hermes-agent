"""Regression tests: binary contents must never reach the shlex tokenizer.

Bug (2026-08-30): terminal_tool._read_script_in_env returned decoded ELF bytes
(no NUL check), overriding lifecycle_guard._read_referenced_script's binary
short-circuit. The guard then tokenized ~1MB of machine code in pure-Python
shlex, holding the GIL for tens of minutes with no timeout — the tool call
appeared to hang and loadavg pinned above 8 on a 4-core Pi.
"""
import os
import sys
import time
from pathlib import Path

import pytest

# parents[2] = repo root: parents[1] resolves to tests/, whose empty
# tests/cron/__init__.py shadows the real cron package and makes this file
# import the installed (uncapped) cron.lifecycle_guard instead of the repo's.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cron.lifecycle_guard import (
    _iter_command_segments,
    contains_gateway_lifecycle_command_or_referenced_script,
)


def _binary_payload(tmp_path: Path, size: int = 300_000) -> Path:
    """An ELF-shaped file: some NULs + one very long 'line' of machine code."""
    blob = b"\x7fELF\x00\x01\x01\x00" + os.urandom(size) + b"\x00\x00"
    p = tmp_path / "fakebin"
    p.write_bytes(blob)
    return p


def test_iter_segments_skips_giant_lines():
    huge = "x" * 300_000
    # giant content on its own line: skipped entirely (fast)
    t0 = time.perf_counter()
    assert list(_iter_command_segments(huge)) == []
    assert time.perf_counter() - t0 < 2.0
    # normal lines still parse
    segs = list(_iter_command_segments("echo ok"))
    assert ["echo", "ok"] in segs


def test_guard_with_buggy_binary_reader_is_fast(tmp_path):
    """Even a NUL-unaware reader cannot wedge the guard (cap defense)."""
    def buggy_reader(p):
        p = Path(p).expanduser()
        if p.is_file() and p.stat().st_size <= 1024 * 1024:
            return p.read_bytes().decode("utf-8", errors="replace")
        return None

    blob = _binary_payload(tmp_path)
    t0 = time.perf_counter()
    result = contains_gateway_lifecycle_command_or_referenced_script(
        str(blob) + " --help", cwd=str(tmp_path), read_remote_script=buggy_reader
    )
    elapsed = time.perf_counter() - t0
    assert result is False
    assert elapsed < 2.0


def test_guard_binary_reader_with_nul_check_skips(tmp_path):
    """Reader implementing the terminal_tool fix: binary => None."""
    def fixed_reader(p):
        p = Path(p).expanduser()
        if p.is_file() and p.stat().st_size <= 1024 * 1024:
            data = p.read_bytes()
            if b"\x00" in data:
                return None
            return data.decode("utf-8", errors="replace")
        return None

    blob = _binary_payload(tmp_path)
    t0 = time.perf_counter()
    assert contains_gateway_lifecycle_command_or_referenced_script(
        str(blob) + " --help", cwd=str(tmp_path), read_remote_script=fixed_reader
    ) is False
    assert time.perf_counter() - t0 < 2.0


def test_guard_still_blocks_lifecycle_commands(tmp_path):
    def fixed_reader(p):
        return None

    assert contains_gateway_lifecycle_command_or_referenced_script(
        "systemctl --user restart hermes-gateway", cwd=str(tmp_path),
        read_remote_script=fixed_reader,
    ) is True
    assert contains_gateway_lifecycle_command_or_referenced_script(
        "ls -la; echo ok", cwd=str(tmp_path), read_remote_script=fixed_reader,
    ) is False

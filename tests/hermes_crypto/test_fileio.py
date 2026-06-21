"""Tests for the stdlib-only fileio helpers added by hardening work.

Cross-platform smoke tests verify the helpers do not crash and produce the
expected on-disk state; the POSIX mode-bit assertions are skipped on Windows
where ``os.chmod`` is a silent no-op for directories and POSIX permission
bits don't apply.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from hermes_crypto.fileio import harden_dir, open_private_append

posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX mode bits don't apply on Windows; chmod is a silent no-op",
)


# ── Cross-platform smoke tests ──────────────────────────────────────────────


def test_harden_dir_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir"
    harden_dir(target)
    assert target.is_dir()


def test_open_private_append_writes_and_appends(tmp_path: Path) -> None:
    target = tmp_path / "log.jsonl"
    fh = open_private_append(target)
    try:
        fh.write("first\n")
        fh.flush()
    finally:
        fh.close()
    assert target.read_text() == "first\n"

    fh = open_private_append(target)
    try:
        fh.write("second\n")
        fh.flush()
    finally:
        fh.close()
    assert target.read_text() == "first\nsecond\n"


# ── POSIX-only permission-bit assertions ────────────────────────────────────


@posix_only
def test_harden_dir_default_sets_0o700(tmp_path: Path) -> None:
    target = tmp_path / "private"
    harden_dir(target)
    assert (target.stat().st_mode & 0o777) == 0o700


@posix_only
def test_harden_dir_group_readable_sets_0o770(tmp_path: Path) -> None:
    target = tmp_path / "managed_logs"
    harden_dir(target, group_readable=True)
    assert (target.stat().st_mode & 0o777) == 0o770


@posix_only
def test_open_private_append_creates_0o600(tmp_path: Path) -> None:
    target = tmp_path / "session.jsonl"
    fh = open_private_append(target)
    try:
        fh.write('{"role": "user"}\n')
        fh.flush()
    finally:
        fh.close()
    assert (target.stat().st_mode & 0o777) == 0o600


@posix_only
def test_open_private_append_tightens_existing_loose_perms(tmp_path: Path) -> None:
    """If the path already exists with looser perms, opening it tightens to 0o600."""
    target = tmp_path / "preexisting.jsonl"
    target.write_text("seed\n")
    os.chmod(target, 0o644)
    assert (target.stat().st_mode & 0o777) == 0o644

    fh = open_private_append(target)
    try:
        fh.write("appended\n")
        fh.flush()
    finally:
        fh.close()
    assert target.read_text() == "seed\nappended\n"
    assert (target.stat().st_mode & 0o777) == 0o600

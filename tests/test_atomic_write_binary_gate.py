"""``_atomic_write``'s binary path must reach ``os.fdopen`` without a ``newline`` argument.

PR #135677 added ``newline=''`` to the fdopen call so a text-mode read-modify-write
rollback reproduces the read content byte-exactly, but applied it to both modes.
CPython rejects the argument in binary mode (``ValueError: binary mode doesn't take
a newline argument``), which broke ``atomic_write_bytes`` — the repo's only
``binary=True`` entry point, used for encrypted vault blobs
(``agent/vault_store.py``) and SOUL.md setup (``hermes_cli/setup_profile.py``).
"""
from __future__ import annotations

from pathlib import Path

from utils import atomic_write_bytes


def test_atomic_write_bytes_binary_mode_writes_content(tmp_path: Path) -> None:
    """Regression: binary=True must work end-to-end (previously raised ValueError)."""
    target = tmp_path / "blob.bin"
    payload = b"\x00\x01\r\nbytes\xff"

    atomic_write_bytes(target, payload, mode=0o600)

    assert target.read_bytes() == payload
    assert target.stat().st_mode & 0o777 == 0o600


def test_atomic_write_bytes_overwrite_preserves_content(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(b"v1")

    atomic_write_bytes(target, b"v2")

    assert target.read_bytes() == b"v2"

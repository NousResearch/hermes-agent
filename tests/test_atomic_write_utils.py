import os
from pathlib import Path

import pytest

from utils import atomic_bytes_write, atomic_text_write


def test_atomic_text_write_preserves_existing_file_mode(tmp_path: Path):
    target = tmp_path / "state.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o644)

    atomic_text_write(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert (target.stat().st_mode & 0o777) == 0o644


def test_atomic_bytes_write_round_trips_binary_payload(tmp_path: Path):
    target = tmp_path / "state.bin"
    payload = b"\x00hermes\xff"

    atomic_bytes_write(target, payload)

    assert target.read_bytes() == payload


def test_atomic_text_write_keeps_original_on_replace_failure(monkeypatch, tmp_path: Path):
    target = tmp_path / "state.txt"
    target.write_text("old", encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_text_write(target, "new")

    assert target.read_text(encoding="utf-8") == "old"


def test_atomic_text_write_cleans_temp_file_on_failure(monkeypatch, tmp_path: Path):
    target = tmp_path / "state.txt"
    target.write_text("old", encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError):
        atomic_text_write(target, "new")

    leftovers = list(tmp_path.glob(".state_*.tmp"))
    assert leftovers == []

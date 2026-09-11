"""Behavioral tests for exact-interpreter SQLite runtime inspection."""

from __future__ import annotations

import json
import os
import shlex
import sqlite3
import sys
from pathlib import Path

import pytest

from hermes_cli.sqlite_runtime import (
    is_sqlite_wal_reset_vulnerable,
    probe_sqlite_runtime,
)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ((3, 6, 23), False),
        ((3, 7, 0), True),
        ((3, 44, 5), True),
        ((3, 44, 6), False),
        ((3, 45, 0), True),
        ((3, 50, 6), True),
        ((3, 50, 7), False),
        ((3, 51, 2), True),
        ((3, 51, 3), False),
        ((3, 53, 1), False),
    ],
)
def test_wal_reset_vulnerability_matrix(
    version: tuple[int, ...],
    expected: bool,
) -> None:
    assert is_sqlite_wal_reset_vulnerable(version) is expected


def test_probe_reports_the_requested_interpreters_linked_sqlite() -> None:
    info = probe_sqlite_runtime(sys.executable)

    assert info is not None
    assert info.executable.resolve() == Path(sys.executable).resolve()
    assert info.base_prefix.resolve() == Path(sys.base_prefix).resolve()
    assert info.python_version == sys.version_info[:3]
    assert info.sqlite_version == sqlite3.sqlite_version_info
    assert info.sqlite_version_string == sqlite3.sqlite_version

    with sqlite3.connect(":memory:") as conn:
        source_id = conn.execute("SELECT sqlite_source_id()").fetchone()[0]
    assert info.sqlite_source_id == source_id


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX executable probe stub")
def test_probe_uses_child_payload_and_sanitizes_python_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_python = tmp_path / "reported-python"
    payload = {
        "base_prefix": str(tmp_path / "reported-base"),
        "executable": str(fake_python),
        "python_version": [3, 11, 15],
        "sqlite_version": [9, 8, 7],
        "sqlite_version_string": "9.8.7-child",
        "sqlite_source_id": "child-source-id",
    }
    fake_python.write_text(
        "\n".join([
            "#!/bin/sh",
            '[ "$1" = "-I" ] && [ "$2" = "-c" ] || exit 10',
            '[ -z "${PYTHONHOME+x}" ] || exit 11',
            '[ -z "${PYTHONPATH+x}" ] || exit 12',
            f"printf '%s\\n' {shlex.quote(json.dumps(payload))}",
        ])
        + "\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "poison-home"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "poison-path"))

    info = probe_sqlite_runtime(fake_python)

    assert info is not None
    assert info.executable == fake_python
    assert info.base_prefix == tmp_path / "reported-base"
    assert info.sqlite_version == (9, 8, 7)
    assert info.sqlite_version_string == "9.8.7-child"
    assert info.sqlite_source_id == "child-source-id"

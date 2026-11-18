"""A referenced SQLite ledger is a mention, not an unreadable script (#303).

The reference walk demands that every *script* it follows be scannable. A database file named
inside a script body (an ops ledger, `ops.db`) is not a script: a POSIX shell cannot execute it,
and refusing the whole command because a 2.7 MB ledger cannot be tokenized takes the daily
dispatch preflight offline.

The control is the other direction: an oversized *script* (shebang, no data magic) must still
fail closed, so the exemption cannot be widened into "big files are fine".
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_repo = Path(__file__).resolve().parents[2]
for _cand in (_repo, _repo / "agent"):  # real repo: <repo>/tests/cron/*; fixture tree: <tree>/agent/tests/cron/*
    if (_cand / "cron" / "lifecycle_guard.py").is_file():
        sys.path.insert(0, str(_cand))
        break

from cron.lifecycle_guard import _is_data_file, scan_gateway_lifecycle  # noqa: E402

SQLITE_HEADER = b"SQLite format 3\x00"


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


class DataFileIsNotAScript(unittest.TestCase):
    def test_sqlite_ledger_is_a_data_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = _write(Path(tmp) / "ops.db", SQLITE_HEADER + b"\x00" * (2 * 1024 * 1024))
            self.assertTrue(_is_data_file(db))

    def test_shell_script_is_not_a_data_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sh = _write(Path(tmp) / "big.sh", b"#!/bin/sh\n" + b"# pad\n" * (512 * 1024))
            self.assertFalse(_is_data_file(sh))

    def test_big_db_reference_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = _write(Path(tmp) / "ops.db", SQLITE_HEADER + b"\x00" * (2 * 1024 * 1024))
            script = Path(tmp) / "preflight.sh"
            script.write_text(f"#!/bin/sh\nsqlite3 {db} 'select 1'\n", encoding="utf-8")
            unsafe, refusal = scan_gateway_lifecycle(f"bash {script}", cwd=tmp)
            self.assertFalse(unsafe, f"a ledger mention must not be a refusal: {refusal}")
            self.assertFalse(refusal, f"a ledger mention must not name a refusal: {refusal}")

    def test_oversized_script_still_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp) / "big.sh"
            big.write_text("#!/bin/sh\n" + "# pad\n" * (300 * 1024), encoding="utf-8")
            unsafe, refusal = scan_gateway_lifecycle(f"bash {big}", cwd=tmp)
            self.assertTrue(unsafe, "an oversized script must still fail closed")
            self.assertTrue(refusal)

    def test_lifecycle_command_in_a_small_script_is_still_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "small.sh"
            script.write_text(
                "#!/bin/sh\nsystemctl --user restart hermes-gateway.service\n", encoding="utf-8"
            )
            unsafe, _ = scan_gateway_lifecycle(f"bash {script}", cwd=tmp)
            self.assertTrue(unsafe, "the exemption must not weaken the lifecycle block")


if __name__ == "__main__":
    unittest.main()

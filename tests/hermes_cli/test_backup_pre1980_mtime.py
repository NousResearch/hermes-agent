"""The automatic zip writer keeps a pre-1980 member (#153657)."""

from __future__ import annotations

import os
import zipfile


def test_automatic_writer_keeps_pre1980_member(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    ancient = home / "ancient.txt"
    ancient.write_text("crate from 1970\n", encoding="utf-8")
    os.utime(ancient, (0, 0))

    from hermes_cli.backup import _write_full_zip_backup

    out = tmp_path / "automatic.zip"
    assert _write_full_zip_backup(out, home) == out
    with zipfile.ZipFile(out) as zf:
        assert zf.read("ancient.txt") == b"crate from 1970\n"
        assert zf.getinfo("ancient.txt").date_time[:3] == (1980, 1, 1)

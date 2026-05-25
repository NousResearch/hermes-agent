"""Regression coverage for the ``libopentui.so`` /tmp leak fix (#32283).

The TUI launcher now redirects ``TMPDIR`` to a profile-scoped path under
``HERMES_HOME`` and sweeps the leak-specific ``.<hex>-<digits>.so``
files before spawning the Ink child.  These tests pin both behaviours
so a future refactor cannot silently re-enable the leak.
"""

from __future__ import annotations

import pytest

from hermes_cli.tui_tmpdir import (
    is_leak_filename,
    prepare_tui_tmpdir,
    sweep_libopentui_leaks,
)


class TestIsLeakFilename:
    @pytest.mark.parametrize(
        "name",
        [
            ".f9f3dfe75bcf4fff-00000000.so",  # canonical shape from the bug report
            ".deadbeef-1.so",
            ".aabbccddeeff0011-12345.so",
            ".0123456789abcdef0123456789abcdef-9.so",  # max-length hex digest
        ],
    )
    def test_matches_leak_shapes(self, name):
        assert is_leak_filename(name)

    @pytest.mark.parametrize(
        "name",
        [
            "libopentui.so",                       # legit shared object
            "f9f3dfe75bcf4fff-00000000.so",        # missing leading dot
            ".f9f3dfe75bcf4fff-00000000.txt",      # wrong extension
            ".F9F3DFE7-0.so",                      # uppercase hex (extractor uses lowercase)
            ".abcd-0.dylib",                       # not .so
            ".abc-12345678901234.so",              # counter too long (>12 digits)
            ".abc-.so",                            # missing counter
            ".-1.so",                              # missing hex
            ".session-active.json",                # unrelated tempfile shape
        ],
    )
    def test_does_not_match_unrelated_files(self, name):
        assert not is_leak_filename(name)


class TestSweep:
    def test_removes_only_leak_files(self, tmp_path):
        leaks = [tmp_path / f".deadbeef-{i}.so" for i in range(3)]
        keep = [
            tmp_path / "libopentui.so",
            tmp_path / ".session-active.json",
            tmp_path / ".abc-1.dylib",
        ]
        for p in leaks + keep:
            p.write_bytes(b"x")

        freed = sweep_libopentui_leaks([tmp_path])

        assert freed == 3
        assert not any(p.exists() for p in leaks)
        assert all(p.exists() for p in keep)

    def test_missing_root_is_silent(self, tmp_path):
        ghost = tmp_path / "does-not-exist"
        assert sweep_libopentui_leaks([ghost]) == 0

    def test_skips_files_passed_as_roots(self, tmp_path):
        f = tmp_path / "regular-file"
        f.write_bytes(b"x")
        assert sweep_libopentui_leaks([f]) == 0
        assert f.exists()

    def test_sweeps_across_multiple_roots(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / ".aaaaaaaa-1.so").write_bytes(b"x")
        (b / ".bbbbbbbb-2.so").write_bytes(b"x")
        assert sweep_libopentui_leaks([a, b]) == 2

    def test_unlink_errors_are_swallowed(self, tmp_path, monkeypatch):
        leak = tmp_path / ".deadbeef-1.so"
        leak.write_bytes(b"x")
        from pathlib import Path as _P

        def boom(self):
            raise PermissionError("read-only fs")

        monkeypatch.setattr(_P, "unlink", boom)
        # Should not raise, and should count zero (unlink failed for the
        # only file present).
        assert sweep_libopentui_leaks([tmp_path]) == 0


class TestPrepareTuiTmpdir:
    def test_sets_tmpdir_under_hermes_home_when_unset(self, tmp_path):
        env: dict = {}
        scoped = prepare_tui_tmpdir(env, tmp_path)

        assert scoped == tmp_path / "run" / "tui-tmp"
        assert env["TMPDIR"] == str(scoped)
        assert scoped.is_dir()

    def test_honours_existing_tmpdir(self, tmp_path):
        custom = tmp_path / "custom-tmp"
        env = {"TMPDIR": str(custom)}

        scoped = prepare_tui_tmpdir(env, tmp_path / "irrelevant")

        assert scoped == custom
        assert env["TMPDIR"] == str(custom)
        assert custom.is_dir()

    def test_sweeps_scoped_dir_before_returning(self, tmp_path):
        custom = tmp_path / "custom-tmp"
        custom.mkdir()
        leak = custom / ".cafef00d-7.so"
        keep = custom / "libopentui.so"
        leak.write_bytes(b"x")
        keep.write_bytes(b"x")

        prepare_tui_tmpdir({"TMPDIR": str(custom)}, tmp_path)

        assert not leak.exists()
        assert keep.exists()

    def test_empty_tmpdir_is_treated_as_unset(self, tmp_path):
        env = {"TMPDIR": "   "}
        scoped = prepare_tui_tmpdir(env, tmp_path)
        assert scoped == tmp_path / "run" / "tui-tmp"
        assert env["TMPDIR"] == str(scoped)

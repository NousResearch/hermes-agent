"""Windows NT/device-namespace paths are rejected on the RAW string, before any resolution.

On Windows, merely resolving ``\\??\\UNC\\host\\share`` (or the ``\\\\?\\UNC\\`` /
``GLOBALROOT`` re-entry forms) initiates SMB authentication and leaks the user's
NTLM hash, and the prefixes bypass the Win32 normalization that prefix denylists
rely on. So the guard must fire on the model-supplied string before
``resolve()``/``realpath()``/the task-base join, on every platform.
"""

import os
import unittest
from unittest.mock import patch

from agent.file_safety import (
    get_read_block_error,
    get_write_denied_error,
    is_nt_namespace_path,
)

BLOCKED_PATHS = [
    "\\??\\UNC\\attacker.example\\share\\x",        # NT object namespace — the canonical leak form
    "\\??\\C:\\Windows\\System32\\config\\SAM",
    "/??/UNC/attacker.example/share/x",             # forward-slash spelling
    "\\\\.\\PhysicalDrive0",                        # Win32 device namespace
    "\\\\.\\pipe\\evil",
    "//./pipe/evil",
    "\\\\?\\UNC\\attacker.example\\share\\x",        # extended-length UNC (remote host)
    "\\\\?\\unc\\attacker.example\\share\\x",        # case-insensitive
    "\\\\?\\GLOBALROOT\\Device\\HarddiskVolume1\\x",  # NT-namespace re-entry
    "//?/UNC/attacker.example/share/x",
]

ALLOWED_PATHS = [
    "/tmp/test.py",
    "C:\\Users\\me\\notes.txt",
    "~/projects/readme.md",
    "relative/path.txt",
    "\\\\?\\C:\\Users\\me\\notes.txt",   # extended-length LOCAL drive: routine, no remote-auth trigger
    "\\\\server\\share\\file.txt",       # plain UNC share: separate policy question, not this guard
    "//server/share/file.txt",
    "/tmp/??/weird-dir/file",            # '??' mid-path is data, not a namespace marker
]


class TestNtNamespaceGuard(unittest.TestCase):
    def test_predicate_and_both_chokepoint_classifiers(self):
        for p in BLOCKED_PATHS:
            self.assertTrue(is_nt_namespace_path(p), p)
            self.assertIn("NT/device namespace", get_read_block_error(p) or "", p)
            self.assertIn("Write denied", get_write_denied_error(p) or "", p)
        for p in ALLOWED_PATHS:
            self.assertFalse(is_nt_namespace_path(p), p)
            self.assertNotIn("NT/device namespace", get_read_block_error(p) or "", p)
            self.assertNotIn("NT/device namespace", get_write_denied_error(p) or "", p)

    def test_file_tools_reject_raw_string_without_resolving(self):
        """Every file-tool entry refuses BEFORE the task-base join / resolve — the
        resolve is the leak, and on POSIX the join would hide the prefix."""
        from tools.file_tools import patch_tool, read_file_tool, search_tool, write_file_tool

        bad = "\\??\\UNC\\attacker.example\\share\\x"
        with patch("agent.file_safety.Path") as fs_path, \
                patch("tools.file_tools._resolve_path_for_task") as ft_resolve, \
                patch("tools.file_tools_write_guards._resolve_path_for_task") as wg_resolve, \
                patch.object(os.path, "realpath", side_effect=AssertionError("must not realpath")):
            results = [
                read_file_tool(bad),
                write_file_tool(bad, "data"),
                patch_tool(mode="replace", path=bad, old_string="a", new_string="b"),
                search_tool("x", path=bad),
            ]
            fs_path.assert_not_called()
            ft_resolve.assert_not_called()
            wg_resolve.assert_not_called()
        for r in results:
            self.assertIn("NT/device namespace", r)


if __name__ == "__main__":
    unittest.main()

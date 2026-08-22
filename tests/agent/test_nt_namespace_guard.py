"""Tests for the Windows NT-namespace path guard in agent.file_safety.

Inspired by Claude Code v2.1.234 (Aug 2026): file accesses reject Windows
NT-namespace (``\\??\\``) paths to harden against the NTLM credential-leak
vector — merely resolving ``\\??\\UNC\\host\\share`` triggers outbound SMB
authentication on Windows, and NT-namespace prefixes bypass the Win32 path
normalization that prefix-based denylists rely on.

The guard must fire on the RAW string before any ``resolve()``/``realpath()``
call, on every platform (path strings can be relayed toward Windows hosts by
remote terminal backends and desktop bridges).
"""

import unittest

from agent.file_safety import (
    get_nt_namespace_error,
    get_read_block_error,
    get_write_denied_error,
    is_nt_namespace_path,
    is_write_denied,
)


BLOCKED_PATHS = [
    # NT object namespace — the canonical NTLM-leak form.
    "\\??\\UNC\\attacker.example\\share\\x",
    "\\??\\C:\\Windows\\System32\\config\\SAM",
    # Forward-slash spellings normalize to the same namespace on Windows.
    "/??/UNC/attacker.example/share/x",
    # Win32 device namespace.
    "\\\\.\\PhysicalDrive0",
    "\\\\.\\pipe\\evil",
    "//./pipe/evil",
    # Extended-length UNC (remote host) and NT-namespace re-entry.
    "\\\\?\\UNC\\attacker.example\\share\\x",
    "\\\\?\\unc\\attacker.example\\share\\x",  # case-insensitive
    "\\\\?\\GLOBALROOT\\Device\\HarddiskVolume1\\x",
    "//?/UNC/attacker.example/share/x",
]

ALLOWED_PATHS = [
    # Ordinary paths on both platforms.
    "/tmp/test.py",
    "/home/user/notes.txt",
    "C:\\Users\\me\\notes.txt",
    "C:/Users/me/notes.txt",
    "~/projects/readme.md",
    "relative/path.txt",
    # Extended-length local drive paths are routine and carry no remote-auth
    # trigger (see hermes_cli/windows_ssh_runtime.py).
    "\\\\?\\C:\\Users\\me\\notes.txt",
    # Plain UNC shares are a separate policy question — NOT blocked by this
    # namespace guard.
    "\\\\server\\share\\file.txt",
    "//server/share/file.txt",
    # A '??' mid-path is data, not a namespace marker.
    "/tmp/??/weird-dir/file",
]


class TestIsNtNamespacePath(unittest.TestCase):
    def test_blocked_forms(self):
        for path in BLOCKED_PATHS:
            self.assertTrue(is_nt_namespace_path(path), f"{path!r} should be detected")

    def test_allowed_forms(self):
        for path in ALLOWED_PATHS:
            self.assertFalse(is_nt_namespace_path(path), f"{path!r} should NOT be detected")


class TestReadGuard(unittest.TestCase):
    def test_read_blocked_with_actionable_message(self):
        for path in BLOCKED_PATHS:
            err = get_read_block_error(path)
            self.assertIsNotNone(err, f"{path!r} should be read-blocked")
            self.assertIn("Read denied", err)
            self.assertIn("NT/device namespace", err)

    def test_legitimate_reads_unaffected(self):
        for path in ("/tmp/test.py", "\\\\?\\C:\\Users\\me\\notes.txt"):
            err = get_read_block_error(path)
            if err is not None:
                # Must not be blocked by THIS guard (other denylist rules
                # could theoretically match in exotic environments).
                self.assertNotIn("NT/device namespace", err)

    def test_guard_does_not_resolve_the_path(self):
        """The check must run on the raw string — resolving is the leak."""
        from unittest.mock import patch

        with patch("agent.file_safety.Path") as mock_path:
            err = get_read_block_error("\\??\\UNC\\attacker.example\\share\\x")
            self.assertIsNotNone(err)
            mock_path.assert_not_called()


class TestWriteGuard(unittest.TestCase):
    def test_write_blocked(self):
        for path in BLOCKED_PATHS:
            self.assertTrue(is_write_denied(path), f"{path!r} should be write-denied")
            err = get_write_denied_error(path)
            self.assertIsNotNone(err)
            self.assertIn("NT/device namespace", err)
            self.assertIn("Write denied", err)

    def test_legitimate_writes_unaffected(self):
        self.assertFalse(is_write_denied("/tmp/scratch/output.txt"))
        self.assertFalse(is_write_denied("\\\\?\\C:\\Users\\me\\notes.txt"))


class TestErrorHelper(unittest.TestCase):
    def test_none_for_normal_paths(self):
        self.assertIsNone(get_nt_namespace_error("/tmp/x"))

    def test_verb_customization(self):
        err = get_nt_namespace_error("\\??\\UNC\\h\\s\\f", verb="Read")
        assert err is not None
        self.assertTrue(err.startswith("Read denied"))


class TestToolLayerChokepoints(unittest.TestCase):
    """The raw string must be rejected at the tool entry points, before the
    task-base join anchors ``\\??\\...`` under a POSIX base dir and hides the
    prefix from resolved-path checks."""

    BAD = "\\??\\UNC\\attacker.example\\share\\x"

    def test_read_file_tool_rejects_raw(self):
        from tools.file_tools import read_file_tool

        result = read_file_tool(self.BAD)
        self.assertIn("NT/device namespace", result)

    def test_write_file_tool_rejects_raw(self):
        from tools.file_tools import write_file_tool

        result = write_file_tool(self.BAD, "data")
        self.assertIn("NT/device namespace", str(result))

    def test_patch_tool_rejects_raw(self):
        from tools.file_tools import patch_tool

        result = patch_tool(mode="replace", path=self.BAD,
                            old_string="a", new_string="b")
        self.assertIn("NT/device namespace", str(result))


if __name__ == "__main__":
    unittest.main()

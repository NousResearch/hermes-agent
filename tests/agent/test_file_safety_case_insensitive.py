"""Case-insensitive filesystem coverage for agent/file_safety.py deny guards.

On macOS (APFS/HFS+) and Windows the filesystem is case-insensitive, so a
credential path addressed through a case variant (``~/.SSH/authorized_keys``
vs ``~/.ssh/authorized_keys``, or ``.ENV`` vs ``.env``) is the SAME on-disk
file. ``os.path.realpath`` / ``Path.resolve`` preserve the typed case rather
than canonicalizing it, so the deny comparisons must casefold on those
platforms or the guard is silently bypassed. On case-sensitive POSIX
filesystems (default Linux) the variants are genuinely different files, so the
guards must stay exact-case. These tests pin both behaviors by forcing the
``_fs_case_insensitive`` detector on and off, which keeps them deterministic on
every CI host.

Run with: python -m pytest tests/agent/test_file_safety_case_insensitive.py -v
"""

import os
import sys

import pytest

import agent.file_safety as file_safety
from agent.file_safety import get_read_block_error, is_write_denied


def _force_case_insensitive(monkeypatch, value):
    monkeypatch.setattr(file_safety, "_fs_case_insensitive", lambda: value)


@pytest.fixture(autouse=True)
def _no_safe_root(monkeypatch):
    # HERMES_WRITE_SAFE_ROOT turns is_write_denied into an allowlist; keep it
    # unset so these denylist assertions are deterministic.
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)


class TestWriteDenyCaseInsensitive:
    def test_ssh_exact_case_is_denied(self, monkeypatch):
        _force_case_insensitive(monkeypatch, False)
        target = os.path.join(os.path.expanduser("~"), ".ssh", "authorized_keys")
        assert is_write_denied(target) is True

    def test_ssh_case_variant_denied_on_case_insensitive_fs(self, monkeypatch):
        _force_case_insensitive(monkeypatch, True)
        # Same on-disk file on APFS/NTFS, so it must be denied.
        target = os.path.join(os.path.expanduser("~"), ".SSH", "authorized_keys")
        assert is_write_denied(target) is True

    def test_aws_prefix_case_variant_denied_on_case_insensitive_fs(self, monkeypatch):
        _force_case_insensitive(monkeypatch, True)
        target = os.path.join(os.path.expanduser("~"), ".AWS", "credentials")
        assert is_write_denied(target) is True

    def test_etc_exact_path_case_variant_denied_on_case_insensitive_fs(self, monkeypatch):
        _force_case_insensitive(monkeypatch, True)
        assert is_write_denied("/etc/PASSWD") is True

    def test_case_variant_not_denied_on_case_sensitive_fs(self, monkeypatch):
        _force_case_insensitive(monkeypatch, False)
        # On a case-sensitive FS these are genuinely different files: behavior
        # must be unchanged (no over-blocking).
        assert is_write_denied("/etc/SUDOERS") is False
        target = os.path.join(os.path.expanduser("~"), ".SSH", "authorized_keys")
        assert is_write_denied(target) is False


class TestReadBlockCaseInsensitive:
    def test_env_exact_case_is_blocked(self, monkeypatch):
        _force_case_insensitive(monkeypatch, False)
        assert get_read_block_error("/tmp/project/.env") is not None

    def test_env_case_variant_blocked_on_case_insensitive_fs(self, monkeypatch):
        _force_case_insensitive(monkeypatch, True)
        assert get_read_block_error("/tmp/project/.ENV") is not None
        assert get_read_block_error("/tmp/project/.Env") is not None

    def test_env_case_variant_not_blocked_on_case_sensitive_fs(self, monkeypatch):
        _force_case_insensitive(monkeypatch, False)
        # Different file on Linux, so it must not be blocked.
        assert get_read_block_error("/tmp/project/.ENV") is None


@pytest.mark.skipif(
    not (sys.platform == "darwin" or os.name == "nt"),
    reason="real case-insensitive filesystem (macOS/Windows) only",
)
class TestRealCaseInsensitiveFilesystem:
    """End-to-end on an actual case-insensitive FS, no monkeypatching."""

    def test_real_ssh_variant_write_denied(self):
        target = os.path.join(os.path.expanduser("~"), ".SSH", "authorized_keys")
        assert is_write_denied(target) is True

    def test_real_env_variant_read_blocked(self):
        assert get_read_block_error("/tmp/project/.ENV") is not None

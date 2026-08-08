"""Tests for dashboard fd-leak fix (#81547).

Verifies:
1. ``_raise_fd_soft_limit`` raises the soft fd limit on startup
2. All ``Path.iterdir()`` calls in dashboard hot paths have been replaced
   with context-managed ``os.scandir()`` to prevent directory fd leaks
3. The profiles listing endpoint uses scandir, not iterdir
"""

import os
import sys
from pathlib import Path
from unittest import mock

import pytest


# ─── _raise_fd_soft_limit ──────────────────────────────────────────────


def test_raise_fd_soft_limit_importable():
    """The helper is importable from web_server."""
    from hermes_cli.web_server import _raise_fd_soft_limit
    assert callable(_raise_fd_soft_limit)


def test_raise_fd_soft_limit_noop_on_windows(monkeypatch):
    """On Windows (no resource module), the function silently returns."""
    # Simulate Windows by making the import fail
    monkeypatch.setitem(sys.modules, "resource", None)
    from hermes_cli.web_server import _raise_fd_soft_limit
    # Should not raise — just return
    _raise_fd_soft_limit()


def test_raise_fd_soft_limit_raises_when_low(monkeypatch):
    """When soft < target, the function calls setrlimit to raise it."""
    # Only test on platforms with the resource module
    resource = pytest.importorskip("resource")

    calls = []

    def fake_getrlimit(which):
        return (256, 65536)  # soft=256, hard=65536

    def fake_setrlimit(which, limits):
        calls.append((which, limits))

    monkeypatch.setattr(resource, "getrlimit", fake_getrlimit)
    monkeypatch.setattr(resource, "setrlimit", fake_setrlimit)

    from hermes_cli.web_server import _raise_fd_soft_limit
    _raise_fd_soft_limit()

    assert len(calls) == 1
    which, (soft, hard) = calls[0]
    assert which == resource.RLIMIT_NOFILE
    assert soft >= 4096  # raised to at least the minimum
    assert hard == 65536  # hard limit unchanged


def test_raise_fd_soft_limit_noop_when_already_high(monkeypatch):
    """When soft limit is already >= target, no setrlimit call is made."""
    resource = pytest.importorskip("resource")

    calls = []

    def fake_getrlimit(which):
        return (8192, 65536)  # already high

    def fake_setrlimit(which, limits):
        calls.append((which, limits))

    monkeypatch.setattr(resource, "getrlimit", fake_getrlimit)
    monkeypatch.setattr(resource, "setrlimit", fake_setrlimit)

    from hermes_cli.web_server import _raise_fd_soft_limit
    _raise_fd_soft_limit()

    assert len(calls) == 0  # no change needed


def test_raise_fd_soft_limit_handles_setrlimit_failure(monkeypatch):
    """If setrlimit raises, the function logs a warning but does not crash."""
    resource = pytest.importorskip("resource")

    def fake_getrlimit(which):
        return (256, 65536)

    def fake_setrlimit(which, limits):
        raise OSError("permission denied")

    monkeypatch.setattr(resource, "getrlimit", fake_getrlimit)
    monkeypatch.setattr(resource, "setrlimit", fake_setrlimit)

    from hermes_cli.web_server import _raise_fd_soft_limit
    # Should not raise
    _raise_fd_soft_limit()


# ─── iterdir → scandir replacement ──────────────────────────────────────


def test_no_iterdir_in_fallback_profile_dicts():
    """_fallback_profile_dicts must use os.scandir, not Path.iterdir."""
    import inspect
    from hermes_cli.web_server import _fallback_profile_dicts
    source = inspect.getsource(_fallback_profile_dicts)
    assert ".iterdir()" not in source, (
        "_fallback_profile_dicts still uses Path.iterdir() — "
        "replace with context-managed os.scandir() to prevent fd leaks (#81547)"
    )
    assert "os.scandir" in source, (
        "_fallback_profile_dicts should use os.scandir for fd-safe directory listing"
    )


def test_no_iterdir_in_file_manager_list():
    """The /api/fs/managed endpoint must use os.scandir, not Path.iterdir."""
    import inspect
    from hermes_cli.web_server import _managed_file_entry
    # Find the calling function — look for the list endpoint pattern
    from hermes_cli import web_server
    source = inspect.getsource(web_server)
    # The file manager list endpoint is the one with _managed_file_entry + iterdir
    # Check that the specific pattern "for child in target.iterdir()" is gone
    assert "target.iterdir()" not in source, (
        "File manager list endpoint still uses target.iterdir() — "
        "replace with os.scandir context manager (#81547)"
    )


def test_no_iterdir_in_checkpoint_listing():
    """The checkpoints listing must use os.scandir, not Path.iterdir."""
    import inspect
    from hermes_cli import web_server
    source = inspect.getsource(web_server)
    # The checkpoint listing has a distinctive pattern
    assert "cp_dir.iterdir()" not in source, (
        "Checkpoint listing still uses cp_dir.iterdir() — "
        "replace with os.scandir context manager (#81547)"
    )


def test_no_iterdir_in_plugin_discovery():
    """The plugin discovery must use os.scandir, not Path.iterdir."""
    import inspect
    from hermes_cli import web_server
    source = inspect.getsource(web_server)
    assert "plugins_root.iterdir()" not in source, (
        "Plugin discovery still uses plugins_root.iterdir() — "
        "replace with os.scandir context manager (#81547)"
    )
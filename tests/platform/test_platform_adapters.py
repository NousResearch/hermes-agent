"""
Tests for Universal Platform Adapters.
"""

import pytest
from hermes_platform.factory import get_platform_adapter, detect_platform_type
from hermes_platform.common.adapter import PlatformType, PlatformAdapter
from hermes_platform.linux.adapter import LinuxAdapter
from hermes_platform.windows.adapter import WindowsAdapter


def test_platform_detection_and_adapter():
    ptype = detect_platform_type()
    assert isinstance(ptype, PlatformType)

    adapter = get_platform_adapter()
    assert isinstance(adapter, PlatformAdapter)
    assert adapter.filesystem is not None
    assert adapter.terminal is not None
    assert adapter.process is not None
    assert adapter.network is not None


def test_filesystem_adapter_operations(tmp_path):
    adapter = LinuxAdapter()
    test_file = str(tmp_path / "hello.txt")

    # Write
    written = adapter.filesystem.write(test_file, "Hello Hermes")
    assert written is True

    # Read
    content = adapter.filesystem.read(test_file)
    assert content == "Hello Hermes"

    # List
    items = adapter.filesystem.list(str(tmp_path))
    assert "hello.txt" in items

    # Delete
    deleted = adapter.filesystem.delete(test_file)
    assert deleted is True

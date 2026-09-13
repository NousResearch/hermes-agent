"""Native macOS shutdown diagnostic contract."""
import os
import time

import pytest

from gateway.shutdown_forensics import spawn_async_diagnostic


@pytest.mark.macos_only
def test_native_diagnostic_without_coreutils(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    path = tmp_path / "diagnostic.log"
    started = time.monotonic()
    pid = spawn_async_diagnostic(path, "SIGTERM", timeout_seconds=2)
    assert pid is not None, "stock macOS must not require GNU timeout"
    assert time.monotonic() - started < 2
    deadline = time.monotonic() + 5
    contents = ""
    while time.monotonic() < deadline:
        contents = path.read_text(encoding="utf-8")
        if "=== end ===" in contents:
            break
        time.sleep(0.05)
    os.waitpid(pid, 0)
    assert "=== end ===" in contents
    assert "SIGTERM" in contents
    assert str(os.getpid()) in contents
    assert "load averages:" in contents

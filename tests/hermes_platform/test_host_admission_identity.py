"""Native Linux admission must not guess PID identity without namespace proof."""

from pathlib import Path

import pytest

from hermes_platform.host import identity

pytestmark = pytest.mark.platforms("linux")


def test_unavailable_pid_namespace_cannot_produce_a_partial_identity(monkeypatch):
    real_stat = Path.stat

    def unavailable(path, *args, **kwargs):
        if path == Path("/proc/self/ns/pid"):
            raise FileNotFoundError("namespace unavailable")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", unavailable)
    with pytest.raises(OSError):
        identity.execution_namespace()

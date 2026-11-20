"""The Linux snapshot releases /proc status before returning."""

import builtins
import os

import pytest


@pytest.mark.platforms("linux")
def test_snapshot_closes_status_file(tmp_path, monkeypatch):
    from evals import fanout_resource_bench as bench

    handles = []
    real_open = builtins.open

    def tracked_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(bench, "open", tracked_open, raising=False)
    result = bench._snap(os.getpid(), str(tmp_path / "missing.db"))

    assert handles
    assert all(handle.closed for handle in handles)
    assert result["threads"] > 0
    assert result["fds"] >= 0

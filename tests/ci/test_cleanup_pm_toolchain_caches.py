"""Smoke cache cleanup is bound to completed PM runs and exact key namespaces."""
from __future__ import annotations

import pytest

from scripts.ci.cleanup_pm_toolchain_caches import cleanup_run_caches


def test_cleanup_collects_all_pages_then_deletes_only_its_run():
    rows = [
        {"id": 1, "key": "setup-pm-tools-x64-smoke-42-1"},
        {"id": 2, "key": "node-cache-Windows-x64-smoke-42-2"},
        {"id": 3, "key": "setup-pm-uv-v2-smoke-prune-42-1-linux-x64-os-python-true-lock"},
        {"id": 4, "key": "setup-pm-tools-x64-smoke-consumers-42-1"},
        {"id": 5, "key": "setup-pm-tools-x64-smoke-420-1"},
        {"id": 6, "key": "node-cache-Windows-x64-normal"},
        {"id": 7, "key": "payload-signatures-smoke-42-1"},
    ]
    removed = []
    pages_read = []

    def request(method, path):
        if path == "actions/runs/42":
            return {"status": "completed", "path": ".github/workflows/pm-toolchain.yml"}
        if method == "DELETE":
            assert pages_read[:4] == [1, 2, 3, 4]
            removed.append(int(path.rsplit("/", 1)[-1]))
            rows[:] = [row for row in rows if row["id"] != removed[-1]]
            return None
        page = int(path.rsplit("page=", 1)[-1])
        pages_read.append(page)
        return {"total_count": len(rows), "actions_caches": rows[(page - 1) * 2:page * 2]}

    assert cleanup_run_caches("42", request, page_size=2) == [1, 2, 3, 4]
    assert removed == [1, 2, 3, 4]
    assert {row["id"] for row in rows} == {5, 6, 7}


@pytest.mark.parametrize("status,path", [
    ("in_progress", ".github/workflows/pm-toolchain.yml"),
    ("completed", ".github/workflows/desktop-bundled-release.yml"),
])
def test_cleanup_refuses_active_or_unrelated_runs(status, path):
    calls = []

    def request(method, route):
        calls.append((method, route))
        return {"status": status, "path": path}

    with pytest.raises(ValueError, match="completed PM Toolchain run"):
        cleanup_run_caches("42", request)
    assert calls == [("GET", "actions/runs/42")]

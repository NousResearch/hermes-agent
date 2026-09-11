"""Tests for the synchronous LSPService wrapper.

Drives the service through ``snapshot_baseline`` →
``get_diagnostics_sync`` against the mock LSP server, exercising the
delta filter that ``tools/file_operations._check_lint_delta`` relies
on.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    ServerDef,
    SpawnSpec,
)


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _install_mock_server(
    monkeypatch, script: str | list[str] = "errors", server_id: str = "pyright"
):
    """Replace one registered server with a wrapper that spawns the mock.

    We reuse ``pyright`` so .py files route to it.  This keeps the
    test free of any LSP toolchain dependency.
    """
    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == server_id)
    original = SERVERS[target_index]
    scripts = [script] if isinstance(script, str) else script
    spawn_count = {"value": 0}

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        index = min(spawn_count["value"], len(scripts) - 1)
        spawn_count["value"] += 1
        env = {"MOCK_LSP_SCRIPT": scripts[index]}
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env=env,
            initialization_options={},
        )

    replacement = ServerDef(
        server_id=server_id,
        extensions=original.extensions,
        resolve_root=lambda fp, ws: ws,  # always use workspace root
        build_spawn=_spawn,
        seed_first_push=False,
        description="mock " + server_id,
    )
    # Patch the SERVERS list element directly + restore on teardown.
    SERVERS[target_index] = replacement

    yield spawn_count

    SERVERS[target_index] = original


@pytest.fixture
def mock_pyright(monkeypatch, tmp_path):
    """Install the mock as ``pyright`` and create a fake git workspace."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")  # so pyright's root resolver finds it
    monkeypatch.chdir(str(repo))
    gen = _install_mock_server(monkeypatch, "errors", "pyright")
    next(gen)
    yield repo
    try:
        next(gen)
    except StopIteration:
        pass


@pytest.fixture
def mock_pyright_silent(monkeypatch, tmp_path):
    """Install the silent mock as ``pyright`` (never pushes diagnostics).

    The silent server accepts the open but never publishes diagnostics
    for the pre-edit content and rejects the pull channel, so the
    baseline snapshot has to wait out its full budget — exactly the
    slow-server shape from the wait_timeout report.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")
    monkeypatch.chdir(str(repo))
    gen = _install_mock_server(monkeypatch, "silent", "pyright")
    next(gen)
    yield repo
    try:
        next(gen)
    except StopIteration:
        pass


def test_snapshot_baseline_honors_wait_timeout(mock_pyright_silent):
    """``snapshot_baseline`` must wait at most ``lsp.wait_timeout``, not
    the hardcoded client fallback of 5s.

    Regression for the report that a 2s wait_timeout was ignored by the
    baseline path: the wait ran without a timeout and fell back to
    ``DIAGNOSTICS_DOCUMENT_WAIT`` (5s).  The silent mock never pushes,
    so the elapsed time directly exposes the effective wait budget.
    """
    repo = mock_pyright_silent
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    try:
        start = time.monotonic()
        svc.snapshot_baseline(str(f))
        elapsed = time.monotonic() - start

        # The wait is deadline-based: it always runs the full budget
        # (never less than wait_timeout) and the server never pushes,
        # so both bounds are stable under load.
        assert elapsed >= 1.5, f"baseline returned before the wait budget: {elapsed:.2f}s"
        assert elapsed < 4.5, (
            f"baseline ignored wait_timeout=2.0 and ran the 5s fallback: {elapsed:.2f}s"
        )
        assert svc.get_status()["broken"] == []
        # No fresh data pre-edit -> empty (never stale) baseline.
        assert svc._delta_baseline[os.path.abspath(str(f))] == []
    finally:
        svc.shutdown()


def test_snapshot_baseline_scales_join_budget_past_8s(mock_pyright_silent):
    """The outer join budget must scale with ``wait_timeout`` instead of
    capping at 8s.

    With wait_timeout=10.0 the inner wait takes 10s; an 8s outer cap
    would fire first, falsely mark the server broken, and truncate the
    snapshot.  The elapsed time must land between the inner budget and
    the (now scaled) outer budget, with the server still healthy.
    """
    repo = mock_pyright_silent
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=10.0,
        install_strategy="manual",
    )
    try:
        start = time.monotonic()
        svc.snapshot_baseline(str(f))
        elapsed = time.monotonic() - start

        assert elapsed >= 9.5, f"baseline cut short by the outer cap: {elapsed:.2f}s"
        assert elapsed < 12.5, f"baseline overran the scaled join budget: {elapsed:.2f}s"
        # A slow-but-alive server must not be marked broken.
        assert svc.get_status()["broken"] == []
    finally:
        svc.shutdown()






def test_service_e2e_delta_filter(mock_pyright):
    """End-to-end: snapshot baseline → wait → delta returned."""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        assert svc.enabled_for(str(f))
        # Baseline first — server pushes 1 error.
        svc.snapshot_baseline(str(f))
        # Re-poll: same error is in baseline, so delta is empty.
        new_diags = svc.get_diagnostics_sync(str(f))
        assert new_diags == []
    finally:
        svc.shutdown()


@pytest.mark.parametrize("failed_script", ["clean_eof", "malformed_frame"])
def test_service_replaces_client_after_reader_failure(
    tmp_path, monkeypatch, failed_script
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")
    source = repo / "x.py"
    source.write_text("print('hi')\n")
    monkeypatch.chdir(str(repo))
    server = _install_mock_server(
        monkeypatch, [failed_script, "clean"], "pyright"
    )
    spawn_count = next(server)

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=0.5,
        install_strategy="manual",
    )
    try:
        async def _break_first_client():
            client = await svc._get_or_spawn(str(source))
            assert client is not None
            reader_task = client._reader_task
            assert reader_task is not None
            await client.open_file(str(source), language_id="python")
            await asyncio.wait_for(asyncio.shield(reader_task), timeout=3.0)
            return client

        first = svc._loop.run(_break_first_client(), timeout=5.0)
        replacement = svc._loop.run(svc._get_or_spawn(str(source)), timeout=5.0)

        assert not first.is_running
        assert replacement is not None
        assert replacement is not first
        assert replacement.is_running
        assert spawn_count["value"] == 2
    finally:
        svc.shutdown()
        try:
            next(server)
        except StopIteration:
            pass


def test_service_e2e_delta_filter_with_line_shift(mock_pyright):
    """End-to-end: an edit that shifts the diagnostic's line still
    filters correctly when ``line_shift`` is supplied.

    The mock LSP server emits a fixed error at line 0; for this test
    we don't need to actually shift the server's output — we just
    need to prove that supplying a line_shift through the API works
    and doesn't break the existing delta path.  The unit tests in
    test_delta_key.py cover the shift semantics in detail.
    """
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        svc.snapshot_baseline(str(f))
        # Identity shift — should behave exactly like no shift.
        new_diags = svc.get_diagnostics_sync(str(f), line_shift=lambda L: L)
        assert new_diags == []
    finally:
        svc.shutdown()






def test_reused_client_refreshes_last_used_and_survives_reap(mock_pyright):
    """A client re-acquired from the cache must have its ``_last_used``
    timestamp refreshed so a subsequent sweep does NOT evict it.

    Covers the timestamp refresh on the existing-client fast path in
    ``_get_or_spawn`` — without it, a client in constant use would be
    reaped ``idle_timeout`` seconds after its FIRST use.
    """
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=60.0,  # sweeps manually below; loop never fires
    )
    try:
        svc.get_diagnostics_sync(str(f))
        key = next(iter(svc._clients))
        first_used = svc._last_used[key]

        # Age the timestamp past the cutoff, then re-acquire the client.
        svc._last_used[key] = first_used - 120.0
        svc.get_diagnostics_sync(str(f))
        assert svc._last_used[key] > first_used - 120.0, (
            "re-acquiring a cached client must refresh _last_used"
        )

        # A sweep right after reuse must keep the client.
        svc._loop.run(svc._reap_idle_once(), timeout=5.0)
        assert key in svc._clients
        assert svc.get_status()["clients"]
    finally:
        svc.shutdown()


def test_reaper_survives_sweep_error(mock_pyright):
    """One failing sweep must not kill the reaper loop — the loop's
    ``except Exception`` guard must swallow the error and keep sweeping."""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0.1,
    )
    try:
        # Sabotage the sweep itself so the reaper-loop except branch
        # actually runs (a failing client.shutdown() would be swallowed
        # by gather(return_exceptions=True) and never reach the loop).
        calls = {"n": 0}
        real_reap = svc._reap_idle_once

        async def _flaky_reap():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("sweep sabotage")
            await real_reap()

        svc._reap_idle_once = _flaky_reap  # type: ignore[method-assign]

        svc.get_diagnostics_sync(str(f))
        assert svc.get_status()["clients"]

        # First sweep raises; later sweeps must still reap the client.
        deadline = time.monotonic() + 3.0
        while svc.get_status()["clients"] and time.monotonic() < deadline:
            time.sleep(0.02)

        assert calls["n"] >= 2, "reaper loop died after the failing sweep"
        assert svc.get_status()["clients"] == []
        assert svc._idle_reaper_task is not None
        assert not svc._idle_reaper_task.done()
    finally:
        svc.shutdown()









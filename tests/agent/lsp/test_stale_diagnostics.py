"""Regression tests for the "ghost diagnostics" staleness bug.

Scenario: the agent edits a TypeScript file, tsserver takes a long
time to re-check it, and the old diagnostics (for the PRE-edit
content) were reported as if they were current — the agent then
chases errors it already fixed.

The contract under test:

- ``wait_for_diagnostics`` must NOT be satisfied by diagnostics left
  over from a previous edit cycle; it returns True only when fresh
  (post-didChange) data arrived, False on timeout.
- ``diagnostics_for(fresh_only=True)`` must exclude stale stores.
- ``LSPService.get_diagnostics_sync`` must return [] ("no data")
  rather than the stale diagnostics when the server never re-checks
  within the wait budget, and must NOT mark the server broken.
- A slow-but-eventually-correct server ("slow_push") is waited on,
  honouring the configured ``lsp.wait_timeout``.
"""
from __future__ import annotations

import os
import logging
import sys
import threading
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _client(workspace: Path, script: str, **env_extra: str) -> LSPClient:
    env = {
        "MOCK_LSP_SCRIPT": script,
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        **env_extra,
    }
    return LSPClient(
        server_id=f"mock-{script}",
        workspace_root=str(workspace),
        command=[sys.executable, MOCK_SERVER],
        env=env,
        cwd=str(workspace),
    )






@pytest.mark.asyncio
async def test_slow_push_is_waited_for(tmp_path: Path):
    """A server that re-checks slowly (but within budget) gets waited on,
    and the fresh (clean) result replaces the old error."""
    f = tmp_path / "x.py"
    f.write_text("bad code\n")

    client = _client(tmp_path, "slow_push", MOCK_LSP_PUSH_DELAY="0.8")
    await client.start()
    try:
        v0 = await client.open_file(str(f), language_id="python")
        assert await client.wait_for_diagnostics(str(f), v0, mode="document", timeout=2.0)
        assert len(client.diagnostics_for(str(f), fresh_only=True)) == 1

        f.write_text("good code\n")
        v1 = await client.open_file(str(f), language_id="python")
        fresh = await client.wait_for_diagnostics(str(f), v1, mode="document", timeout=5.0)
        assert fresh is True, "slow push within budget must satisfy the wait"
        assert client.diagnostics_for(str(f), fresh_only=True) == []
    finally:
        await client.shutdown()






# ---------------------------------------------------------------------------
# Service-level: stale data must surface as "no data", never as errors
# ---------------------------------------------------------------------------


def _install_mock_server(script: str, server_id: str = "pyright"):
    """Replace one registered server with a wrapper spawning the mock.

    Mirrors the helper in test_service.py — reuse pyright so .py files
    route to the mock without a real toolchain.
    """
    from agent.lsp.servers import SERVERS, ServerContext, ServerDef, SpawnSpec

    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == server_id)
    original = SERVERS[target_index]

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env={"MOCK_LSP_SCRIPT": script},
            initialization_options={},
        )

    SERVERS[target_index] = ServerDef(
        server_id=server_id,
        extensions=original.extensions,
        resolve_root=lambda fp, ws: ws,
        build_spawn=_spawn,
        seed_first_push=False,
        description="mock " + server_id,
    )
    return target_index, original


@pytest.fixture
def stale_repo(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")
    monkeypatch.chdir(str(repo))
    idx, original = _install_mock_server("stale")
    yield repo
    from agent.lsp.servers import SERVERS

    SERVERS[idx] = original


def test_service_reports_no_data_not_stale_errors(stale_repo):
    """When the server never re-checks the edited content in budget,
    get_diagnostics_sync must return [] and keep the server usable."""
    from agent.lsp.manager import LSPService

    f = stale_repo / "x.py"
    f.write_text("bad code\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
    )
    try:
        # First contact: didOpen gets the (real) pre-edit error push.
        first = svc.get_diagnostics_sync(str(f), delta=False)
        assert len(first) == 1

        # Edit the file — mock never re-publishes (slow tsserver model).
        f.write_text("good code\n")
        ghost = svc.get_diagnostics_sync(str(f), delta=False)
        assert ghost == [], "stale pre-edit error must not be reported as current"

        # Not marked broken: slow is not dead.
        assert svc.enabled_for(str(f))
        status = svc.get_status()
        assert status["broken"] == []
    finally:
        svc.shutdown()


def test_diagnostics_timeout_skips_later_edit_waits(stale_repo, caplog):
    """One fresh-diagnostics timeout degrades only waiting; later edit hooks must stay non-blocking."""
    from agent.lsp.manager import LSPService

    f = stale_repo / "x.py"
    f.write_text("bad code\n")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=0.1,
        install_strategy="manual",
    )
    try:
        assert len(svc.get_diagnostics_sync(str(f), delta=False)) == 1

        f.write_text("first fix\n")
        assert svc.get_diagnostics_sync(str(f), delta=False) == []

        # The cooldown is scoped to this server/workspace, not every pyright client.
        other_repo = stale_repo.parent / "other-repo"
        other_repo.mkdir()
        (other_repo / ".git").mkdir()
        (other_repo / "pyproject.toml").write_text("")
        other = other_repo / "x.py"
        other.write_text("bad code\n")
        assert len(svc.get_diagnostics_sync(str(other), delta=False)) == 1

        client = svc._clients[("pyright", str(stale_repo))]
        doc = client._docs[os.path.abspath(f)]
        timed_out_version = doc.version
        wait_calls = 0

        async def count_waits(*args, **kwargs):
            nonlocal wait_calls
            wait_calls += 1
            version = args[1]
            doc.push = []
            doc.push_version = version
            return True

        client.wait_for_diagnostics = count_waits
        caplog.set_level(logging.DEBUG, logger="hermes.lint.lsp")
        caplog.clear()
        svc.snapshot_baseline(str(f))
        f.write_text("second fix\n")
        assert svc.get_diagnostics_sync(str(f), delta=False) == []

        assert wait_calls == 0
        assert doc.version == timed_out_version
        messages = [record.getMessage() for record in caplog.records]
        assert any("skipped diagnostics wait during timeout cooldown" in message for message in messages)
        assert not any(" clean (" in message for message in messages)
        assert svc.enabled_for(str(f))
        assert svc.get_status()["broken"] == []

        # Expiring the bounded cooldown permits one normal probe; a fresh verdict recovers the pair.
        svc._diagnostics_degraded_until = {
            key: 0.0 for key in svc._diagnostics_degraded_until
        }
        assert svc.get_diagnostics_sync(str(f), delta=False) == []
        assert wait_calls == 1
        assert doc.version == timed_out_version + 1
        assert svc._diagnostics_degraded_until == {}
    finally:
        svc.shutdown()


def test_expired_cooldown_allows_exactly_one_concurrent_probe(stale_repo):
    """The half-open transition is atomic: peers keep skipping while one caller probes."""
    from agent.lsp.manager import LSPService

    f = stale_repo / "x.py"
    f.write_text("good code\n")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=0.1,
        install_strategy="manual",
    )
    try:
        svc._degrade_diagnostics(str(f))
        with svc._state_lock:
            key = next(iter(svc._diagnostics_degraded_until))
            svc._diagnostics_degraded_until[key] = 0.0

        barrier = threading.Barrier(8)
        skipped = []

        def attempt_probe():
            barrier.wait()
            skipped.append(svc._skip_degraded_diagnostics(str(f)))

        threads = [threading.Thread(target=attempt_probe) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert skipped.count(False) == 1
        assert skipped.count(True) == 7
        assert svc._diagnostics_probe_inflight == {key}

        svc._mark_broken_for_file(str(f), RuntimeError("transport failed"))
        assert key not in svc._diagnostics_degraded_until
        assert key not in svc._diagnostics_probe_inflight
    finally:
        svc.shutdown()


@pytest.mark.parametrize("delta", [True, False])
def test_recovery_probe_after_skipped_baseline_preserves_mode(stale_repo, delta):
    """Recovery suppresses unsafe deltas but preserves requested full diagnostics."""
    import agent.lsp.manager as manager

    f = stale_repo / "x.py"
    f.write_text("bad code\n")
    existing = {
        "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}},
        "severity": 1,
        "message": "pre-existing error",
    }
    svc = manager.LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=0.1,
        install_strategy="manual",
    )
    try:
        svc._degrade_diagnostics(str(f))
        svc.snapshot_baseline(str(f))
        abs_path = os.path.abspath(f)
        assert abs_path in svc._skipped_delta_baselines

        with svc._state_lock:
            key = next(iter(svc._diagnostics_degraded_until))
            svc._diagnostics_degraded_until[key] = 0.0

        async def fresh_verdict(*args, **kwargs):
            return [existing]

        svc._open_and_wait_async = fresh_verdict
        observed = svc.get_diagnostics_sync(str(f), delta=delta)
        assert observed == ([] if delta else [existing])
        if delta:
            assert svc._delta_baseline[abs_path] == [existing]
        assert abs_path not in svc._skipped_delta_baselines
        assert svc._diagnostics_degraded_until == {}
        assert svc._diagnostics_probe_inflight == set()
    finally:
        svc.shutdown()


def test_skipped_baseline_survives_skips_and_retimeouts_until_fresh_verdict(stale_repo, monkeypatch):
    """Unknown baseline provenance persists across a whole degraded edit burst."""
    import agent.lsp.manager as manager

    f = stale_repo / "x.py"
    f.write_text("bad code\n")
    existing = {
        "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 3}},
        "severity": 1,
        "message": "pre-existing error",
    }
    now = [100.0]
    monkeypatch.setattr(manager.time, "monotonic", lambda: now[0])
    svc = manager.LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=0.1,
        install_strategy="manual",
    )
    try:
        svc._degrade_diagnostics(str(f))
        svc.snapshot_baseline(str(f))
        abs_path = os.path.abspath(f)
        assert abs_path in svc._skipped_delta_baselines

        # The paired post-edit call also skips; provenance must remain unknown.
        assert svc.get_diagnostics_sync(str(f), delta=True) == []
        assert abs_path in svc._skipped_delta_baselines

        # The first half-open probe gets no verdict and reopens the cooldown.
        now[0] = 131.0

        async def no_verdict(*args, **kwargs):
            return None

        svc._open_and_wait_async = no_verdict
        assert svc.get_diagnostics_sync(str(f), delta=True) == []
        assert abs_path in svc._skipped_delta_baselines

        # A later fresh verdict reseeds the baseline without false attribution.
        now[0] = 162.0

        async def fresh_verdict(*args, **kwargs):
            return [existing]

        svc._open_and_wait_async = fresh_verdict
        assert svc.get_diagnostics_sync(str(f), delta=True) == []
        assert svc._delta_baseline[abs_path] == [existing]
        assert abs_path not in svc._skipped_delta_baselines
    finally:
        svc.shutdown()


def test_unavailable_client_does_not_open_timeout_cooldown(stale_repo, caplog):
    """No client is unavailable, not a live-server freshness timeout."""
    import agent.lsp.manager as manager

    f = stale_repo / "x.py"
    f.write_text("good code\n")
    svc = manager.LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=0.1,
        install_strategy="manual",
    )
    try:
        async def unavailable(*args, **kwargs):
            return manager._LSP_UNAVAILABLE

        svc._open_and_wait_async = unavailable
        caplog.set_level(logging.DEBUG, logger="hermes.lint.lsp")
        svc.snapshot_baseline(str(f))

        assert svc._diagnostics_degraded_until == {}
        assert not any("timed out" in record.getMessage() for record in caplog.records)
    finally:
        svc.shutdown()

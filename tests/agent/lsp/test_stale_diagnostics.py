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
import sys
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
async def test_stale_push_does_not_satisfy_wait(tmp_path: Path):
    """A push from the previous edit cycle must not end the wait early.

    The 'stale' mock publishes an error for the original content and
    then goes silent — the wait after the edit must time out (False),
    not return instantly on the leftover push.
    """
    f = tmp_path / "x.py"
    f.write_text("bad code\n")

    client = _client(tmp_path, "stale")
    await client.start()
    try:
        v0 = await client.open_file(str(f), language_id="python")
        assert await client.wait_for_diagnostics(str(f), v0, mode="document", timeout=2.0)
        assert len(client.diagnostics_for(str(f))) == 1  # pre-edit error is real

        # Fix the file.  The stale server never re-checks.
        f.write_text("good code\n")
        v1 = await client.open_file(str(f), language_id="python")
        fresh = await client.wait_for_diagnostics(str(f), v1, mode="document", timeout=1.0)
        assert fresh is False, "wait must not be satisfied by pre-edit leftovers"
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_fresh_only_excludes_stale_stores(tmp_path: Path):
    f = tmp_path / "x.py"
    f.write_text("bad code\n")

    client = _client(tmp_path, "stale")
    await client.start()
    try:
        v0 = await client.open_file(str(f), language_id="python")
        await client.wait_for_diagnostics(str(f), v0, mode="document", timeout=2.0)

        f.write_text("good code\n")
        await client.open_file(str(f), language_id="python")
        # Merged legacy view still exposes the leftover push...
        assert len(client.diagnostics_for(str(f))) == 1
        # ...but the fresh-only view correctly reports no verdict yet.
        assert client.diagnostics_for(str(f), fresh_only=True) == []
    finally:
        await client.shutdown()


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


@pytest.mark.asyncio
async def test_wait_timeout_param_overrides_mode_budget(tmp_path: Path):
    """The explicit timeout must control the wait budget (config plumb)."""
    import asyncio

    f = tmp_path / "x.py"
    f.write_text("bad code\n")

    client = _client(tmp_path, "stale")
    await client.start()
    try:
        v0 = await client.open_file(str(f), language_id="python")
        await client.wait_for_diagnostics(str(f), v0, mode="document", timeout=2.0)
        f.write_text("good code\n")
        v1 = await client.open_file(str(f), language_id="python")

        loop = asyncio.get_event_loop()
        start = loop.time()
        fresh = await client.wait_for_diagnostics(str(f), v1, mode="document", timeout=0.5)
        elapsed = loop.time() - start
        assert fresh is False
        # Must respect ~0.5s, not the 5s document default.
        assert elapsed < 3.0
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_stale_pull_result_dropped_when_change_races(tmp_path: Path):
    """A pull answered for pre-edit content must not read as fresh after
    a didChange raced past it (version-tag anchoring)."""
    f = tmp_path / "x.py"
    f.write_text("bad code\n")

    client = _client(tmp_path, "clean")
    await client.start()
    try:
        v0 = await client.open_file(str(f), language_id="python")
        await client.wait_for_diagnostics(str(f), v0, mode="document", timeout=2.0)
        doc = client._docs[os.path.abspath(str(f))]
        assert doc.fresh_pull()

        # Simulate an edit racing in: the version bump invalidates the
        # stored pull without any explicit clearing.
        f.write_text("good code\n")
        await client.open_file(str(f), language_id="python")
        assert not doc.fresh_pull()
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

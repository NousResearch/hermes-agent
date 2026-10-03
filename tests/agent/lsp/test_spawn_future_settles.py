"""A raising/cancelled spawn must settle the shared future and mark broken.

``_get_or_spawn`` registers one ``Future`` in ``_spawning`` and every
concurrent caller for the same (server, root) parks on it.  Only the
success path called ``set_result``; any other exit (``build_spawn``
raising, or the outer ``_loop.run`` timeout cancelling the in-flight
``await client.start()``) popped the key in ``finally`` while leaving
the future unsettled, so waiters blocked until their own budget
expired and the pair was never added to ``_broken``.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

from agent.lsp import manager as manager_mod
from agent.lsp.manager import LSPService
from agent.lsp.servers import SERVERS, ServerContext, ServerDef, SpawnSpec
from agent.lsp.workspace import clear_cache


def _make_git_workspace(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='t'\n")
    return repo


def _install_spec_server(monkeypatch, build_spawn, server_id="pyright"):
    """Swap pyright's ServerDef for one using ``build_spawn``; restore after."""
    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == server_id)
    original = SERVERS[target_index]
    replacement = ServerDef(
        server_id=server_id,
        extensions=original.extensions,
        resolve_root=lambda fp, ws: ws,
        build_spawn=build_spawn,
        seed_first_push=False,
        description="mock " + server_id,
    )
    SERVERS[target_index] = replacement
    return original, target_index


class _HangingClient:
    """Fake LSPClient whose start() never finishes until cancelled."""

    def __init__(self, *args, **kwargs):
        self.server_id = kwargs.get("server_id", "pyright")
        self.workspace_root = kwargs.get("workspace_root", "")
        self.is_running = True

    async def start(self):
        await asyncio.sleep(3600)

    async def shutdown(self):
        self.is_running = False


def test_cancelled_spawn_settles_shared_future(tmp_path, monkeypatch):
    """Owner cancelled mid-spawn (outer timeout) must release the waiter.

    RED on unfixed main: the waiter stays parked on the orphaned future
    and the bounded wait below times out; the pair is never marked broken.
    """
    clear_cache()
    repo = _make_git_workspace(tmp_path)
    src = repo / "x.py"
    src.write_text("")
    monkeypatch.chdir(str(repo))

    calls = {"n": 0}

    def _ok_spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        calls["n"] += 1
        return SpawnSpec(
            command=["mock-lsp"],
            workspace_root=root,
            cwd=root,
            env={},
            initialization_options={},
        )

    saved = list(SERVERS)
    _install_spec_server(monkeypatch, _ok_spawn)

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    try:
        with patch.object(manager_mod, "LSPClient", _HangingClient):
            async def _scenario():
                t0 = time.monotonic()
                owner = asyncio.ensure_future(svc._get_or_spawn(str(src)))
                await asyncio.sleep(0.5)  # let owner enter start()
                waiter = asyncio.ensure_future(svc._get_or_spawn(str(src)))
                await asyncio.sleep(0.5)  # let waiter park on the future
                assert not waiter.done()  # parked, sharing one spawn
                assert calls["n"] == 1  # single spawn for both callers
                owner.cancel()  # simulate the outer _loop.run timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(owner, waiter, return_exceptions=True),
                        timeout=5.0,
                    )
                finally:
                    for t in (owner, waiter):
                        if not t.done():
                            t.cancel()
                return time.monotonic() - t0, results

            elapsed, results = svc._loop.run(_scenario(), timeout=30.0)
        assert elapsed < 10.0, "waiter was stuck on an unsettled future"
        assert all(isinstance(r, asyncio.CancelledError) for r in results)
        assert ("pyright", str(repo)) in svc._broken
    finally:
        SERVERS[:] = saved
        svc.shutdown()
        clear_cache()


def test_raising_build_spawn_marks_broken_once(tmp_path, monkeypatch):
    """A raising build_spawn must surface once, then negative-cache.

    RED on unfixed main: the exception propagates but the pair is never
    added to ``_broken``, so every call re-pays the spawn attempt.
    """
    clear_cache()
    repo = _make_git_workspace(tmp_path)
    src = repo / "x.py"
    src.write_text("")
    monkeypatch.chdir(str(repo))

    calls = {"n": 0}

    def _boom_spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        calls["n"] += 1
        raise RuntimeError("malformed lsp.servers override")

    saved = list(SERVERS)
    _install_spec_server(monkeypatch, _boom_spawn)

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    try:
        import pytest

        with pytest.raises(RuntimeError, match="malformed"):
            svc._loop.run(svc._get_or_spawn(str(src)), timeout=10.0)
        assert ("pyright", str(repo)) in svc._broken
        # Second call short-circuits on the broken-set: no re-spawn.
        assert svc._loop.run(svc._get_or_spawn(str(src)), timeout=10.0) is None
        assert calls["n"] == 1
    finally:
        SERVERS[:] = saved
        svc.shutdown()
        clear_cache()

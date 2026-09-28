"""Shared spawn completion belongs to the owner, not individual waiters."""
import asyncio
from types import SimpleNamespace

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import ServerDef


@pytest.fixture
def service(tmp_path, monkeypatch):
    root = str(tmp_path)
    srv = ServerDef("pyright", (".py",), lambda *_: root, lambda *_: None)
    svc = LSPService(enabled=False, wait_mode="document", wait_timeout=5,
                     install_strategy="manual", idle_timeout=0)
    monkeypatch.setattr(svc, "_server_for", lambda _: srv)
    monkeypatch.setattr("agent.lsp.manager.resolve_workspace_for_file", lambda _: (root, True))
    return svc, srv, str(tmp_path / "example.py"), root


@pytest.mark.asyncio
async def test_spawn_exception_settles_shared_future(service, monkeypatch):
    svc, srv, path, root = service
    started, release = asyncio.Event(), asyncio.Event()
    async def fail(*_):
        started.set()
        await release.wait()
        raise OSError("spawn fixture failed")
    monkeypatch.setattr(svc, "_spawn_client", fail)
    owner = asyncio.create_task(svc._get_or_spawn(path))
    await asyncio.wait_for(started.wait(), 5)
    shared = next(iter(svc._spawning.values()))
    waiter = asyncio.create_task(svc._get_or_spawn(path))
    await asyncio.sleep(0)
    release.set()
    try:
        await asyncio.gather(owner, return_exceptions=True)
        assert shared.done(), "spawn failure left concurrent callers on an orphan future"
        assert await asyncio.wait_for(waiter, 5) is None
        assert svc._is_broken((srv.server_id, root))
        assert svc._spawning == {}
        assert await svc._get_or_spawn(path) is None
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


def test_build_spawn_failure_uses_public_fallback(tmp_path, monkeypatch, caplog):
    (tmp_path / ".git").mkdir()
    source = tmp_path / "example.py"
    source.write_text("pass\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls = []
    def fail(root, ctx):
        calls.append(ctx.install_strategy)
        raise OSError("cannot resolve fixture binary")
    srv = ServerDef("pyright", (".py",), lambda *_: str(tmp_path), fail)
    svc = LSPService(enabled=True, wait_mode="document", wait_timeout=5,
                     install_strategy="manual", idle_timeout=0, extra_servers=[srv])
    try:
        assert svc.get_diagnostics_sync(str(source)) == []
        assert svc.get_diagnostics_sync(str(source)) == []
        assert calls == ["manual"]
        assert (srv.server_id, str(tmp_path)) in svc.get_status()["broken"]
        assert any("spawn" in r.getMessage() and "cannot resolve fixture binary" in r.getMessage()
                   for r in caplog.records)
    finally:
        svc.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_owner", [False, True])
async def test_cancellation_does_not_strand_other_waiters(service, monkeypatch, cancel_owner):
    svc, srv, path, root = service
    started, release = asyncio.Event(), asyncio.Event()
    client = SimpleNamespace(is_running=True)
    async def spawn(*_):
        started.set()
        await release.wait()
        return client
    monkeypatch.setattr(svc, "_spawn_client", spawn)
    owner = asyncio.create_task(svc._get_or_spawn(path))
    await asyncio.wait_for(started.wait(), 5)
    shared = next(iter(svc._spawning.values()))
    waiters = [asyncio.create_task(svc._get_or_spawn(path)) for _ in range(2)]
    await asyncio.sleep(0)
    cancelled = owner if cancel_owner else waiters[0]
    cancelled.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        if cancel_owner:
            assert shared.done(), "owner cancellation orphaned the shared future"
            assert await asyncio.wait_for(waiters[1], 5) is None
        else:
            assert not shared.cancelled(), "one waiter cancelled the shared spawn"
            release.set()
            assert await asyncio.wait_for(owner, 5) is client
            assert await asyncio.wait_for(waiters[1], 5) is client
        assert svc._spawning == {}
    finally:
        for task in [owner, *waiters]:
            task.cancel()
        await asyncio.gather(owner, *waiters, return_exceptions=True)

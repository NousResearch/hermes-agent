"""A failed spawn must complete the shared result before releasing ownership."""
import asyncio

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
@pytest.mark.parametrize("outcome", ["exception", "unavailable", "cancel"])
async def test_spawn_failure_settles_shared_future(service, monkeypatch, outcome):
    svc, srv, path, root = service
    started, release = asyncio.Event(), asyncio.Event()

    async def fail(*_):
        started.set()
        await release.wait()
        if outcome == "exception":
            raise OSError("spawn fixture failed")
        return None

    monkeypatch.setattr(svc, "_spawn_client", fail)
    owner = asyncio.create_task(svc._get_or_spawn(path))
    await asyncio.wait_for(started.wait(), 5)
    shared = next(iter(svc._spawning.values()))
    waiter = asyncio.create_task(svc._get_or_spawn(path))
    await asyncio.sleep(0)
    if outcome == "cancel":
        owner.cancel()
    else:
        release.set()
    try:
        results = await asyncio.gather(owner, return_exceptions=True)
        if outcome == "cancel":
            assert isinstance(results[0], asyncio.CancelledError)
        else:
            assert results == [None]
        assert shared.done(), "spawn exit left concurrent callers on an orphan future"
        assert await asyncio.wait_for(waiter, 5) is None
        assert svc._spawning == {}
        if outcome != "cancel":
            assert svc._is_broken((srv.server_id, root))
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
        assert any("spawn/initialize failed" in r.getMessage()
                   and "cannot resolve fixture binary" in r.getMessage() for r in caplog.records)
    finally:
        svc.shutdown()

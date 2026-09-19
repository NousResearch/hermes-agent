"""Concurrency invariants for the LSP service state lock."""
from __future__ import annotations

import asyncio

import agent.lsp.manager as manager
from agent.lsp.manager import LSPService
from agent.lsp.servers import ServerDef


def test_reused_multiroot_client_attaches_outside_state_lock(monkeypatch):
    """Awaitable multi-root attachment must never run while ``_state_lock`` is held.

    A synchronous ``threading.Lock`` held across an await can deadlock the single
    LSP event-loop thread as soon as another coroutine tries to acquire that lock.
    """
    service = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=0.1,
        install_strategy="manual",
        idle_timeout=0,
    )
    root = "/repo/worktree-b"
    server = ServerDef(
        server_id="pyright",
        extensions=(".py",),
        resolve_root=lambda _file_path, _workspace_root: root,
        build_spawn=lambda _root, _ctx: None,
        multi_root=True,
    )

    class StubClient:
        is_running = True

        async def add_workspace_folder(self, attached_root: str) -> None:
            assert attached_root == root
            lock_was_free = service._state_lock.acquire(blocking=False)
            if lock_was_free:
                service._state_lock.release()
            assert lock_was_free, (
                "_state_lock must be released before awaiting workspace attachment"
            )

    client = StubClient()
    service._clients[(server.server_id, "")] = client  # multi-root client key
    service._last_used[(server.server_id, "")] = 0.0

    monkeypatch.setattr(manager, "find_server_for_file", lambda _path: server)
    monkeypatch.setattr(
        manager,
        "resolve_workspace_for_file",
        lambda _path: ("/repo", True),
    )
    monkeypatch.setattr(manager.eventlog, "log_active", lambda *_args, **_kwargs: None)

    result = asyncio.run(service._get_or_spawn("/repo/worktree-b/example.py"))

    assert result is client


def test_delta_baseline_is_capped_by_write_recency():
    """Baselines beyond _DELTA_BASELINE_CAP evict the path written longest ago, and rewriting a
    path refreshes it instead of aging it out ahead of paths never touched again."""
    service = LSPService(enabled=False, wait_mode="document", wait_timeout=0.1,
                         install_strategy="manual", idle_timeout=0)
    cap = manager._DELTA_BASELINE_CAP
    for i in range(cap):
        service._set_delta_baseline(f"/repo/f{i}.py", [])
    service._set_delta_baseline("/repo/f0.py", [{"message": "rewritten"}])
    service._set_delta_baseline("/repo/new.py", [])
    assert len(service._delta_baseline) == cap
    assert "/repo/f1.py" not in service._delta_baseline
    assert service._delta_baseline["/repo/f0.py"] == [{"message": "rewritten"}]

"""Tests for the LSP concurrency cap (``lsp.max_clients``).

2026-09-17: a worker resolved two roots and left two pyright servers running
(~45% CPU combined). ``_evict_lru_if_over_cap`` bounds how many language
servers one service keeps alive.
"""
from __future__ import annotations

from agent.lsp.manager import DEFAULT_MAX_CLIENTS, LSPService


class _FakeClient:
    def __init__(self) -> None:
        self.is_running = True
        self.shutdown_calls = 0

    async def shutdown(self) -> None:  # pragma: no cover - trivial
        self.shutdown_calls += 1


def _svc(max_clients: int) -> LSPService:
    return LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="auto",
        idle_timeout=0,
        max_clients=max_clients,
    )


def test_evicts_lru_when_over_cap():
    svc = _svc(1)
    try:
        old, new = _FakeClient(), _FakeClient()
        svc._clients = {("s", "/old"): old, ("s", "/new"): new}
        svc._last_used = {("s", "/old"): 1.0, ("s", "/new"): 2.0}
        svc._evict_lru_if_over_cap()
        assert ("s", "/old") not in svc._clients
        assert ("s", "/new") in svc._clients
        assert old.shutdown_calls == 1
    finally:
        svc.shutdown()


def test_no_evict_when_under_cap():
    svc = _svc(4)
    try:
        c = _FakeClient()
        svc._clients = {("s", "/a"): c}
        svc._last_used = {("s", "/a"): 1.0}
        svc._evict_lru_if_over_cap()
        assert ("s", "/a") in svc._clients
        assert c.shutdown_calls == 0
    finally:
        svc.shutdown()


def test_zero_disables_cap():
    svc = _svc(0)
    try:
        for i in range(5):
            svc._clients[("s", f"/{i}")] = _FakeClient()
            svc._last_used[("s", f"/{i}")] = float(i)
        svc._evict_lru_if_over_cap()
        assert len(svc._clients) == 5
    finally:
        svc.shutdown()


def test_default_max_clients_is_sane():
    assert DEFAULT_MAX_CLIENTS >= 1

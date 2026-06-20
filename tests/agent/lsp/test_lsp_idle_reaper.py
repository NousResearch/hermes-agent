import pytest

from agent.lsp.manager import LSPService


class FakeClient:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.mark.asyncio
async def test_reaps_idle_clients_and_removes_state() -> None:
    service = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="never",
        idle_timeout=10.0,
    )
    stale = FakeClient()
    fresh = FakeClient()
    stale_key = ("pyright", "/tmp/old-worktree")
    fresh_key = ("pyright", "/tmp/fresh-worktree")
    service._clients[stale_key] = stale  # type: ignore[attr-defined]
    service._clients[fresh_key] = fresh  # type: ignore[attr-defined]
    service._last_used[stale_key] = 80.0  # type: ignore[attr-defined]
    service._last_used[fresh_key] = 95.0  # type: ignore[attr-defined]

    await service._reap_idle_clients_async(now=100.0)  # type: ignore[attr-defined]

    assert stale.shutdown_calls == 1
    assert fresh.shutdown_calls == 0
    assert stale_key not in service._clients  # type: ignore[attr-defined]
    assert stale_key not in service._last_used  # type: ignore[attr-defined]
    assert fresh_key in service._clients  # type: ignore[attr-defined]
    assert service._last_used[fresh_key] == 95.0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_zero_idle_timeout_disables_reaper() -> None:
    service = LSPService(
        enabled=False,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="never",
        idle_timeout=0.0,
    )
    client = FakeClient()
    key = ("pyright", "/tmp/old-worktree")
    service._clients[key] = client  # type: ignore[attr-defined]
    service._last_used[key] = 1.0  # type: ignore[attr-defined]

    await service._reap_idle_clients_async(now=10_000.0)  # type: ignore[attr-defined]

    assert client.shutdown_calls == 0
    assert key in service._clients  # type: ignore[attr-defined]

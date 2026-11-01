"""The Discord typing loop must be bounded by ``_typing_max_lifetime_secs``.

Regression guard for the "stuck is typing…" family (#85427 / #101783): the
adapter's typing loop is the only owner that keeps POSTing ``/typing`` every
12s, and it is normally stopped by ``stop_typing()``. When a turn ends without
that call — e.g. a session eviction / turn interrupt that never reaches the
per-turn ``finally`` — the loop used to POST forever. It is now bounded, so an
orphaned loop self-terminates instead of leaving the indicator on indefinitely.
"""

import asyncio
import types

import pytest

import plugins.platforms.discord.adapter as adapter_mod
from plugins.platforms.discord.adapter import DiscordAdapter


class _FakeHTTP:
    def __init__(self) -> None:
        self.calls = 0

    async def request(self, route) -> None:  # noqa: ANN001 - mirrors discord.http
        self.calls += 1


class _FakeClient:
    def __init__(self) -> None:
        self.http = _FakeHTTP()


@pytest.fixture(autouse=True)
def _stub_discord_route(monkeypatch):
    """The adapter builds its route via ``discord.http.Route``. The test env stubs
    ``discord``, so provide a harmless ``Route`` — otherwise the loop exits on an
    error and the bound is never actually exercised (and no POST happens)."""
    monkeypatch.setattr(
        adapter_mod, "discord",
        types.SimpleNamespace(http=types.SimpleNamespace(Route=lambda *a, **k: object())),
    )


def _make_adapter(max_lifetime_secs: float) -> DiscordAdapter:
    # Skip the heavy __init__: only the typing surface is exercised here.
    adapter = object.__new__(DiscordAdapter)
    adapter._client = _FakeClient()  # type: ignore[assignment]
    adapter._typing_tasks = {}
    adapter._typing_max_lifetime_secs = max_lifetime_secs
    return adapter


@pytest.mark.asyncio
async def test_typing_loop_is_bounded_and_cleans_up():
    """Without any stop_typing(), the loop must still self-terminate and clear its slot."""
    adapter = _make_adapter(0.05)
    await adapter.send_typing("chan-bounded")
    assert "chan-bounded" in adapter._typing_tasks

    await asyncio.sleep(0.3)  # well past the 50ms lifetime

    assert "chan-bounded" not in adapter._typing_tasks
    assert adapter._client.http.calls >= 1  # it POSTed, then bounded itself


@pytest.mark.asyncio
async def test_stop_typing_still_cancels_immediately():
    """The normal path (explicit stop) is unaffected by the bound."""
    adapter = _make_adapter(60.0)
    await adapter.send_typing("chan-stop")
    assert "chan-stop" in adapter._typing_tasks

    await adapter.stop_typing("chan-stop")

    assert "chan-stop" not in adapter._typing_tasks

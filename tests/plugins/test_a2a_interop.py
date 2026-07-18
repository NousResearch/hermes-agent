"""Real-socket interoperability coverage for the official A2A SDK path.

Run explicitly with::

    uv run --extra a2a --extra dev pytest -m integration \
        tests/plugins/test_a2a_interop.py
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.a2a import adapter, config, server, setup
from plugins.platforms.a2a.client import NamedPeerClient
from plugins.platforms.a2a.executor import HermesA2AExecutor
from a2a.types.a2a_pb2 import TASK_STATE_COMPLETED


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


async def _finish_despite_cancellation(awaitable) -> asyncio.CancelledError | None:
    """Finish one owned cleanup even if the test task is being cancelled."""
    task = asyncio.create_task(awaitable)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as exc:
        await task
        return exc
    return None


async def test_real_adapter_and_official_client_interoperate(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(adapter, "_current_profile_name", lambda: "default")

    port = _free_loopback_port()
    rpc_url = f"http://127.0.0.1:{port}/a2a"
    production_url = "https://interop.invalid/a2a"
    setup.ensure_a2a_platform_config(public_url=production_url)
    assert config.configured_public_url(production=True) == production_url
    token = setup.add_principal("local-interop", profile="default")
    setup.add_peer("self", url=rpc_url, token=str(token))

    platform_config = PlatformConfig(
        enabled=True,
        extra={
            "host": "127.0.0.1",
            "port": port,
            "public_url": production_url,
            "principals": {
                "local-interop": {
                    "credential_ref": "inbound:local-interop",
                    "profile": "default",
                }
            },
        },
    )

    # Production keeps requiring an HTTPS public URL. This integration test only
    # replaces the adapter's already-validated runtime tuple so it can exercise a
    # real loopback socket without weakening product URL validation.
    monkeypatch.setattr(
        adapter,
        "_runtime_settings",
        lambda _config: ("127.0.0.1", port, rpc_url, "default"),
    )

    instance = adapter.A2AAdapter(platform_config)

    async def message_handler(event):
        return f"reply:{event.text}"

    instance.set_message_handler(message_handler)
    client = NamedPeerClient()
    cancellation = None
    try:
        assert await instance.connect() is True
        assert isinstance(instance._executor, HermesA2AExecutor)

        card = await client.fetch_card("self")
        assert len(card.supported_interfaces) == 1
        assert card.supported_interfaces[0].url == rpc_url

        first, first_text = await client.ask("self", "one")
        second, second_text = await client.ask("self", "two")

        assert first.status.state == TASK_STATE_COMPLETED
        assert second.status.state == TASK_STATE_COMPLETED
        assert first_text == ["reply:one"]
        assert second_text == ["reply:two"]
        assert first.context_id == second.context_id
        assert first.id != second.id

        fetched = await client.get_task("self", first.id)
        assert fetched.id == first.id
        assert fetched.context_id == first.context_id
        assert fetched.status.state == first.status.state

        listed = await client.list_tasks("self")
        listed_ids = {task.id for task in listed.tasks}
        assert {first.id, second.id} <= listed_ids
    finally:
        cancellation = await _finish_despite_cancellation(client.aclose())
        disconnect_cancellation = await _finish_despite_cancellation(instance.disconnect())
        cancellation = cancellation or disconnect_cancellation
        await asyncio.sleep(0)

        assert not client._owned_tasks
        assert not instance.is_connected
        assert instance._server_task is None
        assert instance._monitor_task is None
        assert instance._executor_cleanup_task is None
        assert instance._store_close_task is None
        assert instance._deferred_cleanup_task is None
        assert not {
            task.get_name()
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name().startswith("a2a-")
        }
        if cancellation is not None:
            raise cancellation

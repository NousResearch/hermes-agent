"""Recovery must retain plugin platforms that are not built-in enum iterations."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as ledger
from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.platforms.base import SendResult
from tests.gateway.test_relay_ledger_recovery import record, setup_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["native", "relay"])
async def test_boot_recovers_dynamically_registered_platform(
    tmp_path, monkeypatch, transport
):
    runner, _, connector = setup_runner(tmp_path, monkeypatch)
    name = "recoveryprobe-" + transport
    platform_registry.register(
        PlatformEntry(
            name=name,
            label="Recovery Probe",
            adapter_factory=lambda config: None,
            check_fn=lambda: True,
        )
    )
    try:
        native = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True)))
        if transport == "native":
            runner.adapters = {Platform(name): native}
        else:
            # Do not materialize Platform(name) first: a cold boot only has the
            # connector's advertised strings and the persisted ledger row.
            connector._identities = [(name, "bot-probe")]
        record(logical=name)
        with ledger._connect() as db:
            db.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1"
            )
        assert await runner._redeliver_pending_obligations() == 1
        if transport == "native":
            native.send.assert_awaited_once()
            assert not connector.sent
        else:
            assert connector.sent_platforms == [name]
    finally:
        platform_registry.unregister(name)


@pytest.mark.asyncio
async def test_relay_cannot_recursively_front_its_own_transport(tmp_path, monkeypatch):
    runner, _, connector = setup_runner(tmp_path, monkeypatch)
    connector._identities = [("relay", "invalid-logical-platform")]
    assert await runner._redeliver_failed_obligations_for_platform(Platform.RELAY) == 0

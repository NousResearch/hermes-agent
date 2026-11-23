"""Regression tests for #33711.

The agent slot (``_running_agents``) is released in the turn's ``finally``
(``_run_agent_cleanup_turn_tasks``), but a turn's post-handler delivery keeps
running afterwards: the final text send, auto-TTS playback, MEDIA/local-file
attachments and the streamed-final attachment rescan all upload after the
slot is gone. A ``--replace`` restart drains on ``_running_agents`` alone, so
it tore the transport down mid-upload and lost the attachment.

Every post-handler attachment path now holds a scoped delivery lease; the
shutdown drain (``_drain_active_agents`` via ``_drain_work_counts``) stays
pending until the last lease is released.
"""

import asyncio

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner


async def _hold_lease_over_upload(adapter, runner, done: asyncio.Event):
    """A post-handler attachment upload that outlives its agent slot, holding a lease."""
    async with adapter._delivery_lease("post-handler attachments"):
        await asyncio.wait_for(done.wait(), timeout=5.0)


@pytest.mark.asyncio
async def test_drain_waits_for_inflight_post_handler_delivery():
    """Agent state released + attachment upload still in flight: the drain waits for it."""
    runner, adapter = make_restart_runner()
    adapter.gateway_runner = runner  # adapters resolve the lease registry through the runner

    done = asyncio.Event()
    upload = asyncio.create_task(_hold_lease_over_upload(adapter, runner, done))
    await asyncio.sleep(0.05)  # let the upload take its lease

    # The agent slot is already released; the drain's only pending work is the upload lease.
    assert runner._running_agent_count() == 0
    assert runner._delivery_lease_count() == 1

    drain = asyncio.create_task(runner._drain_active_agents(5.0, None))
    await asyncio.sleep(0.1)
    assert not drain.done(), "drain finished while a post-handler upload lease was still held"

    done.set()
    snapshot, timed_out = await drain
    await upload
    assert timed_out is False
    assert runner._delivery_lease_count() == 0


@pytest.mark.asyncio
async def test_expired_drain_budget_reports_inflight_upload_as_timed_out():
    """An upload that outlives the drain budget still reports timed_out so shutdown proceeds."""
    runner, adapter = make_restart_runner()
    adapter.gateway_runner = runner

    done = asyncio.Event()
    upload = asyncio.create_task(_hold_lease_over_upload(adapter, runner, done))
    await asyncio.sleep(0.05)

    snapshot, timed_out = await runner._drain_active_agents(0.05, None)
    assert timed_out is True, "an in-flight post-handler upload must count as pending work"

    done.set()
    await upload
    assert runner._delivery_lease_count() == 0


@pytest.mark.asyncio
async def test_delivery_lease_registry_releases_on_error():
    """A delivery that raises must not leave its lease in the registry."""
    from gateway.delivery_lease import DeliveryLeaseRegistry

    registry = DeliveryLeaseRegistry()
    with pytest.raises(RuntimeError):
        async with registry.lease("attachments"):
            assert len(registry) == 1
            raise RuntimeError("upload failed")
    assert len(registry) == 0

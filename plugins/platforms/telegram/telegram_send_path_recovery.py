"""Telegram in-place send-path recovery: the degraded -> healthy edge of one adapter instance.

A final refused with ``send_path_degraded`` stays ``failed`` in the delivery ledger until a runtime
sweep claims it. The runner sweeps when it REPLACES an adapter (reconnect), but Telegram's polling
ladder usually heals the same adapter in place, so without this edge the reply waited for a restart.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# Lets a final refused just before the edge commit its ``failed`` row before the sweep reads the
# ledger: ``_send_with_retry`` returns the refusal and the row is marked a few ms later.
_SWEEP_SETTLE_SECONDS = 1.0


def mark_send_path_recovered(adapter) -> None:
    """First proof getUpdates is flowing for this generation: flip a published "retrying" (degraded
    connect, reconnect stamp, mid-session recovery) back to "connected" (#101391) and replay the
    finals this adapter refused while degraded."""
    if not adapter._send_path_degraded:
        return
    adapter._send_path_degraded = False
    if getattr(adapter, "_running", False) and not adapter.has_fatal_error:
        adapter._write_runtime_status_safe(
            "connected", platform_state="connected", error_code=None, error_message=None,
        )
        _schedule_recovery_sweep(adapter)


def _schedule_recovery_sweep(adapter) -> None:
    """One coalesced runtime-ledger sweep per adapter; an edge during a running sweep reruns it."""
    runner = getattr(adapter, "gateway_runner", None)
    redeliver = getattr(runner, "_redeliver_failed_obligations_for_platform", None)
    if not callable(redeliver):
        return
    task = getattr(adapter, "_send_path_recovery_task", None)
    if task is not None and not task.done():
        adapter._send_path_recovery_rerun = True
        return
    adapter._send_path_recovery_rerun = False

    async def _sweep() -> None:
        while True:
            await asyncio.sleep(_SWEEP_SETTLE_SECONDS)
            adapter._send_path_recovery_rerun = False
            if adapter.send_path_degraded or not getattr(adapter, "_running", False):
                return
            delivered = await redeliver(adapter.platform, profile=getattr(adapter, "_owner_profile", None))
            if delivered:
                logger.info("[%s] Redelivered %d final(s) refused while the send path was degraded",
                            adapter.name, delivered)
            if not adapter._send_path_recovery_rerun:
                return

    task = adapter._send_path_recovery_task = asyncio.get_running_loop().create_task(
        _sweep(), name=f"telegram-send-path-recovery:{adapter.name}")
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)

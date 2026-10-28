"""The startup gate for an external drain marker already present at boot (mixin bound via the MRO).

The drain-control watcher (``GatewayShutdownMixin._drain_control_watcher``) polls
``.drain_request.json`` (``gateway/drain_control.py``) and owns release, but it is spawned at the
very end of ``start()``. This lane reads the marker once before readiness, so a gateway started
under an active drain publishes ``draining`` instead of its serving state.

Split out of ``gateway/run_startup.py`` (its size ratchet is at its target); bound onto
``GatewayRunner`` through ``GatewayStartupMixin``.
"""

from __future__ import annotations

import asyncio


class GatewayStartupDrainMixin:
    """Read the external drain marker before readiness and publish the first state accordingly."""

    @staticmethod
    async def _read_startup_drain_marker() -> bool:
        """The drain watcher only starts after startup restore has admitted the queued inbound, so
        a marker already present must gate those turns and the first published state itself."""
        from gateway.drain_control import drain_requested
        return await asyncio.to_thread(drain_requested)

    def _publish_startup_state(self, drain_at_start: bool) -> None:
        """First published state of a boot: ``draining`` under an active marker, else serving."""
        if drain_at_start:
            self._enter_external_drain()
        else:
            # A boot that could not start every configured platform is not a normal run: stamp ``degraded``
            # so ``gateway status`` / /api/status / the health snapshot surface it, instead of only a log
            # line next to "Gateway running with N platform(s)".
            self._update_runtime_status(self._serving_state())

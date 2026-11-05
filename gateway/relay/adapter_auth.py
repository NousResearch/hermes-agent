"""Relay credential lifecycle for the relay adapter: waiting for credentials, and revocation.

``RelayAdapter`` mixes this in. ``connect()`` calls ``_await_relay_credentials`` when the
transport refuses to dial without credentials, and ``_start_revocation_monitor`` once a
handshake succeeds so a later terminal 4401 disables the adapter cleanly.
"""

from __future__ import annotations

import asyncio
import logging

from gateway.relay.transport import RelayCredentialsMissing

logger = logging.getLogger("gateway.relay.adapter")


class RelayAuthLifecycleMixin:
    """Fatal-error states for a relay that has no credentials yet or had them revoked."""

    def _await_relay_credentials(self, exc: RelayCredentialsMissing) -> bool:
        """Record why the relay is not connected and report the connect as failed.

        Retryable: the gateway's provision retry obtains credentials and pulls the
        reconnect watcher forward, which rebuilds this adapter from them. The message
        (with the connector's last refusal) is what ``hermes status`` and the dashboard
        show while the relay waits."""
        self._set_fatal_error("relay_credentials_missing", str(exc), retryable=True)
        return False

    def _start_revocation_monitor(self) -> None:
        """Spawn (once) the task turning a transport auth-revocation into a clean
        non-retryable 'relay disabled' fatal. Idempotent."""
        if self._revocation_monitor is not None and not self._revocation_monitor.done():
            return
        try:
            self._revocation_monitor = asyncio.create_task(
                self._watch_for_revocation(), name="relay-revocation-monitor"
            )
        except RuntimeError:
            # No running loop (a unit test calling connect() via a stub).
            self._revocation_monitor = None

    async def _watch_for_revocation(self, poll_interval_s: float = 1.0) -> None:
        """Poll for a terminal 4401 revocation (opt-out), then surface a non-retryable
        `relay_disabled` fatal so the adapter is cleanly removed rather than queued
        for reconnection (the credential is dead until the instance is recreated)."""
        transport = self._transport
        if transport is None:
            return
        while not getattr(transport, "auth_revoked", False):
            await asyncio.sleep(poll_interval_s)
        logger.warning("relay credential revoked (opt-out) — marking the relay adapter disabled")
        self._set_fatal_error(
            "relay_disabled", "Relay disabled (opted out — recreate the instance to re-enable)",
            retryable=False,
        )
        try:
            await self._notify_fatal_error()
        except Exception:
            logger.debug("relay revocation fatal-error notify failed", exc_info=True)

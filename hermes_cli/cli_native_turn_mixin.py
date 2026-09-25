"""Native-turn source admission and lifecycle hooks for the interactive CLI."""

from __future__ import annotations

from contextlib import suppress
from typing import Any


class CLINativeTurnMixin:
    def _native_session_view(self: Any):
        """Return immutable exact identity and ownership facts for this CLI session."""
        from hermes_constants import get_hermes_home
        from hermes_cli.native_turn_sources import NativeSessionView
        from hermes_cli.profiles import get_active_profile_name

        session_id = str(getattr(self.agent, "session_id", "") or "")
        lineage = (session_id,)
        db = getattr(self.agent, "session_db", None)
        getter = getattr(db, "get_compression_lineage", None)
        if callable(getter):
            with suppress(Exception):
                lineage = tuple(getter(session_id)) or lineage
        lease = getattr(self, "_active_session_lease", None)
        return NativeSessionView(
            profile=get_active_profile_name(),
            profile_home=get_hermes_home(),
            session_id=session_id,
            surface="cli",
            compression_lineage=lineage,
            session_key=session_id,
            owner_token=str(getattr(lease, "lease_id", "") or "") or None,
        )

    def _fence_native_turn_sources(self: Any, reason: str) -> None:
        """Synchronously cancel source events before a CLI session boundary."""
        from hermes_cli.native_turn_sources import fence_native_turn_sources

        if (
            getattr(self, "_active_session_lease", None) is not None
            and getattr(self, "agent", None)
        ):
            fence_native_turn_sources(self._native_session_view(), reason)

    def _maybe_start_native_turn(self: Any) -> bool:
        """Poll and admit one native lease through the ordinary CLI turn path."""
        from hermes_cli.native_turn_sources import poll_native_turn

        if (
            self._agent_running
            or self._should_exit
            or getattr(self, "_active_session_lease", None) is None
            or not self._pending_input.empty()
            or not self._interrupt_queue.empty()
        ):
            return False
        view = self._native_session_view()
        admission = poll_native_turn(view)
        if admission is None:
            return False

        # Reserve the host slot before the pre-model commit. Recheck human queues
        # after reservation because polling crosses into plugin code.
        self._agent_running = True
        if (
            self._should_exit
            or not self._pending_input.empty()
            or not self._interrupt_queue.empty()
            or self._native_session_view() != view
        ):
            self._agent_running = False
            admission.abort_if_open(
                "not_sent",
                "CLI session became busy or changed before host admission",
            )
            return False
        if not admission.commit():
            self._agent_running = False
            return False
        try:
            self._tui_process_one_input(admission.lease.prompt)
        finally:
            # The ordinary chat path normally clears this; retain a backstop for
            # exceptions before that path reaches its own finally block.
            if self._agent_running:
                self._agent_running = False
        return True

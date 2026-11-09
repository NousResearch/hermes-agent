"""Per-channel slash-command cap gate for inbound messaging turns.

Extracted from ``run_inbound._handle_message`` (the code-health remedy for an already deep
pipeline: split the phase into a topical sibling). The gate answers one question — may the
command this event carries run in this exact channel at all, and if not, what does the sender
read? — and records the command metric only for what actually proceeds, so a denial never
counts as a run.

The cap evaluates the event **as admission left it**: after profile routing, the ignored-channel
guard and the ``pre_gateway_dispatch`` plugin hook. That ordering is deliberate. A hook that
rewrites an allowed typed command into a capped one is gated on the rewritten canonical name, not
on what was typed, so a hook cannot smuggle a command past the cap; a hook can observe inbound
text the same way it observes every other inbound message, and the per-sink checks in
``_check_slash_access`` still stand behind this one for aliases and later rewrites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from gateway.session import SessionSource


class GatewayInboundCommandCapsMixin:
    """The channel-cap phase of the inbound pipeline, ahead of estop, pending-reply interception
    and every command dispatch sink."""

    def _hm_channel_command_gate(
        self, event: Any, source: SessionSource, is_internal: bool
    ) -> Optional[str]:
        """Deny a capped command, else record its command metric and let the turn proceed."""
        if is_internal:
            return None
        denial = self._hm_channel_command_cap_denial(event, source)
        if denial is not None:
            return denial
        from hermes_cli.observability.shared_metrics_events import record_gateway_slash_command

        record_gateway_slash_command(event)
        return None

    def _hm_channel_command_cap_denial(self, event: Any, source: SessionSource) -> Optional[str]:
        """Exact per-channel cap denial for the command this event now carries, if it carries one.

        Shares :meth:`GatewayBusySessionMixin._check_channel_command_access` with the dispatch sinks
        so ingress and every later check resolve the cap through the same profile-aware seam.
        """
        command = event.get_command()
        if not command:
            return None
        from hermes_cli.commands import resolve_command

        resolved = resolve_command(command)
        canonical = (
            resolved.name if resolved
            else command.strip().lstrip("/").lower().replace("_", "-")
        )
        return self._check_channel_command_access(source, canonical)

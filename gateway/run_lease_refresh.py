"""Lease-wait refresh suppression for the gateway status callback (#89166).

Moved out of ``gateway/run.py`` wholesale when it grew past that facade's FILE_LINES
cap (moved code keeps its own cap; see root AGENTS.md). Pure policy helpers: they
decide, they never send — delivery stays in ``_status_callback_sync`` /
``_send_or_update_status_coro``.
"""
import re

_SESSION_TURN_LEASE_REFRESH_RE = None


def _session_turn_lease_refresh_re():
    """Compile-once matcher for the turn-lease wait refresh.

    Derived from the SAME template constant the emit site formats
    (SESSION_TURN_LEASE_WAIT_REFRESH_STATUS_TEMPLATE in agent.turn_facade_lease, #89166) —
    never re-inlined wording, same convention as _COMPRESSION_PROGRESS_STATUS_RE. The
    imports stay lazy: this sibling must not drag gateway.run (module-scope heavy) or
    the agent layer into its own import time.
    """
    global _SESSION_TURN_LEASE_REFRESH_RE
    if _SESSION_TURN_LEASE_REFRESH_RE is None:
        from agent.turn_facade_lease import SESSION_TURN_LEASE_WAIT_REFRESH_STATUS_TEMPLATE
        from gateway.run import _status_template_to_regex

        _SESSION_TURN_LEASE_REFRESH_RE = re.compile(
            _status_template_to_regex(SESSION_TURN_LEASE_WAIT_REFRESH_STATUS_TEMPLATE),
            re.IGNORECASE,
        )
    return _SESSION_TURN_LEASE_REFRESH_RE


def _should_suppress_lease_wait_refresh(platform, adapter, message) -> bool:
    """True when a periodic turn-lease wait refresh would flood a HUMAN chat.

    The periodic refresh exists to update the initial "Another Hermes process
    is using this session..." notice in place. On human-chat adapters without
    ``send_or_update_status`` (WeCom, Weixin, QQ, Signal, ... — all
    SUPPORTS_MESSAGE_EDITING = False) ``_send_or_update_status_coro`` falls
    back to a plain send, so every ~15s refresh lands as another standalone
    chat message (#89166). Suppress the refresh there only:
      * adapters with ``send_or_update_status`` edit the same bubble, so the
        refresh is not noise;
      * raw programmatic surfaces (``_gateway_surface_passes_raw_text``:
        webhook, api_server, local, msgraph_webhook) feed machine consumers
        that read the periodic heartbeat as lease liveness — dropping it makes
        automation treat an admitted, still-waiting turn as stalled.
    The initial notice and the lease-timeout warning use different wording
    and are always delivered.
    """
    from gateway.run import _gateway_surface_passes_raw_text

    if _gateway_surface_passes_raw_text(platform):
        return False
    if callable(getattr(adapter, "send_or_update_status", None)):
        return False
    return bool(_session_turn_lease_refresh_re().search(str(message or "")))

"""Bulk prune while the gateway is up: the owner applies it, this process never writes.

The one always-on gateway owns ``state.db``; a CLI rewriting it underneath is the second writer
the held-store refusal exists to stop. So when a ready owner serves this home, ``hermes sessions
prune`` reads its preview from a read-only view and sends the selected filters to the owner
(``session.prune``), which deletes through the same ``retire_prunable`` path housekeeping uses.
With no gateway, prune stays the local command it always was (holder scan, ``--force``).
"""
from __future__ import annotations

#: A cron-heavy store prunes tens of thousands of rows in one transaction; the owner shields the
#: write, so a short client budget would only misreport a prune that still commits.
PRUNE_RPC_TIMEOUT = 600


def live_owner():
    """The ready gateway endpoint serving this home, else None. Discovery only: never starts one."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_constants import get_hermes_home
    found = discover_gateway_endpoint(get_hermes_home().resolve())
    return found.endpoint if found.state == "ready" else None


def owner_prune(endpoint, **params) -> dict:
    """Run one ``session.prune`` on *endpoint*; raises ``GatewayClientError`` on refusal/loss."""
    import asyncio
    from hermes_cli.gateway_client import connect_gateway

    async def _run():
        async with connect_gateway(endpoint) as client:
            return await client.rpc("session.prune", _timeout=PRUNE_RPC_TIMEOUT, **params)
    return asyncio.run(_run())


def report_owner_failure(exc) -> int:
    print(f"Error: the gateway did not complete the prune ({exc}). Run `hermes sessions prune --dry-run` "
          "with the same filters to see what is left.")
    return 1

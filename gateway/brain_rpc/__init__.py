"""Host-side Brain RPC (G3.2) — MVP read methods on the customer Hermes host.

Accepts ``brain_rpc_request`` frames on an authenticated relay WebSocket session
and returns ``brain_rpc_result`` frames. Vault/settings/projects ops run locally
on the host; there is no public inbound brain-RPC port.

Contract (authority): company-brain-deploy ``references/brain-rpc-contract.md``.
Local summary: ``docs/lanyard-brain-rpc.md``.
"""

from __future__ import annotations

from gateway.brain_rpc.dispatcher import (
    BRAIN_RPC_CONTRACT_VERSION,
    BrainRpcDispatcher,
    handle_brain_rpc_request,
    is_brain_rpc_enabled,
)

__all__ = [
    "BRAIN_RPC_CONTRACT_VERSION",
    "BrainRpcDispatcher",
    "handle_brain_rpc_request",
    "is_brain_rpc_enabled",
]

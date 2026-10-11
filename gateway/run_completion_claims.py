"""Durable completion claim preflight and ownership transitions."""
from __future__ import annotations
import dataclasses
import logging
from typing import Optional

logger = logging.getLogger("gateway.run")

class GatewayCompletionClaimsMixin:

    @dataclasses.dataclass
    class _CompletionClaim:
        """Pre-flight outcome for one completion delivery."""

        delegation_id: str = ""
        claim_id: str = ""
        proceed: bool = True
        early_result: Optional[bool] = None

    async def _preflight_completion_delivery(
        self, evt: dict, *, durable_claim_id: Optional[str] = None,
        claim: Optional[_CompletionClaim] = None,
    ) -> _CompletionClaim:
        """Claim the durable row (async delegations) and verify the target before adapter acceptance.

        Adapter acceptance is not proof of delivery: the inner resolver can still fail closed inside
        the pipeline after acceptance, falsely acking the durable row. Verifying first gives drops an
        honest durable disposition.
        """
        if claim is None:
            claim = self._CompletionClaim()
        evt_type = evt.get("type")
        if (evt_type == "async_delegation" and durable_claim_id is None
                and not await self._completion_delivery_ready(evt)):
            claim.proceed, claim.early_result = False, False
            return claim
        # An interim per-task notice shares the batch's delegation_id but is not the durable
        # completion; claiming that row here would acknowledge the FINAL result before it exists.
        if evt_type == "async_delegation" and not evt.get("task_failure_notice"):
            claim.delegation_id = str(evt.get("delegation_id") or "")
            if durable_claim_id is not None:
                claim.claim_id = durable_claim_id
            elif claim.delegation_id:
                try:
                    from tools.async_delegation import claim_completion_delivery
                    claim.claim_id = f"gateway:{id(self)}:{__import__('uuid').uuid4().hex}"
                    if not claim_completion_delivery(claim.delegation_id, claim.claim_id):
                        claim.proceed = False
                        return claim
                except Exception as exc:
                    logger.warning("Could not claim durable async completion %s: %s", claim.delegation_id, exc, exc_info=True)
                    claim.proceed, claim.early_result = False, False
                    return claim
        elif evt_type != "completion":
            return claim
        # Background completions carry only session_key, so after /new the OLD session's notification
        # would land in the NEW one. Stamped events get the async-delegation pre-flight; unstamped deliver.
        parent_session_id = str(evt.get("parent_session_id") or "").strip()
        if not parent_session_id:
            return claim
        # Pre-flight (#65838-class): adapter acceptance is NOT proof of delivery — the inner #55578 resolver
        # can still fail closed inside the message pipeline AFTER the adapter accepted, which would falsely
        # acknowledge the durable row as delivered. Verify the target here, before acceptance, and give
        # drops an honest durable disposition.
        verdict = await self._classify_completion_target(parent_session_id)
        if verdict == "terminal":
            if evt_type == "async_delegation":
                logger.warning(
                    "Async delegation %s targets permanently-gone session %s; "
                    "terminally dropping delivery (result remains in the delegation records).",
                    claim.delegation_id or "<legacy>", parent_session_id,
                )
                if claim.claim_id:
                    self._settle_durable_claim("drop", claim.delegation_id, claim.claim_id)
            else:
                logger.warning(
                    "Background process %s completion targets "
                    "permanently-gone session %s (user boundary such as "
                    "/new); dropping notification (output remains available via process(action='log')).",
                    evt.get("session_id") or "<unknown>", parent_session_id,
                )
            claim.proceed = False
        elif verdict == "retry":
            # Transient uncertainty: tell the watcher to re-poll rather than drop or misroute.
            if claim.claim_id:
                self._settle_durable_claim("release", claim.delegation_id, claim.claim_id)
            claim.proceed, claim.early_result = False, False
        return claim

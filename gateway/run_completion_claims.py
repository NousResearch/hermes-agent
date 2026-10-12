"""Coalesced delegation batch delivery using the current notification claim and owner proof.

Bound through GatewayNotificationsMixin; the runner facade remains the late-import seam.
The current _CompletionClaim and preflight stay owned by run_notifications.py.
"""
from __future__ import annotations

from typing import Optional


class GatewayCompletionBatchMixin:
    async def _deliver_async_delegation_group(self, group: list[dict]) -> Optional[bool]:
        """Deliver a same-session batch of async completions as ONE turn: the primary carries the
        consolidated text of every sibling THIS runner claimed (siblings owned elsewhere are excluded;
        their claims are acked only after adapter acceptance). True after acceptance, False to requeue
        the group, None when nothing is deliverable here (retry siblings requeued)."""
        # The group shares one session_key, hence one profile: scope the pre-checks and sibling claims too.
        async with self._completion_event_scope(group[0]):
            return await self._deliver_async_delegation_group_scoped(group)


    async def _deliver_async_delegation_group_scoped(self, group: list[dict]) -> Optional[bool]:
        from gateway.run import _format_gateway_process_notification
        from tools.process_registry import process_registry as _pr
        # API delivery does not start a model turn, so there is nothing to coalesce.
        # Keep each unit's stable identity with its row across partial delivery/retry.
        if group and group[0].get("origin_session_id"):
            outcomes = []
            for evt in group:
                text = _format_gateway_process_notification(evt)
                if text:
                    outcomes.append(await self._deliver_completion_notification(text, evt))
            return False if False in outcomes else True
        deliverable: list[tuple[dict, str]] = []
        for evt in group:
            synth_text = _format_gateway_process_notification(evt)
            if not synth_text:
                continue
            parent = str(evt.get("parent_session_id") or "")
            if parent and await self._classify_completion_target(parent, str(evt.get("session_key") or "")) == "terminal":
                await self._deliver_completion_notification(synth_text, evt)
                continue
            identity = self._completion_delivery_identity(evt)
            if identity is not None and self._completion_identity_seen(identity):
                continue
            deliverable.append((evt, synth_text))
        if not deliverable:
            return None
        if len(deliverable) == 1:
            evt, synth_text = deliverable[0]
            return await self._deliver_completion_notification(synth_text, evt)
        return await self._deliver_async_delegation_batch(deliverable, _pr)

    async def _deliver_async_delegation_batch(self, deliverable: list[tuple[dict, str]], _pr) -> Optional[bool]:
        """Claim and admit a ready batch, retaining exact primary proof and failure settlement."""
        # Check the entire group before claiming ANY row: an unavailable sibling must
        # not exhaust its budget just because the primary has a usable route.
        for evt, _text in deliverable:
            if not await self._completion_delivery_ready(evt):
                return False
        from tools.async_delegation import claim_event_delivery
        primary_evt, primary_text = deliverable[0]
        # A competing primary owner must leave every sibling's retry budget untouched.
        primary_claim_id = claim_event_delivery(primary_evt, f"gateway-batch-primary:{id(self)}")
        if primary_claim_id is None:
            return None
        handoff_returned = False
        primary_claim = self._CompletionClaim(
            delegation_id=str(primary_evt.get("delegation_id") or ""), claim_id=primary_claim_id,
        )
        siblings: list[tuple[dict, str]] = []
        try:
            # Prove the primary owner before claiming siblings; carry that same proof to admission.
            primary_claim = await self._preflight_completion_delivery(
                primary_evt, durable_claim_id=primary_claim_id, claim=primary_claim,
            )
            if not primary_claim.proceed:
                if primary_claim.early_result is None:
                    for sibling_evt, _text in deliverable[1:]:
                        _pr.completion_queue.put(sibling_evt)
                return primary_claim.early_result
            blocks = [primary_text]
            for evt, synth_text in deliverable[1:]:
                claim_id = claim_event_delivery(evt, f"gateway-batch:{id(self)}")
                if claim_id is None:
                    # Another consumer owns this row: never include its result in our injection.
                    continue
                siblings.append((evt, claim_id))
                blocks.append(synth_text)
            header = (
                f"[IMPORTANT: {len(blocks)} background subagent delegations "
                "completed for this session. Treat these results as one "
                "completion batch and send at most one consolidated user-facing "
                "response. If a result does not change the current conclusion, absorb it silently.]"
            )
            consolidated = "\n\n".join([header, *blocks]) if siblings else primary_text
            delivered = await self._deliver_completion_notification(
                consolidated, primary_evt, sibling_claims=siblings,
                _durable_claim_id=primary_claim_id, _preflight_claim=primary_claim,
            )
            handoff_returned = True
        finally:
            # The wrapper may fail while entering its scope, before its claim finalizer.
            # Release is identity-checked and safe if the inner finalizer already ran.
            if not handoff_returned:
                for event, claim_id in [(primary_evt, primary_claim_id), *siblings]:
                    if claim_id:
                        self._settle_durable_claim("defer" if primary_claim.defer else "release", event["delegation_id"], claim_id)
        if delivered is None:
            # Primary dropped/owned elsewhere: retry the unadmitted siblings.
            for evt, _claim_id in siblings:
                _pr.completion_queue.put(evt)
        return delivered

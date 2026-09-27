# Conversation framing

Status: behavior agreed; runtime implementation pending.

## Decision

Retain native conversation framing and Telegram transport. Preserve the native
separation of observed conversation from the current addressed request. Add
these employee semantics where missing:

1. **Person-aware sender labels.** Use consistent labels backed by stable
   person records across chats. Disambiguate colliding display names using the
   employee label convention. Display labels never become identity keys.
2. **Short conversation references.** Expose compact `@session:...` handles
   that `session_search` can resolve. Preserve useful thread pointers without
   filling model context with long internal identifiers. Scope resolution to
   the owning employee and existing access boundaries.
3. **Confirmed outbound-message context.** On the destination conversation's
   next turn, include confirmed messages delivered there by schedules or
   `send_message` as observed context, including in DMs. This lets the employee
   understand replies to work delivered outside that conversation's live turn.

## Context contract

Observed messages are context, not fresh requests. The current addressed
message remains distinct, followed by the current speaker's personal-memory
block and recalled context in [the agreed order](tool-surface.md).

Keep clean user content separate from the API-bound framing. Preserve exact
historical API content on replay; do not rewrite prior turns or mutate a live
cached prompt to insert a delivery. Do not introduce synthetic user requests
or duplicate assistant turns for outbound sends.

Only confirmed deliveries qualify as delivered context. Failed or ambiguous
sends must not be presented as successful. Deduplicate delivery context so a
send is not newly introduced on every subsequent turn; historical replay of
its original context remains unchanged. Ordinary replies already represented
in conversation history must not be added again as external deliveries.

## Native integration

Use native sender/session metadata, history assembly, search and delivery
owners. Extend those paths directly rather than importing hosted channel
adapters or a parallel conversation store. Preserve native attachment,
delegation, mid-turn steering and Telegram display behavior.

Reconcile existing transcript mirroring with the new outbound-context contract
so a delivery has one model-visible representation. Native session IDs may
remain internal; compact handles are a model-facing reference, not permission
to access another profile's history.

## Verification when implemented

Verify colliding names remain attributable, compact handles resolve the correct
conversation, a reply to an externally delivered report receives that report's
context, and failed/duplicate deliveries do not appear as new successes. Cover
DM and group flows, session restoration, profile isolation and byte-stable
historical context. Verify the actual model request, not merely stored rows.

# Background learning

Status: behavior agreed; runtime implementation pending.

## Decision

Preserve the employee simplification: one unified review prompt covers authored
memory, service manuals and responsibilities. Do not retain separate model-facing
memory-review and skill-review doctrines or reintroduce skills through review.

Preserve the native review mechanics rather than building a second learning
system: lifecycle triggers, cadence, thresholds, review ceilings, conversation
forking and applicable restoration/cache behavior. Integrate the unified prompt
through the existing review engine. The model-facing simplification does not
justify replacing the scheduler or adding a parallel reviewer.

## Review contract

Use the established employee review prompt verbatim except for agreed local
paths and excluded capabilities. Preserve its instructions to read current
records before writing, improve existing knowledge rather than duplicate it,
consolidate superseded content, and file corrections at the source.

Reviews can update authored memory, service manuals and responsibility records
under the employee write rules. Preserve the review-specific prohibition on
schedule/webhook declaration edits; proposed automation changes belong in
`STATE.md`. Reviews must not send messages or delegate. Product guides remain
product-owned, and removed tools such as `report_issue` must not appear in the
review's schema or instructions.

Adapt the reference prohibition on creating connection folders: local service
manuals are employee-owned and no connection-store event creates them. A review
may create a missing manual only when the source conversation establishes
verified service access; preserve read-before-create deduplication. Do not
invent verified access from a service mention or an old listing. This follows
the approved [manual creation rule](guides.md).

## Context and execution boundaries

The review uses a conversation snapshot and the inherited employee prompt.
Do not append its harness or edits to the source conversation, rewrite past
context or rebuild the live system prompt. Preserve memory/person attribution
when processing a shared conversation; a correction must reach the intended
person or shared store, not whichever user happens to be bound later.

Hindsight retention and recall are separate from this authored-knowledge review.
Do not introduce a second reconciler for Hindsight as part of this change.

Retain native review lifecycle, concurrency and restart behavior. The unified
prompt decision does not import hosted queues, leases or storage. User-visible
review notices were not separately selected; retain the native default unless
a later display decision changes it.

## Verification when implemented

Verify the actual review fork receives one unified prompt regardless of the
trigger combination; relevant native cadence and ceilings remain intact;
corrections update the correct knowledge home; declaration writes are refused;
and repeated review consolidates rather than multiplies records. Check profile
and person isolation, unchanged source transcripts and cached prompt bytes.

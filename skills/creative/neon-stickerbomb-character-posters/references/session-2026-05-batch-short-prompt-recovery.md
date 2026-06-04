# Batch Short-Prompt Recovery After Empty Responses

Session note from 2026-05-21.

When a multi-image neon batch starts failing with repeated `empty_response`, `500`, or `connection reset` errors on the same slot, the fix that worked was:

1. Keep the successful indices in the manifest untouched.
2. Retry the failing slot with a much shorter prompt.
3. Preserve only the durable anchors: subject identity, one composition archetype, `NickZag` placement, palette, safety, and a compact negative block.
4. If the same slot still fails and the user asked for a batch by count rather than a strict roster, swap that slot to another fitting character/subject so the batch completes instead of thrashing.
5. After interruptions, treat the manifest as the source of truth and resume from completed items, not from conversation recency.

This is a compact recovery pattern for image batches that keeps output moving when verbose prompts or one specific subject keep stalling.

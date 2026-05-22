# Recap and synthesis

Reference for the `recap_round` and `recap_session` flags on `create_deliberation` and `append_round`, and the `round_recap` / `session_synthesis` artifacts they produce.

## Flags

| Flag | Accepted on | Effect |
|---|---|---|
| `recap_round` | `create_deliberation` AND `append_round` | Generates a structured `round_recap` for that round (`title`, `tldr`, `agenda`, `sections`). Surfaces on `get_session` once written. |
| `recap_session` | `append_round` ONLY (rejected on `create_deliberation` with a 400) | Generates `session_synthesis` (`title`, `tldr`, `origin`, `arcs`) over the round-recap set when this round completes. Implicitly covers `recap_round` for the trigger round. |

Both default `false` — no behavior change unless you opt in. `recap_session` is rejected on `create_deliberation` because synthesis requires ≥2 rounds; on round 0 it would degenerate to a round-recap-only behavior.

## Cascade behavior

Setting `recap_session: true` triggers `round_recap` generation for every prior round that doesn't already have one — session synthesis needs the round-recap set as input.

Example: if you only opted into `recap_round` for rounds 2 and 4 and then set `recap_session: true` on round 5, the cascade backfills `round_recap` for rounds 1 and 3 as well. The trigger round (5) is recap'd automatically — you don't need to set `recap_round: true` alongside `recap_session: true`.

## Cost

Recap and synthesis bill via the standard credit wallet at **0 bps markup** (at-cost passthrough). A typical 3-round cascade lands around ~$0.04 in inference cost — the cascade is the surprising part of the bill, so check session length before flipping `recap_session: true` on a long session.

## Reading the artifacts

`get_session` returns:

- `rounds[].round_recap` — per-round, populated for rounds whose recap generation has completed (either directly opted in via `recap_round` or backfilled via the `recap_session` cascade). Null otherwise.
- `session_synthesis` — session-level, populated once the cascade completes. Null until then.

Use these as the structured "what happened" read path instead of parsing raw prose — they're produced specifically for the agent-consumption surface.

## When to set them

- `recap_round` is cheap and worth setting any time the user might want a round-level summary later, or any time the round's claim map has significant structure you may need to reference.
- `recap_session` is the right move when the user signals they're done deliberating and wants a session-level takeaway — set it on the final `append_round`, not earlier. Setting mid-session triggers the cascade and surprises the cost.

## Public contract

Full contract: https://mumo.chat/docs/mcp#recap-artifacts

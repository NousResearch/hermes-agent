# Truth Ledger reduced-MVP handoff

## Status

The reduced MVP remains isolated, unmerged, and disabled. The prior candidate (`f3395ac`) failed a later adversarial independent review; its findings have been remediated in a new candidate that must pass verification and exact-commit re-review before rollout.

- Branch: `feat/truth-ledger-option-2`
- Worktree: `/Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2`
- Final evaluation: `docs/truth-ledger/evaluation/report.md`
- Final QA record: `docs/truth-ledger/qa/mvp-final.md`

## Delivered behavior

- Eligible successful top-level turns enqueue a sanitized source envelope with bounded in-process suppression and durable idempotency across pending, processing, dead-letter, and completed spool records. The design is not a distributed exactly-once protocol.
- Missing trusted identity blocks user-scoped activation.
- Structured extraction may abstain and uses strict canonical identities, keys, kinds, and values.
- Assert, confirm, supersede, duplicate, and retract behavior is append-only and deterministic.
- Projection rebuild is deterministic.
- Duplicate/replay handling, completed tombstones, stale-processing recovery, and orphan-payload recovery are implemented and covered by focused tests.
- Secret-like input is redacted before persistence.
- Provider/plugin failure remains fail-open for chat.
- Ledger events are validated against the frozen schema before persistence.
- No automatic writes target USER.md, MEMORY.md, Hermes Memory, or GBrain.

## Enablement boundary

Do not enable the plugin, modify the default profile, restart the gateway, merge, push, or publish without explicit authorization. The current artifacts establish technical readiness only.

## Known reduced-MVP limitation

The MVP does not include an automatic runtime consumer. Pending records require the explicit, dry-run-first operator command `/truth-ledger process --limit N [--apply]`, where `N` is 1–3. The command is item-bounded, but its history and projection operations scan the ledger, so it is not claimed to be strictly wall-clock- or total-I/O-bounded.

## Deferred items

- Human authorization for merge/release/live enablement
- Release packaging and publication
- Live-profile canary and observation window
- Automatic bounded pending-spool consumer
- Optional UI/operational polish

## Prior evidence requiring revalidation

- 145/145 measured extraction turns
- Observed route: `openai-codex/gpt-5.6-sol` on every turn
- Precision/recall/abstention/accuracy: 1.0000
- Leakage: 0.0000 on that specific synthetic corpus; this is not a general zero-leakage guarantee
- 133/133 targeted and integration tests passed on the prior candidate
- Disposable temporary-home canary passed on the prior candidate
- Later adversarial independent review: REQUEST_CHANGES; fresh approval is required for the remediated exact commit

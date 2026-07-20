# Truth Ledger reduced-MVP handoff

## Status

The reduced MVP has passed its final technical acceptance gate and independent review. It remains isolated, unmerged, and disabled.

- Branch: `feat/truth-ledger-option-2`
- Worktree: `/Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2`
- Final evaluation: `docs/truth-ledger/evaluation/report.md`
- Final QA record: `docs/truth-ledger/qa/mvp-final.md`

## Delivered behavior

- Eligible successful top-level turns enqueue a sanitized source envelope exactly once.
- Missing trusted identity blocks user-scoped activation.
- Structured extraction may abstain and uses strict canonical identities, keys, kinds, and values.
- Assert, confirm, supersede, duplicate, and retract behavior is append-only and deterministic.
- Projection rebuild is deterministic.
- Duplicate/replay handling and restart recovery are covered.
- Secret-like input is redacted before persistence.
- Provider/plugin failure remains fail-open for chat.
- Ledger events are validated against the frozen schema before persistence.
- No automatic writes target USER.md, MEMORY.md, Hermes Memory, or GBrain.

## Enablement boundary

Do not enable the plugin, modify the default profile, restart the gateway, merge, push, or publish without explicit authorization. The current artifacts establish technical readiness only.

## Known reduced-MVP limitation

The MVP does not include a bounded automatic runtime consumer that continuously drains `spool/pending` through extraction, reconciliation, ledger append, and projection. Pending records require the documented operator path. This is intentional and documented in `docs/truth-ledger/operator-privacy-recovery-rollout.md`.

## Deferred items

- Human authorization for merge/release/live enablement
- Release packaging and publication
- Live-profile canary and observation window
- Automatic bounded pending-spool consumer
- Optional UI/operational polish

## Final evidence

- 145/145 measured extraction turns
- Observed route: `openai-codex/gpt-5.6-sol` on every turn
- Precision/recall/abstention/accuracy: 1.0000
- Leakage: 0.0000
- 133/133 targeted and integration tests passed
- Disposable temporary-home canary passed
- Independent final review: APPROVE

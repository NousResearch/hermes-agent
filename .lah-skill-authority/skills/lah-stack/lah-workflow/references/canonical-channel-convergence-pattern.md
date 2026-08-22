# Canonical Channel Convergence Pattern

Established during CLOE_TELEGRAM_GATEWAY_BEHAVIOR_PARITY_V1 (2026-08-01): unify two
channels (Gateway / OpenClaw Center + Telegram) onto ONE canonical conversational
service, with Telegram becoming a transport adapter instead of a second cognition.

## When to use

Any mission whose goal is "same behavior on both channels" — same identity, same
router, same brain, same memory policy, same governance, same provider/model,
channel-only rendering differences. The failure mode being fixed is a channel
that calls the provider directly or formats answers locally instead of routing
through the shared cognitive pipeline.

## Sequence (proven order)

1. **Baseline audit first (Lot 0)** — verify the divergences in the CURRENT HEAD
   before building anything. If the architecture is already unified, do NOT build
   a second abstraction: test, repair remaining gaps, document the proof.
   Deliverables: `docs/audit/CHANNEL_PARITY_BASELINE_V1.{md,json}` with a table of
   confirmed divergences (each with file:line evidence) and a catalog of EXISTING
   canonical components to compose (never rebuild).
2. **Canonical contract (Lot 1)** — `src/contracts/<name>-conversation-contract.js`:
   deterministic validation, explicit defaults, channel is DATA not behavior, no
   Telegram/WebSocket/network dependency, `sanitizeTrace` allowlist (secret fields
   stripped from traces), `hashSessionKey` (never expose raw session keys). Test
   purity statically (filter comment lines before `includes()` checks — JSDoc
   design-intent prose matches string assertions otherwise).
3. **Extract shared brain boundary (Lot 2/3)** — if the brain router lives inside
   one channel's adapter, extract it to its own module so the canonical service
   and the adapter share it WITHOUT an ESM import cycle. See ESM pitfall below.
4. **Canonical service (Lot 2)** — `src/services/<name>-canonical-conversation-service.js`
   composes existing components: contract → cognitive front router (conversation
   router + readonly router + supervisor natural-action router + shared brain
   router) → single session store → contract-shaped canonical response. Injectable
   deps (store, brainAsk, env, fetchImpl, routers) for tests.
5. **Migrate the reference channel first (Lot 3)** — make the Gateway an adapter
   of the canonical service (transport validation + persistence format + relay
   events stay local; cognition routes through the service). ALL existing channel
   tests must stay green BEFORE touching the other channel. Option
   `persistTranscript=false` lets an adapter keep its historical transcript format.
6. **Unify session identity (Lot 4)** — canonical convention `channel:<operatorId>:<chatId>`,
   stable, deterministic, collision-free, secret-free. Operator mapping:
   chat id → user id → operator identity (explicit env id, else primary chat) →
   canonical key. Allowlist helper shared with the channel handler.
7. **Flag-gated migration (Lot 5-6)** — flags (e.g. `CLOE_TELEGRAM_CANONICAL_PIPELINE_ENABLED`,
   `CLOE_TELEGRAM_CANONICAL_COMMANDS_ENABLED`): `false` → legacy pipeline intact
   (this IS the rollback), `true` → canonical service. NEVER delete the old path
   before certification.
8. **Command taxonomy (Lot 6)** — classify commands: A_CONVERSATIONAL_ALIAS (route
   to canonical service with a mapped canonical prompt), B_DETERMINISTIC_COLLECT_PRESENT
   (collect structured surface, canonical service presents), C_SYSTEM_OUTPUT
   (technical, explicitly labelled, never a second persona).
9. **Persona authority (Lot 7)** — single persona module (identity, role, language,
   concision, uncertainty behaviour, recommendation/limitation/approval phrasing,
   fact/hypothesis/recommendation distinction). Forbidden: persona redefinition in
   channel handlers, adapters, formatters, or provider fallbacks.
10. **Honest errors** — map internal router types to contract types; propagate
    `error_code` in sanitized traces; brain unavailable → honest LLM_UNAVAILABLE
    (ok:true clarification), never a fabricated provider answer.

## Pitfalls (session-verified)

### ESM re-export does not create a local binding
```js
// WRONG — other modules can import it, but THIS module gets
// ReferenceError: createGatewayBrainRouter is not defined at runtime
export { createGatewayBrainRouter } from './gateway-brain-router.js';
// RIGHT
import { createGatewayBrainRouter } from './gateway-brain-router.js';
export { createGatewayBrainRouter };
```
`node --check` passes either way; only execution reveals the bug. Run the affected
tests after extracting a shared module.

### Front-router internal types ≠ contract response types
`createCognitiveFrontRouter` returns `response_type` values like
`cloe_conversation`, `memory_boundary`, `deterministic_read_only`,
`supervisor_routed`, `llm_unavailable`. Feeding them into a strict contract
validator fails (`RESPONSE_TYPE_UNSUPPORTED`). Map explicitly:
- `cloe_conversation` → `provider_backed`
- `memory_boundary` / `deterministic_read_only` → `local_read_only`
- `supervisor_routed` → `governance_required`
- `llm_unavailable` / `clarification` → `clarification`
- unknown + ok → `provider_backed`, unknown + !ok → `error`

### "Must not call brainAsk" assertions are wrong for a shared brain boundary
A canonical service LEGITIMATELY calls the shared brain boundary for brain-backed
questions. Asserting `brainCalled === false` fails even when routing is correct.
The right proof of canonical routing: transcript persisted under the canonical
session key with the service's answer (check `session.messages`), and the route /
response_type fields. Only assert absence of brain calls for paths that must NOT
reach the provider (identity, governance-blocked).

### Worktree path artifacts in tests
Tests that resolve the module root via `import.meta.url` (e.g. "temp cwd still
resolves stack services from the canonical module root") fail in a feature
worktree but pass in the canonical checkout. Before declaring a regression,
re-run in the canonical checkout. If green there → worktree artifact, not a code
regression. Document it; do not change production code for it.

### Legacy suite green with flag OFF = rollback proof
The flag-OFF path is the rollback. Run the full legacy suite with flags OFF before
committing the flag-ON path. Both green (legacy OFF + new ON) = reversible
migration; a broken legacy suite means rollback is unproven.

### Session store singleton for stateless handlers
A webhook handler is stateless per invocation; session continuity across messages
needs a memoized store instance (lazy singleton) shared by every invocation, while
still being the SAME `createGatewaySessionStore` used by the other channel (one
canonical store, not a second one).

## Verification checklist

- [ ] Baseline divergences confirmed with file:line evidence BEFORE building
- [ ] Existing canonical components catalogued; compose, don't rebuild
- [ ] Contract has zero channel-implementation dependencies (static test)
- [ ] Shared brain boundary extracted with import+re-export (no ESM cycle)
- [ ] Reference channel migrated first, all its tests green
- [ ] Session identity: stable, deterministic, collision-free, secret-free
- [ ] Flags OFF = legacy green (rollback provable); flags ON = new tests green
- [ ] Honest errors: no fabricated provider answers, typed error_code in traces
- [ ] No persona redefinition in channel handlers/adapters/formatters

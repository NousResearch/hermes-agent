# CLOE Persona Fidelity & Grounded Synthesis (persona-restoration missions)

Established during CLOE_PERSONA_FIDELITY_AND_GROUNDED_SYNTHESIS_REPAIR_V1.
Applies to any mission restoring a conversational persona (LLM-brain) while
keeping retrieval/grounding, or adding a persona authority across channels.

## Canonical persona authority (files vs code)

- The persona can be restored by CANONICAL FILES alone: `SOUL.md`,
  `IDENTITY.md`, `USER.md` in the agent workspace
  (`~/.openclaw/agents/<id>/workspace/`). On a fresh session the OpenClaw
  runtime injects them into the system prompt; the deployed brain then behaves
  with that persona WITHOUT any code change.
- Operator precedent: PR #651 (code candidate) was CLOSED WITHOUT MERGE as
  `SUPERSEDED_BY_CANONICAL_PERSONA_FILES` after a live test proved the files
  suffice. Before writing a persona code fix, run the live fresh-session
  validation (see `references/live-openclaw-agent-validation-pattern.md`) and
  present both options to the operator.
- If a code persona authority IS warranted (cross-channel parity), make it a
  single frozen declarative module (e.g. `src/brain/cloe-persona.js`) with
  render helpers; channels RENDER it, never redefine it. Preserve existing
  identity/blocked strings byte-for-byte so legacy tests don't break.

## Persona doctrine (mandatory traits)

- Uses the operator's first name naturally (e.g. "Cedrick").
- Tutoiement EXCLUSIVE; never "vous"; never "Comment puis-je vous aider
  aujourd'hui ?".
- Familiar, complicit, energetic tone; light humor; direct answers; proposes
  the next move.
- No repetitive self-presentation of role/stack; no interchangeable
  support-assistant identity.
- Reference style (reproduce the STYLE, not a fixed template):
  "Salut Cedrick ! 👋 Toujours en vie, toujours aussi cool. Qu'est-ce qu'on fait aujourd'hui ?"

## Social fast path

- `isSocialQuery()` (broader than greeting): salut/bonjour/ça va/merci/bonne
  nuit/on continue/j'en ai marre/j'ai une idée/qu'est-ce que t'en penses/t'es
  toujours là/on y va... → ZERO retrieval, ZERO tools, ONE persona-complete LLM
  synthesis. Retrieval forbidden for these intents unless the message
  references a personal/project fact.
- Pitfall: `\b` word boundary FAILS after accented chars (`là` ends in U+00E0,
  non-word) — use `(\b|\s|$)` or drop the boundary for patterns ending in an
  accent.
- Social detection must reject long questions (e.g. `length > 80`) and any
  message mentioning campaign/commit/postback/offer/zone/stats — those are real
  questions, not smalltalk.

## Strict factuality (grounded synthesis)

- Inject a FACTUALITÉ STRICTE directive into grounding, targeted-verification
  AND insufficient-retrieval paths: the LLM must never turn a hypothesis,
  ambiguous, partial or ABSENT datum into a fact; on missing data (cap, payout,
  statut, montant) it says so naturally and proposes a targeted verification.
- ANSWER_SUFFICIENT means zero further EXPLORATION, never zero LLM: the final
  response is always LLM-synthesized from retrieval-as-internal-context.
- Runtime source of truth: the comparison/trial harness must override stale
  seeded deployment snapshots with the LIVE deployed commit
  (e.g. `CLOE_LIVE_GIT_COMMIT` read at test time), otherwise the candidate
  answers from an old snapshot as if it were the current runtime.
- Empty candidate responses in a comparison harness: cause is usually the
  harness forcing `tools` + `tool_choice:'auto'` so the real provider calls
  `exec` and returns `content: null`. Fix: social prompts get NO tools;
  factual prompts get `tool_choice:'none'`; reproduce the persona system
  prompt like the real gateway client does.

## Fidelity test suite shape

- Two layers: (1) deterministic persona-directive assertions (stable block /
  system prompt contain Cedrick, tutoiement, banned generic phrases, reference
  style), (2) behavioral fast-path assertions (social prompts → zero retrieval,
  zero tool, one LLM call, `governor.greeting_direct=true`).
- Use double-quoted JS strings in test files to avoid apostrophe escape drift
  (`\` vs `\\'` corruption in single-quoted French text).
- Non-regression: keep the anti-generic-assistant and natural-answers
  invariants ("Do not describe yourself as a generic standalone AI assistant",
  "can answer ordinary questions naturally") alongside the new persona — tests
  pin those exact strings.

## Cross-channel parity: canonical persona context builder (2026-08-02)

When Telegram/another channel must receive the SAME persona as the Center
without channel-specific duplication, build ONE shared
`src/services/cloe/canonical-persona-context-builder.js` that:

- reads SOUL/IDENTITY/USER from `CLOE_PERSONA_WORKSPACE_DIR` (default
  `~/.openclaw/agents/<agent>/workspace`);
- extracts deterministically (no hardcoded persona values — absent file means
  absent field, never invented);
- produces the contract `{ personaId, assistantIdentity, userIdentity,
  relationship, language, addressMode, tone, role, operatingPrinciples,
  conversationRules, sourceRefs, version, hash }`;
- computes `hash = sha256(concat SOUL + IDENTITY + USER)` — the STABLE parity
  proof. Do NOT use the runtime `systemPromptReport.hash` (varies per session
  due to bootstrap + daily memory);
- fails open (`{ok:false, persona:null}`) so legacy fallbacks stay
  byte-for-byte intact for rollback.

Injection points: `buildOpenClawBrainContext` accepts `personaContext` and
appends a `=== CANONICAL PERSONA CONTEXT ===` section to the system prompt +
`grounding.persona_context_injected/persona_hash/persona_version`;
`buildBrainAskResponse` builds the context and surfaces those fields in `data`
(trace-level parity observability); `conversation-router.js` renders the
deterministic identity from the builder (Cloe, tutoiement, user name) with the
legacy hardcoded strings as fallback only.

Deployment trap: if the brain container does NOT mount the agent workspace, the
builder fails open → fix is INERT in production. Add a read-only volume
(`/home/deploy/.openclaw/agents/cloe-poc/workspace:/app/cloe-persona:ro`) +
`CLOE_PERSONA_WORKSPACE_DIR=/app/cloe-persona` to docker-compose and rebuild
before claiming the parity fix is live. Full pattern + baseline harness
method in `references/cloe-persona-context-parity-pattern.md`.

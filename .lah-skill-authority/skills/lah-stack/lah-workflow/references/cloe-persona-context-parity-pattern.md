# Canonical Persona Context Parity (cross-channel persona repair)

Established during CLOE_TELEGRAM_CANONICAL_PERSONA_CONTEXT_PARITY_REPAIR_V1
(2026-08-02). Applies to any mission where two channels (OpenClaw Center /
Gateway and Telegram / webhook) must deliver the SAME conversational persona.

## Root cause pattern

- Center (OpenClaw runtime, npm `openclaw` gateway) injects the agent workspace
  canonical files (SOUL.md / IDENTITY.md / USER.md) into the system prompt at
  session start. Proven via `systemPromptReport.workspaceDir` in the session
  store + live replies citing file-only content.
- Telegram (lah-openclaw-mvp brainAsk path) did NOT load those files. Identity
  came from hardcoded modules: `stable-block.js` (`OPENCLAUD_IDENTITY`, no
  tutoiement rule, no user name) and `cloe-persona.js`/`conversation-router.js`
  (rendered "Comment puis-je vous aider ?" — the vouvoiement defect).
- Symptom: occasional "vous" on Telegram despite the canonical tutoiement rule.

## Architecture: one shared builder, channels render it

Create `src/services/cloe/canonical-persona-context-builder.js`:

- Reads SOUL/IDENTITY/USER from `CLOE_PERSONA_WORKSPACE_DIR` (default
  `~/.openclaw/agents/<agent>/workspace`).
- Extracts deterministically (no hardcoded persona values): name, creature,
  vibe, emoji, preferred name, language, address mode, tone, role, operating
  principles, conversation rules, sourceRefs.
- Produces the contract: `{ personaId, assistantIdentity, userIdentity,
  relationship, language, addressMode, tone, role, operatingPrinciples,
  conversationRules, sourceRefs, version, hash }`.
- `hash = sha256(concat SOUL.md + IDENTITY.md + USER.md)` — STABLE across
  sessions and channels (parity proof). Do NOT use the runtime
  systemPromptReport.hash (varies per session due to bootstrap + daily memory).
- Fail-open: missing files → `{ ok:false, persona:null }`; callers keep legacy
  fallback unchanged (byte-for-byte) for rollback.
- No channel dependency, no provider calls, no writes, no secrets.

Wiring:
- `openclaw-brain-context-builder.js`: accept `personaContext` param; inject a
  `=== CANONICAL PERSONA CONTEXT ===` section into the system prompt; expose
  `grounding.persona_context_injected / persona_hash / persona_version`.
- `readonly-operator-cli-client.js` `buildBrainAskResponse`: build the persona
  context and pass it; surface persona fields in the response `data` so traces
  prove parity (LOT 11 observability).
- `conversation-router.js`: render the deterministic identity from the builder
  (Cloe, tutoiement, user name), keep legacy `renderCloeIdentity()` as fallback.

## Deployment requirement (critical)

The container running the brain path may NOT mount the agent workspace. The
builder then fails open → behavior unchanged → the parity fix is INERT in
production. To activate: add a read-only volume
(`/home/deploy/.openclaw/agents/cloe-poc/workspace:/app/cloe-persona:ro`) and
`CLOE_PERSONA_WORKSPACE_DIR=/app/cloe-persona` to docker-compose, then rebuild
from the new HEAD. Without this, do NOT claim the fix is live.

## Baseline proof method (LOT 0)

- Center: fresh session via `openclaw terminal --local --session
  'agent:<id>:<fresh-key>' --message '<question>'` — see
  `live-openclaw-agent-validation-pattern.md` for pty capture and extraction.
- Telegram: operator requires the FULL `handleTelegramWebhook()` harness
  (production env, real flags, real session store, real brainAsk binding) — NOT
  a direct POST to /chat/completions and NOT an isolated `buildBrainAskResponse`
  call (those bypass the layers being compared). Mock fetch ONLY for
  `api.telegram.org` (capture outbound text, never send), real fetch for the
  provider. Run inside the container via `docker cp` + `docker exec node`.
  Read secrets from env in-process; never print tokens/chat IDs (hash the
  session key).
- Sanitize replies (redact `bot\d+:...`, long digit runs).
- Compare: persona authority, user identity, language, address mode, role,
  context sources, prompt layers, provider/model (must stay identical).

## Pitfalls

- `git ls-tree <ref> -- <path>` exits 0 even when path absent — use
  `git cat-file -e <ref>:<path>` for existence.
- Re-run file inventory AFTER `git rebase origin-https/main`; a pre-rebase scan
  against a stale base reports files as missing.
- Workspace clone `origin` (plain SSH) can be stale; branch from
  `origin-https/main` and verify `git merge-base --is-ancestor <PROD_SHA>
  origin-https/main`.
- Tests pinning the old hardcoded persona strings ("cockpit conversationnel",
  "Comment puis-je vous aider ?") must be updated deliberately — they pin the
  defect, not a stable contract. Update them as part of the fix, don't leave
  them failing.
- The canonical persona builder must NEVER invent values when a file is absent
  (missing USER.md → userIdentity absent, not guessed "Cedrick").

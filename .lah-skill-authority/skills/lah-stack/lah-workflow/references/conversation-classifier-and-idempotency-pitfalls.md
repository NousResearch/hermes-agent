# Conversation classifier, idempotency, and baseline-proof pitfalls

Established during CLOE_CAMPAIGN_MEMORY_LOT_1 (2026-08-04). Three pitfalls that
cost real debugging time and are not obvious from the lah-workflow traps table.

## 1. Bare domain keyword in classifier/mutation list over-captures unrelated prompts

Adding a read-only intent for a domain whose bare term is ALSO in the mutation
keyword list causes two distinct failure modes:

- **Placement**: the read-only check must come BEFORE the mutation keyword check.
  The legacy `classifyReadonlyConversationIntent` mutation list contained the bare
  word `campaign`; any read-only question ("Que sais-tu de la campagne X ?") was
  classified `mutating` and blocked. Placing `campaign_memory` detection after the
  mutation check keeps the bug.
- **Bare-word over-capture**: a pattern `/\b(campaign|campagne)s?\b/` hijacks
  unrelated prompts. Real regression caught by `gateway-readonly-adapter.test.js`:
  `"Find affiliate angle for this campaign."` was routed to campaign_memory instead
  of brainAsk, breaking `gateway affiliate prompts reach brainAsk` (provider_backed).

Working rule (used in `src/cognitive/intent-classifier.js` and
`src/services/gateway/readonly-conversation-router.js`):

```js
const hasCampaignIdentity = hasAny(normalized, [/\bCAMP-/i, /\bcampaign memory\b/, /\bm[ée]moire de campagne\b/]);
const hasCampaignTerm = hasAny(normalized, [/\b(campaign|campagne)s?\b/]);
const hasMemoryContext = hasAny(normalized, [/\b(m[ée]moire|memory|history|historique|r[ée]sum|summar|performe|worked|fonctionn|failed|[ée]chou|winning|gagnan|creative|cr[ée]ativ|conversion|zone|conflit|conflict|stale|p[ée]rim|fra[îi]ch|reuse|r[ée]utilis)\w*/i]);
// campaign intent iff: explicit identity (CAMP-) OR (campaign term AND memory context)
// AND no mutation verb (delete/launch/create/play/pause/...)
```

Re-run the router/adapter test batteries (gateway-readonly-conversation-router,
cognitive-front-router, gateway-readonly-adapter, cloe-context-wiring) after any
classifier edit — a single hijacked prompt shows as exactly one regression.

## 2. Idempotency identity must EXCLUDE payload_hash

`deriveIdempotencyKey()` including `payload_hash` in the identity breaks conflict
detection: a same-identity/different-payload retry produces a DIFFERENT key, so the
second write is `ACCEPTED` instead of `CONFLICT_DETECTED`. Spec §7.B requires
"same identity + different normalized payload → CONFLICT_DETECTED, both traceable".

Correct design:

```js
// identity from STABLE fields only
const parts = [provider, provider_account, event_type, source_record_id, campaign_id, event_timestamp];
// payload_hash compared SEPARATELY:
//   same key + same hash  → IDEMPOTENT_REPLAY (no new file)
//   same key + diff hash  → CONFLICT_DETECTED (new file marked conflicted,
//                            conflicts_with_record_ids=[existing], no overwrite)
```

When a RED test expects CONFLICT_DETECTED but gets ACCEPTED, check whether the
hash is part of the identity before debugging the writer.

## 3. Proving pre-existing test failures (baseline-worktree technique)

"Pre-existing" must be proven, not claimed — a changed module can add exactly ONE
regression hidden inside a 20+ failure baseline. In this session
`cloe-cognitive-phase1.test.js` failed (`identity.name` expected `OpenClaw`, actual
`Chloé`) on main BEFORE any Lot 1 change; the gateway router stack had 21 baseline
failures. Naively attributing them to the diff would have polluted the PR.

Protocol:

```bash
# 1. Clean baseline worktree at the SAME base SHA (the mission's origin/main)
git -C /home/deploy/lah-stack-repos/openclaw-runtime \
  worktree add /tmp/cloe-cm-baseline-verify 5d6e9cb5fdcb8b67132ec66c58915a09a1d17c40
cd /tmp/cloe-cm-baseline-verify/lah-openclaw-mvp && npm ci --no-audit --no-fund

# 2. Run the SAME test globs in BOTH trees; capture sorted failing names
node --test --test-concurrency=1 <globs> 2>&1 | grep '^not ok' | sed 's/^not ok [0-9]* - //' | sort > /tmp/base-fails.txt
# ... same in the worktree → /tmp/mine-fails.txt

# 3. Isolate regressions: in mine but NOT in baseline
comm -23 /tmp/mine-fails.txt /tmp/base-fails.txt   # ← these are YOUR regressions
comm -13 /tmp/mine-fails.txt /tmp/base-fails.txt   # fixed-by-chance (informational)

# 4. Attribute pre-existing failures via git log on the module under the assertion
git log --oneline -3 -- lah-openclaw-mvp/src/cognitive/stable-block.js

# 5. Cleanup
git -C /home/deploy/lah-stack-repos/openclaw-runtime worktree remove --force /tmp/cloe-cm-baseline-verify
```

Gotchas: run the SAME file set in both trees (comparing different globs is
meaningless); the temp worktree shares the canonical `.git` so dirty-canonical
noise does not affect a detached-HEAD baseline; always `--force` the removal or the
stale entry pollutes `git worktree list`.

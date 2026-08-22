# CLOE Conversational Soul Restore — Grounded Fast Retrieval Pattern

Mission: `CLOE_CONVERSATIONAL_SOUL_RESTORE_WITH_GROUNDED_FAST_RETRIEVAL_V1` (2026-08-01).
The LLM must remain Chloé's brain; the runtime assists, never replaces it. Retrieval is a hidden
navigation map — internal context only, never the user-facing answer.

## The bug class this repairs

Deployed runtime `2855674` (merge PR #649 = `b582b08` "retrieval-first tool budget") contained the
**forbidden flow** in `lah-openclaw-mvp/src/services/chat-completions-service.js`:

```
user message → retrieval → direct registry response → user      (WRONG)
```

- `composeAnswerFromRetrieval()` (l.348) built the answer verbatim from registry evidence with the
  robotic header `Réponse basée sur la connaissance indexée (retrieval-first) :` and exposed layer
  names (EXACT_REGISTRY_FTS, CAMPAIGN_MEMORY...).
- `buildNativeChatCompletions()` (l.571) returned that dump for `ANSWER_SUFFICIENT && retrieval.ok`
  WITHOUT calling the provider — zero LLM.
- `AUTHORITATIVE_SOURCE_MISSING` (l.604) returned a canned runtime disclosure, also bypassing the LLM.

## The fix (Phase 1 + 5)

- `ANSWER_SUFFICIENT` means **zero further exploration, never zero LLM**. Inject
  `buildRetrievalGroundingMessage(retrieval, query)` as a SYSTEM message (internal context, with a
  directive: don't cite layers, don't say "connaissance indexée" unless the user asks for a technical
  diagnostic), strip tools, force one LLM synthesis.
- `AUTHORITATIVE_SOURCE_MISSING` → inject `HONESTY POLICY (conversational)` system directive; the LLM
  discloses honestly. Provider bypass is forbidden except deterministic technical endpoints or
  provider-failure fallback (timeout → `buildDeterministicFallback` in natural French with explicit
  limitations).
- Delete the registry write of the raw user query (`indexer.applyEvent('TARGETED_VERIFICATION', {fact_key: retrieval-...})`)
  — that was the conversational-contamination vector: "Ta réponse est hors sujet" / "salut Chloé"
  would be indexed as facts.

## Phase 4 — provenance classes + hygiene

- `PROVENANCE_CLASSES`: AUTHORITATIVE_FACT, OPERATOR_DECISION, CAMPAIGN_EVENT, SYSTEM_STATUS,
  CONVERSATIONAL_EPHEMERA. Ephemera must NEVER satisfy a factual query.
- Schema: add `provenance_class TEXT NOT NULL DEFAULT 'AUTHORITATIVE_FACT'` to `entities` +
  **additive migration** for existing DBs:
  ```js
  try { db.exec("ALTER TABLE entities ADD COLUMN provenance_class TEXT NOT NULL DEFAULT 'AUTHORITATIVE_FACT'"); } catch {}
  ```
  (both sqlite and json backends — declare the const at module level, NOT inside one backend, or the
  other backend throws `ReferenceError: CONVERSATIONAL_EPHEMERA is not defined`).
- Read filter: `getByExact/getByPrefix/ftsSearch(…, {includeEphemera=false})` exclude ephemera.
- Write guard in `incremental-indexer.js`: `looksConversationalEphemera(factKey, summary)` rejects
  legacy slugified queries (`retrieval-` prefix, `answered from retrieval` summary, `ta réponse`,
  `reprends le sujet`, `salut`, `bonjour`...) with `CONVERSATIONAL_EPHEMERA_REJECTED`.

## Trap: dispatcher swallows handler rejections

`applyEvent()` called `handler(payload)` and always returned `{ok:true}` — the rejection was lost.
Propagate structured results:
```js
const result = handler(payload);
if (result && typeof result === 'object' && 'ok' in result) {
  return { ok: result.ok, event_type, reason: result.reason ?? null, stable_id: result.stable_id ?? null };
}
```

## Phase 6 — entity relevance (exclude, not rank lower)

`filterEvidenceForFactualContext(evidence, {query, entityHints})` + `extractEntityLabels(query)`:
- drop CONVERSATIONAL_EPHEMERA evidence;
- when the query names an entity (ourdream, c99, offer NNNNN), drop evidence whose text references a
  DIFFERENT known entity label; keep entity-agnostic evidence. Applied inside
  `classifyRetrievalDecision` before deciding ANSWER_SUFFICIENT.

## Phase 5 — persona + greeting

- `stable-block.js` OPENCLAUD_IDENTITY: name `Chloé`, warm/direct French persona instruction; keep the
  substring "ordinary questions naturally" — a brain-context-builder test pins it.
- `openclaw-brain-context-builder.js` system prompt rules 12–13: natural conversation, clarify,
  recommend, challenge; never expose HOT_CURRENT_STATE/EXACT_REGISTRY/EXACT_REGISTRY_FTS/
  CAMPAIGN_MEMORY/ALIASES, never open with "Réponse basée sur la connaissance indexée".
- Greeting: `isGreetingQuery()` (short standalone greetings, <60 chars) → skip retrieval entirely,
  one LLM synthesis, zero tools, no metadata. **Compatibility trap**: only strip tools /
  force `tool_choice:'none'` when tools were actually supplied; when none were supplied leave
  tool_choice undefined, or the adapter test "does not add tools when none supplied" breaks
  (expects `undefined`, not `'none'`).
- `readonly-conversation-router.js` cloe_assistant: remove the `evidence:` technical lines
  (direct_mutation_enabled, operator_summary_verdict, v5_decision_context_injected) from the
  user-visible answer — keep them in metadata only.

## Classifier hint trap

`authoritativeAbsenceHints` must not contain common French phrases that appear in ordinary factual
questions. `en production` misclassified "Quelle est la latence du pipeline de landing en production ?"
as AUTHORITATIVE_SOURCE_MISSING (→ HONESTY POLICY) instead of RETRIEVAL_INSUFFICIENT (→ bounded
2-reads directive). Removed it; keep specific hints (dashboard|login|authentifi|est-il activ|live status).

## Test strategy when a mission changes a behavioral contract

- Existing tests pin the FORBIDDEN behavior: `retrieval-first-tool-budget.test.js` asserted
  `providerCalled === false` on ANSWER_SUFFICIENT, `AUTHORITATIVE_SOURCE_MISSING` without provider,
  and English fallback text. These pins ARE the acceptance change — invert them deliberately with a
  comment citing the mission phase.
- Baseline pre-existing failures with a reference worktree BEFORE assuming regression:
  ```bash
  git worktree add /tmp/ref origin/main && cd /tmp/ref/lah-openclaw-mvp && npm ci
  node --test --test-concurrency=1 <same suites>   # compare counts
  ```
  Example: `readonly-conversation-router-stack.test.js` failed 3/17 on origin/main already (pre-existing).
- Mock-fetch capture: provider is called `fetchImpl(url, options)` — capture body with
  `(_url, options) => JSON.parse(options.body)`, NOT `({body})`.

## Phase 0 — production safety (operator gate)

- The mission said "restore the rollback image if the current runtime still answers directly from
  retrieval", but the operator BLOCKED both `.env`/compose inspection AND `docker stop`+`rm`.
  **Mission sanction ≠ operator authorization for mutating a running production container.** Prepare
  everything without touching the container; defer the physical swap to the explicit operator gate.
- Preserve evidence: tag the current image `lah-openclaw-mvp:current-<sha>-<feature>` (non-destructive),
  keep the rollback tag (`rollback-302ed75-20260801-171352`) intact, restore any temporary `latest`
  retag afterwards so zero drift remains.
- Prove the deployed flow from CODE: find the deployed commit lineage (image build time → merge commit
  `2855674` → `b582b08`), grep the exact symbol in `origin/main`, not from a dirty working tree.

## Branch-base discipline

Workspace clone sat on `feat/cloe-instant-retrieval-metrics-v1` @ 302ed75 while the deployed runtime
was the later merge 2855674; the canonical working tree (155 dirty files) held a non-committed variant.
Before implementing: `git fetch origin main`, create mission branch from the DEPLOYED lineage
(`git checkout -b <mission> origin/main`), verify the buggy symbol exists in new HEAD. Never implement
on a dirty canonical working tree.

## Phase 7 — required acceptance (8 tests, `test/cloe-conversational-soul-restore.test.js`)

T1 "salut Chloé" → natural greeting, one LLM, zero retrieval, zero tool, no metadata.
T2 known OurDream question → facts retrieved, zero repo exploration, one grounded LLM synthesis, no C99.
T3 recommendation → retrieval supplies facts, LLM reasons, not a factual template.
T4 ambiguous → natural clarification, no uncontrolled exploration.
T5 unknown fact → at most two targeted reads, LLM summarizes confirmed + unresolved.
T6 same question twice → second uses memory, zero exploration, one synthesis, lower latency.
T7 provider timeout → deterministic factual fallback, explicit limitations, no mutation.
T8 contamination → indexed criticism never retrieved as factual context.

## Test-37 double-call root cause: dotenv.config() at module import

`test/brain-context-builder.test.js:37` ("exactly one DeepSeek call") got 2 calls in the workspace but 1 in a clean reference worktree. Root cause: `src/server.js:1-2` does `dotenv.config()` at import; any test file importing `createApp` loads the workspace `.env` into `process.env` (notably `CARTELOGIC_MEMORY_URL=http://cartelogic-remote-memory:8741`). `buildBrainAskResponse` then runs its CarteLogic memory probe (fetch #1) before the DeepSeek call (fetch #2). The clean worktree has no `.env`, so the probe early-returns and only DeepSeek is counted.

Fix (do NOT relax the 1-call assertion): sandbox the env passed to the function under test:
```js
const sandboxEnv = { ...process.env, CARTELOGIC_MEMORY_URL: '', CARTELOGIC_OPERATOR_API_KEY: '' };
await buildBrainAskResponse({ env: sandboxEnv, prompt: 'Qui es-tu ?', fetchImpl });
assert.equal(capturedPayloads.length, 1); // stays strict
```
Diagnosis recipe: one-off `node -e` importing ONLY `readonly-operator-cli-client.js` → 1 call; adding `import('./src/server.js')` first → 2 calls. Probe URL in captured calls (`http://cartelogic-remote-memory:8741/cartelogic/v1/memory?…`) identifies it as the memory probe, not a second provider call.

## Phase 0 execution when the operator gate opens (rollback swap)

Once authorized, the physical swap splits into guard-safe steps (the Hermes terminal tool rejects foreground `docker compose up -d` as a long-lived server, and `docker stop/rm` may prompt for approval):
```bash
# BEFORE evidence — never skip
docker inspect <c> --format 'container_id={{.Id}} image_id={{.Image}} restart_count={{.RestartCount}} status={{.State.Status}}'
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:4000/health
docker inspect <c> --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^(GIT_COMMIT|EXOCLICK_LIVE_ENABLED)='   # mask secret-bearing vars
# 1) retag rollback as latest (safe, no approval)
docker tag lah-openclaw-mvp:<rollback-tag> lah-openclaw-mvp-lah-openclaw-mvp:latest
# 2) stop+rm (may need approval)
docker stop <c> && docker rm <c>
# 3) bring up via compose with background=true + notify_on_complete
docker compose up -d --no-deps <service>
# AFTER verification: image ID == rollback ID, restart_count 0, /health 200,
# GIT_COMMIT == full expected SHA, EXOCLICK_LIVE_ENABLED=false, then a real greeting probe.
```
Verify the conversational answer against the LIVE container by reading the admin key from the container env WITHOUT printing it: `docker inspect <c> --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^ADMIN_API_KEY=*** | cut -d= -f2-`, then `curl -X POST /chat/completions -H "Authorization: Bearer $KEY"`. In a comparison harness, prefer `process.env.CLOE_ADMIN_API_KEY` over argv (argv leaks into /proc) and run with `node --env-file=.env`.

## Untracked files block git pull into a degraded canonical checkout

A canonical checkout with residual dirty state (files tracked in origin/main but untracked locally) makes `git pull` abort with "Please move or remove them before you merge" — and `git stash` does NOT take untracked files. Sequence that works:
1. `git stash push -m "<label>"` (captures tracked modifications).
2. `git status --porcelain | grep '^??'` → identify files the merge will create (under src/ and test/).
3. Preserve-move them aside: `mv <file> /tmp/<preserve>/`.
4. `git pull origin main`.
5. For a broadly degraded checkout (35+ tracked mods + 20+ untracked source files), do NOT trust it post-merge — verify in a FRESH worktree: `git worktree add /tmp/<verify> origin/main && cd /tmp/<verify>/<subdir> && npm ci && node --test --test-concurrency=1 <bounded-globs>`, then `git worktree remove --force /tmp/<verify>`.
6. Restore preserved files; keep the stash labeled. Never `git reset --hard` the operator's residual state without an explicit request.

## Post-merge when GitHub Actions is unavailable

If GHA is repo-wide unavailable, required checks stay PENDING forever (`gh run list --branch main` shows runs with no conclusion, no successes) and block `gh pr merge` despite green local tests. Protocol: document the dead check in the PR with evidence, get operator authorization via `clarify`, then merge with the guard chain:
```bash
gh pr merge --help | grep -c match-head                     # flag supported?
gh pr view <PR> --json headRefOid -q .headRefOid            # head unchanged?
gh pr merge <PR> --admin --merge --subject "..." --match-head-commit <full-sha>
```
Then verify from a fresh worktree of origin/main (see above) — that is ground truth, not the dirty canonical checkout.

# Null-dir / default-override defect class (campaign-memory + siblings)

## The defect class
Optional-path/override arguments passed as `null` silently OVERRIDE the callee's default when
the default only applies to `undefined`. In JS:

```js
function f(opts = {}) { const { dir = DEFAULT } = opts; ... }
f({ dir: null })  // dir === null, NOT DEFAULT
```

Symptom: `existsSync(null)` returns false (no throw) → "store does not exist yet" while the
store actually has data.

## Concrete instance — CLOE Campaign Memory (Lot 1 finalization, 2026-08-04)
- Reader: `queryCampaignMemory({ memoryEventsDir = DEFAULT_MEMORY_EVENTS_DIR })`; missing dir →
  `buildResult('empty', 'Campaign memory store does not exist yet')`.
- **Surface 1 (FIXED by PR #684 / f30eaea)**: HTTP route `campaign-memory.routes.js` →
  `queryOptions()` does `if (memoryDir) options.memoryEventsDir = memoryDir;` — unset → reader default.
- **Surface 2 (MISSED by the fix PR, found at runtime)**: conversational handler
  `handleCampaignMemoryPrompt` in `src/services/gateway/readonly-conversation-router.js` still did
  `queryCampaignMemory({ query, memoryEventsDir: env?.CLOE_CAMPAIGN_MEMORY_DIR || null })` → null →
  "store does not exist yet" while HTTP route AND canonical reader both saw 35 events.
- Fix (same pattern as PR #684):
  ```js
  const retrievalOptions = { query };
  if (memoryDir) retrievalOptions.memoryEventsDir = memoryDir;
  const retrieval = queryCampaignMemory(retrievalOptions);
  ```

## How the runtime proof exposed it (why unit tests missed it)
- `/brain/ask` AND `/chat/completions` (non-tools path) BOTH go through `buildBrainAskResponse`,
  whose narrow `isLocalReadOnlyStackPrompt` allow-list does NOT include campaign questions → those
  HTTP endpoints route campaign questions to the provider (`uses_external_llm: true`). That is
  expected pre-existing behavior, NOT the defect — but it is a trap when testing "the conversational path".
- The REAL conversational path (what Chloé uses): gateway/telegram →
  `createCloeCanonicalConversationService` → `createCognitiveFrontRouter` →
  `readOnlyRouter.route()` (**async!**) → intent `campaign_memory` → `provider_used: false`.
- Diagnostic that found the defect: run the canonical service directly inside the container with a
  `.mjs` script (docker cp + `node /tmp/x.mjs`):
  ```js
  import { createCloeCanonicalConversationService } from '/app/src/services/cloe-canonical-conversation-service.js';
  const svc = createCloeCanonicalConversationService({ env: {}, fetchImpl: globalThis.fetch, persistTranscript: false });
  const r = await svc.respond({ message: 'campaign memory CAMP-CONV', sessionKey: 'x', channel: 'Gateway', entrypoint: 'gateway' });
  ```
  Result was `route: memory_boundary`, `provider: null`, `provider_used: false` — correct routing,
  but empty "store does not exist yet" → handler-level null-dir bug confirmed.

## Second defect found during RED: case-preserving identifier extraction
- `interpretCampaignQuery` lowercased the prompt (`normalized`), then
  `/CAMP[-]([a-z0-9_-]+)/i` captured only the suffix → `campaign_id: "conv"` for `CAMP-CONV`
  (prefix lost) → store filter never matched the stored `CAMP-CONV`.
- A legacy shorthand in `createCampaignFilters` (`tagValue.startsWith('CAMP-') && tagValue.slice(5) === cid`)
  masked it in older tests (query '001' matched tag 'CAMP-001').
- Fix: extract from the ORIGINAL prompt (case preserved), capture the full token:
  ```js
  const original = String(userPrompt ?? '');
  const campMatch = original.match(/\bCAMP[-][A-Za-z0-9_-]+\b/);
  if (campMatch && !query.campaign_id) query.campaign_id = campMatch[0];
  // same for OFFER-
  ```
- Lesson: a legacy-shorthand filter branch can hide extraction bugs by making the WRONG value still match.

## RED test pattern (deterministic, self-cleaning)
- Write one event into the reader's DEFAULT dir, derived the same way the reader does (NOT `process.cwd()`):
  `join(dirname(fileURLToPath(import.meta.url)), '..', 'data', 'memory-events')`.
- Call `createReadonlyConversationRouter({ env: {} })` then `await router.route({ prompt, sessionKey })`
  — `route()` is **async**; forgetting the await yields `intent: undefined`.
- Assert answer includes the campaign ID and NOT "does not exist yet"; clean up in `finally { rmSync(...) }`.
- ESM test files have no `__dirname` — use `fileURLToPath(import.meta.url)`.

## Deployment verification without ephemeral containers (operator preference)
- `docker run --rm --entrypoint sh <image> -c 'echo $GIT_COMMIT'` creates an ephemeral container —
  the operator may BLOCK it. Verify statically and read-only:
  `docker image inspect <image> --format '{{json .Config.Env}}'`.
- Confirm rollback availability: `docker image inspect sha256:<previous-image-id>` still exists.
- Test side-effects (legacy writer tests write into the default store dir) → archive with `mv` to
  `/tmp/cloe-cm-test-artifacts/` — the operator blocks `rm -rf` even in /tmp.

## Validation evidence (fix branch 2ad9a39, 2026-08-04)
- 91/91 campaign-memory suites (memory + routes + lot1-repair + operator-scenarios)
- 31/31 lot1-repair, 105/105 conversational suites (canonical-conversation + campaign-creation + brain-context)
- E2E 20/21 PASS, 1 REVIEW (D1) — verified pre-existing (identical without the fix via git stash)
- `node --check` + `git diff --check` clean

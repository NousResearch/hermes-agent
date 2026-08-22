# Live Activation & Runtime Certification Pattern (deploy a merged commit + certify in real)

Established during CLOE_RETRIEVAL_FIRST_TOOL_BUDGET_AND_PROGRESSIVE_REPLY_LIVE_ACTIVATION_V1 (2026-08-01): operator authorized deploying merge commit 2855674 into the canonical lah-openclaw-mvp container and running live validation A-F against OpenClaw Control `/chat/completions`.

Companion to `docker-compose-safe-deployment-pattern.md` (same mission family). This file adds the commit-exactness, runtime-certification, and retrieval-first validation specifics.

## 1. Preflight — the canonical checkout may be dirty and OLDER than the merge

The canonical checkout (`/home/deploy/lah-stack-repos/openclaw-runtime`) was on the feature branch (0458509) with 174 dirty files (13 modified + 161 untracked) whose content was an EARLIER state than the target merge 2855674 (e.g. budget/duplicate precedence order inverted). `git diff 2855674 -- <src files>` was non-empty.

**Do NOT build from the dirty canonical checkout** — `COPY . .` would ship the dirty tree AND the local `node_modules/` (which existed in the checkout) into the image.

**Do NOT reset/clean it either** — it carries unrelated in-progress work.

**Fix — build from a clean detached worktree at the exact commit:**
```bash
git worktree add --detach /tmp/<mission>-<sha> 2855674b7af84fd4793e863d5a64d40bfb55406c
cd /tmp/<mission>-<sha>/lah-openclaw-mvp
# verify: git status --short | wc -l  → 0 ; node_modules ABSENT (clean)
GIT_COMMIT=<full-sha> docker compose build --build-arg GIT_COMMIT=<full-sha> lah-openclaw-mvp
```
Then deploy from the canonical checkout (where `.env` and `data/` live) with the ALREADY-BUILT image:
```bash
cd /home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp
docker stop lah-openclaw-mvp && docker rm lah-openclaw-mvp
docker compose up -d --no-deps --no-build lah-openclaw-mvp   # --no-build: reuse tagged image
```
The compose tag is `lah-openclaw-mvp-lah-openclaw-mvp:latest` regardless of build dir (project name = dir basename), so the image built in /tmp is picked up. `--no-build` prevents compose from rebuilding from the dirty canonical context.

Cleanup after: `git worktree remove --force /tmp/<mission>-<sha>`.

## 2. Exact-SHA proof — check image env AND container env (they can differ!)

`env_file: .env` in compose OVERRIDES Dockerfile `ENV` at runtime. A stale `GIT_COMMIT=` line in `.env` (observed: `5d21a47b`, an old unrelated commit #471) made the CONTAINER report the wrong commit while the IMAGE was correct.

Diagnosis flow:
```bash
docker run --rm <image-tag> sh -c 'echo "IMAGE_GIT_COMMIT=$GIT_COMMIT"'   # proves build
docker exec <container> sh -c 'echo "CONTAINER_GIT_COMMIT=$GIT_COMMIT"'   # proves runtime
grep -n "^GIT_COMMIT" .env                                                 # stale line source
```
Fix (config-only, no code): `cp .env .env.bak-<date>-<sha>`, `sed -i 's|^GIT_COMMIT=.*|GIT_COMMIT=<full-sha>|' .env`, recreate container. Verify both again.

## 3. FastSafe pre-deploy — grep hits are usually static, verify each

The 15 checks fire false positives on this codebase: `sk-[...]` regex matched the phrase `risk free profit` in a content filter; `TRUNCATE` matched `...`; `affiliate` matched the data field `affiliate_url`. Rule: grep gives CANDIDATES, always inspect the actual line for each hit before flagging. The live-gate check that matters: `docker exec <c> sh -c 'echo $EXOCLICK_LIVE_ENABLED'` must be `false`, AUTOCUT unset.

## 4. Rollback — tag BEFORE rebuild (tag-overwrite trap)

```bash
docker tag e567cb1fd3ed... lah-openclaw-mvp:rollback-<shortsha>-$(date +%Y%m%d-%H%M%S)
```
Old image `e567cb1fd3ed` (runtime 302ed75) preserved as `lah-openclaw-mvp:rollback-302ed75-20260801-171352` before the new build untagged `:latest`.

## 5. Knowledge registry seeding for retrieval-first live validation

The KnowledgeCollector registry (`data/knowledge/lah-knowledge-registry.sqlite` → JSON fallback, since the runtime is Node 20 and `node:sqlite` is Node 22+) is EMPTY at boot; only tests seed it. For scenarios that need "question connue et indexée" (A/D), seed via a temp script in the container:

```bash
docker cp /tmp/seed-registry.mjs lah-openclaw-mvp:/tmp/seed-registry.mjs
docker exec -w /app lah-openclaw-mvp node /tmp/seed-registry.mjs
docker exec lah-openclaw-mvp rm -f /tmp/seed-registry.mjs
```
Script imports `/app/src/knowledge/lah-knowledge-registry.js` + `pilot-corpus.js`, creates registry at `resolve(process.cwd(),'data','knowledge','lah-knowledge-registry.sqlite')`, calls `seedPilotCorpus` (~22 events incl. `ourdream-postback-active`). JSON backend reloads on mtime change (freshness check), so no restart needed. This is a LOCAL knowledge index write, not a mem0/Qdrant/Campaign Memory mutation — FastSafe-compliant.

## 6. Retrieval-first validation traps (A-F live scenarios)

- **`ANSWER_SUFFICIENT` short-circuits BEFORE the provider** — zero provider calls, zero tools, ~10-70ms. Proving "zéro provider" = decision `ANSWER_SUFFICIENT` + fast latency + `retrieval_authority` set. To exercise the provider path you need a question with ZERO lexical overlap with the seeded registry (a "Neptune temperature" style question) — and note the registry SELF-ENRICHES (Gate 9 indexes every ANSWER_SUFFICIENT query as a TARGETED_VERIFICATION), so a question that was "not indexed" at start may resolve by the end. Run not-indexed probes BEFORE the indexed ones, or use a fully orthogonal query.
- **Replay of the 14-activity OurDream case**: pass the accumulated history (7 tool_calls + 7 results = 14 activities) in ONE call; governor counts history tool_calls → `budget_exceeded=true` + `TOOL_SEARCH_DUPLICATE_SKIPPED` → forced synthesis with 0 NEW tool calls (observed: 14 before → 0 after, 37ms).
- **Timeout fallback proof**: spec allows "réel OU simulation au niveau runtime". Simulate inside the deployed container: import the REAL `/app/src/services/chat-completions-service.js` modules, pass a `fetchImpl` that hangs (setTimeout 300ms, throws) with `timeoutMs: 60`, feed history with tool receipts. Assert: `ok=true`, `_cloe.timeout_fallback=true`, content includes confirmed facts ("landing pipeline latency measured at 120ms"), "Unresolved items", "no action was taken", and NEVER equals "The agent run failed before producing a reply." Observed 75ms.
- **SSE streaming proof**: real streaming = 4 distinct chunks (checkpoint `_cloe_checkpoint` BEFORE answer, role chunk, content chunk, finish chunk) + `data: [DONE]`, content-type `text/event-stream`. A single buffered chunk = not real streaming. Checkpoint text confirmed: "I confirmed that the indexed knowledge is sufficient…". Note: WebUI visual rendering may not be re-testable from CLI — provide protocol-equivalent evidence and document it as a limitation.

## 7. Admin key handling for live probes (never print)

```bash
docker exec lah-openclaw-mvp sh -c 'echo -n "$ADMIN_API_KEY"' > /tmp/.cloe-adminkey
chmod 600 /tmp/.cloe-adminkey
KEY_FILE=/tmp/.cloe-adminkey node /tmp/validate-real.mjs   # script reads key from file
rm -f /tmp/.cloe-adminkey
```
Scripts POST `http://127.0.0.1:4000/chat/completions` with `x-admin-api-key` header, `model:'brain'`, `messages`, `tools` (non-empty → native tool path), optional `stream:true`. Parse `_cloe.governor` from response JSON.

## 8. Post-deploy mutation audit

`docker logs <c> --since <deploy-iso> 2>&1 | grep -icE "CAMPAIGN_PAUSE_SENT|CAMPAIGN_PLAY_SENT|live_sent.*true|provider.*mutation|execution_receipt.*created"` must be 0; `grep -icE "campaign.*created"` must be 0; EXOCLICK_LIVE_ENABLED still false; health 200; RestartCount 0.

## Verdict identifier (continuity JSON must match mission contract)

`CLOE_RETRIEVAL_FIRST_TOOL_BUDGET_AND_PROGRESSIVE_REPLY_LIVE_ACTIVATION_V1_CERTIFIED` — validate against the mission's allowed list at Gate 11 (see `continuity-json-schema-pitfalls.md`).

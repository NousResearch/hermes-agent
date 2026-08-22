# Worktree residual-dirty + shadow-comparison pitfalls (CLOE_TELEGRAM_CANONICAL_PIPELINE_PRODUCTION_CANARY_V1)

Session evidence: 2026-08-01/02, openclaw-runtime worktree `cloe-telegram-gateway-behavior-parity-v1`
(base 0458509, later merged with PR #650's main at 53641a8). Three distinct traps, each cost
real debugging time. All three are CLASS-LEVEL — they will recur on any worktree-based LAH mission.

## 1. Worktree residual dirty files from a LATER-merged PR break grouped test runs

### Symptom
A worktree created from an OLDER base (e.g. 0458509, before PR #650 SOUL_RESTORE merged)
carries **residual dirty files** in the working tree. Their content imports modules that only
exist on the NEWER main:

```
# dirty working-tree file (residual from PR #650):
src/services/chat-completions-service.js:42: } from './conversation-tool-governor.js';
# but conversation-tool-governor.js does NOT exist on base 0458509:
$ ls src/services/conversation-tool-governor.js
No such file or directory
```

`node --test` GROUPED runs then fail with:

```
# Error: A resource generated asynchronous activity after the test ended. This activity created
# the error "Error [ERR_MODULE_NOT_FOUND]: Cannot find module '.../conversation-tool-governor.js'
# imported from .../chat-completions-service.js" which triggered an uncaughtException event,
# caught by the test runner.
```

Key misdirection: the error fires **asynchronously AFTER the test ended**, so the failing
test name is the whole FILE, not a specific test. Solo runs of the same suite can pass
(56/56 in the observed case) — only grouped runs trip the lazy dynamic import.

### Diagnosis (stash-check, 30 seconds)
```bash
# 1. What's dirty?
git status --porcelain | head -20        # all '??' or mixed 'M' — identify residual vs mission
# 2. Is the import broken? Does the module exist?
grep -rn "conversation-tool-governor" src/ | head
ls src/services/conversation-tool-governor.js
# 3. STASH-CHECK: hide the residual state and re-run the failing grouped suite
git stash push -m "residual-check"
node --test --test-concurrency=1 test/<failing-suite>.test.js   # if PASS → residual is the cause
git stash pop
```
Also confirm provenance: `git log --oneline --all -- <missing-module>` — if the only commit
adding it is on main (later than the worktree base), it is a residual-of-later-PR problem.

### Permanent fix (merge main into the branch)
```bash
git merge origin/main --no-edit        # brings the missing modules (e.g. conversation-tool-governor.js)
# stash-pop conflicts on the SAME files → main's official version is authoritative:
git checkout --ours <residual-file>    # in a stash-pop context, --ours = current HEAD (main-merged)
git add <residual-file>
git stash drop "stash@{0}"             # only after confirming main carries the content
```
Verify with a full grouped run. The merge also converts the PR's `mergeStateStatus` from
`BEHIND` to `CLEAN`/`BLOCKED` (checks), which is the desired state before merge.

### Rule
NEVER attribute a grouped-test failure to mission code before running the stash-check.
A dirty worktree from a later-merged PR is a recurring environmental cause.

## 2. Shadow/canary comparison: legacy router returns a PROMISE with no trace fields

### Symptom
A shadow harness comparing legacy pipeline vs canonical service:

```js
const legacy = legacyRouter.route({ prompt, sessionKey });   // ← NOT awaited
console.log(legacy.route);                                    // undefined
```
produces `PARITY_DIVERGENCE` on EVERY axis for EVERY scenario because the legacy
read-only conversation router returns a thenable with shape:
`{ title, answer, surfaces_used, limitations, suggested_next_safe_query }` — no
`route`, no `trace`, no `provider_used`.

### Fix
```js
const raw = legacyRouter.route({ prompt, sessionKey });
const legacyResolved = raw && typeof raw.then === 'function' ? await raw : raw;
// normalize to the canonical parity axes:
const legacyTrace = {
  route: 'local_read_only',                                   // legacy router never sets route
  provider_used: false,
  memory_used: Array.isArray(legacyResolved?.surfaces_used) && legacyResolved.surfaces_used.length > 0,
};
```
Route divergence legacy-vs-canonical is EXPECTED (`local_read_only` vs `provider_backed`)
and is NOT itself a failure — the canonical pipeline is the one being introduced. Compare
SUBSTANCE on critical axes (identity, governance, memory, security) by inspecting answer
content, e.g.:
- identity: both answers mention the same persona (CLOE/Chloé)
- governance/forbidden: both refuse live mutation (regex on refusal language)
- memory: the stored code string appears in the recall-turn answer
Then report `critical axes` (match/check) separately from `parity verdict` (route-level).

## 3. Session collector exposes only a count → cross-turn memory never reaches the LLM

### Symptom
Shadow scenario "Retiens le code CANARY-42." → "Quel code t'ai-je demandé de retenir ?"
fails on BOTH pipelines: the LLM says it has no trace, even though the store persisted the
messages. Route/trace comparison does NOT catch this — it is invisible until you compare
answer CONTENT on the memory axis.

### Root cause
`collectSessionCollector` in `src/brain/cognitive-context-engine.js` exposed only:
```js
metadata: { session_key, message_count, last_role }
```
The actual transcript was never rendered into the provider prompt, and the `session`
collector was only requested when the prompt literally contained "session"/"conversation".

### Fix (3 parts, all required)
1. **Collector carries the transcript (bounded)**:
```js
const history = messages.slice(-12).map((m) => `${m.role}: ${String(m.content ?? '').slice(0, 500)}`);
// add to metadata: history
```
2. **Formatter renders it** (`src/brain/cognitive-context-formatters.js`, `formatItem`):
```js
const history = Array.isArray(item.metadata?.history) ? item.metadata.history : [];
if (history.length > 0) pieces.push(`conversation_history=${history.join(' | ')}`);
```
3. **Collector always included when a session object exists** (`requestedCollectors` in
`cognitive-context-engine.js`):
```js
...(session && typeof session === 'object' ? ['session'] : []),
```
(not only when the prompt contains "session")

### Test pattern (must be added with the fix)
```js
// store → recall within one session object
const pack = buildCognitiveContextPack({ prompt: 'Quel code t\'ai-je demandé de retenir ?',
  sessionKey, session: makeSession([{role:'user',content:'Retiens le code CANARY-42.'},
                                    {role:'assistant',content:'Je note CANARY-42.'}]), env });
const item = pack.available_items.find(i => i.kind === 'session');
assert.ok(item.metadata.history[0].includes('CANARY-42'));
const text = formatCognitiveContextPack(pack);
assert.ok(text.includes('conversation_history=') && text.includes('CANARY-42'));
```
Plus an empty-session no-crash test and a bound test (history ≤ 12 while message_count
shows the total).

## 4. Secret token masking breaks inline bash (Hermes terminal)

When a shell command interpolates a secret token read via `$(cat /tmp/key)` or
`Bearer $TOKEN`, the Hermes terminal's secret-masking can rewrite the command and bash
fails with `unexpected EOF while looking for matching '`'` (exit 2) — the command is
destroyed before execution, and it looks like YOUR syntax error.

Working patterns (use these instead of inline interpolation):
- Write the token to a file, then run the whole operation from a script file
  (`bash /tmp/script.sh`) so the token never appears in the one-liner.
- Read the token inside the script: `KEY=$(docker inspect ... | grep '^ADMIN_API_KEY' | cut -d= -f2-)`.
- For GitHub REST: `gh api -X PATCH ... -f body=@file` (file-based, no inline secret) or a
  small Python heredoc using `gh auth token` via subprocess (never print it).
- Verify what was actually executed: if the error is an EOF/quote error with `***` visible,
  it is masking, not your quoting.

## 5. `gh pr edit --body-file` can silently no-op on a GraphQL deprecation warning

`gh pr edit <PR> --body-file body.md` printed a non-fatal deprecation warning
(`GraphQL: Projects (classic) is being deprecated ... repository.pullRequest.projectCards`)
with exit 0 — but the PR body was NOT updated (subsequent `gh pr view --json body` still
showed the old text). Do not trust exit 0 alone.

Verify after edit:
```bash
gh api repos/<owner>/<repo>/pulls/<N> --jq '.body' | grep -c "<expected-marker>"
```
Fallback that always worked:
```bash
gh api -X PATCH repos/<owner>/<repo>/pulls/<N> -f body=@<file> --jq '.body | length'
# or with a JSON body via Python urllib using `gh auth token` (never echo the token)
```
The API `PATCH` path also lets you assert on the returned body (length + markers) in one step.

## 6. Provider/model field-name mismatch between boundary adapters (unit mocks mask it)

### Symptom
Canary Lot 6 real run: canonical trace showed `provider:null, model:null` on
provider_backed replies while `provider_used:true`. The canonical service exposed
no provider/model even though the parity suite (PR #652 lot 8) passed 4/4.

### Root cause
The shared brain router read `brainResult?.data?.provider ?? ...`, but the REAL
`buildBrainAskResponse` returns `data.llm_provider` / `data.llm_model`. The
parity tests passed because their MOCK brainAsk returned the field name the
consumer expected (`data.provider`), never the producer's actual shape.

### Fix (CLOE_TELEGRAM_CANARY_PROVIDER_TRACE_FIX_V1, PR #655)
Accept both field names in the boundary adapter:
```js
trace: {
  provider: brainResult?.data?.provider ?? brainResult?.data?.llm_provider ?? null,
  model:    brainResult?.data?.model    ?? brainResult?.data?.llm_model    ?? null,
  ...
}
```

### Rule
When a consumer is tested with a MOCKED producer, the mock can silently encode
the wrong field-name contract. Before certifying trace/parity, verify the real
producer's output shape (grep `data: {` in the producer) and add a test that
uses the REAL field names (P2b in the parity suite). Check `grep -n "llm_provider\|data: {" <producer>`.

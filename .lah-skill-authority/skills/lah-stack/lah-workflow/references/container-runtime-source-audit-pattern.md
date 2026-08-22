# Container Runtime Source Audit Pattern

Use this when auditing whether a feature branch's code is actually deployed in the running production container.

## Context

The production container (`lah-openclaw-mvp`, package name `lah-openclaw-mvp`) has a **flat source structure** — `/app/src/` contains all source files directly, without the `lah-openclaw-mvp/` prefix that exists in the git repository. The build may also embed `GIT_COMMIT=unknown`, making branch identity non-trivial.

## 5-Question Runtime Audit

For each link in the action path, answer:

1. **Does the function EXIST in the runtime source?** → `docker exec` + `sha256sum` comparison
2. **Does the runtime version match MAIN or the FEATURE branch?** → hash comparison
3. **Is the function IMPORTED by any production call chain?** → `grep -rn` in container
4. **Is the function ACTUALLY CALLED when the route is exercised?** → HTTP probe + inspect `_cloe.grounding.intent`
5. **Does the probe produce STRUCTURED pipeline output or generic LLM fallback?** → check `intent` field

## Phase 1 — Source Fingerprint

### Step 1: Identify container source root

```bash
CONTAINER="ec0ff4393102_lah-openclaw-mvp"
docker exec "$CONTAINER" find /app -maxdepth 1 -type d | sort
docker exec "$CONTAINER" test -d /app/src && echo "FLAT_STRUCTURE"
docker exec "$CONTAINER" test -d /app/lah-openclaw-mvp && echo "PREFIXED_STRUCTURE"
```

**Pitfall:** The container typically has `FLAT_STRUCTURE` — files at `/app/src/services/foo.js`, NOT `/app/lah-openclaw-mvp/src/services/foo.js`. Always check before assuming paths.

### Step 2: Compare runtime vs branch (SHA-256)

```bash
RUNTIME_HASH=$(docker exec "$CONTAINER" sha256sum /app/src/services/foo.js 2>/dev/null | cut -d' ' -f1 || echo "MISSING")
MAIN_HASH=$(sha256sum /home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp/src/services/foo.js 2>/dev/null | cut -d' ' -f1 || echo "MISSING")
FEATURE_HASH=$(sha256sum /home/deploy/lah-stack-worktrees/<worktree>/lah-openclaw-mvp/src/services/foo.js 2>/dev/null | cut -d' ' -f1 || echo "MISSING")

if [ "$RUNTIME_HASH" = "MISSING" ]; then echo "NOT_RUNTIME"
elif [ "$RUNTIME_HASH" = "$MAIN_HASH" ] && [ "$RUNTIME_HASH" = "$FEATURE_HASH" ]; then echo "ALL_MATCH"
elif [ "$RUNTIME_HASH" = "$MAIN_HASH" ]; then echo "MAIN_MATCH"
elif [ "$RUNTIME_HASH" = "$FEATURE_HASH" ]; then echo "FEATURE_MATCH"
else echo "UNIQUE"; fi
```

### Step 3: Check build identity

```bash
docker exec "$CONTAINER" env | grep GIT_COMMIT
docker exec "$CONTAINER" cat /app/package.json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('version','unknown'))"
```

**Pitfall:** When `GIT_COMMIT=unknown`, you CANNOT rely on embedded git metadata. Use SHA-256 content hash comparison exclusively. Also check whether the container has `.git` — production builds typically strip it.

## Phase 2 — Static Call Graph

Trace each symbol across three dimensions:

```bash
SYMBOL="handleCreateCampaignIntent"
echo "=== RUNTIME ==="
docker exec "$CONTAINER" grep -rn "$SYMBOL" /app/src/ 2>/dev/null | head -5 || echo "NOT_FOUND"
echo "=== FEATURE ==="
grep -rn "$SYMBOL" /home/deploy/lah-stack-worktrees/<worktree>/lah-openclaw-mvp/src/ 2>/dev/null | head -5
echo "=== MAIN ==="
grep -rn "$SYMBOL" /home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp/src/ 2>/dev/null | head -5
```

Also verify definition FILE existence:

```bash
docker exec "$CONTAINER" sh -c 'test -f /app/src/services/creative-import-bridge.js && echo "EXISTS" || echo "MISSING"'
```

## Phase 3 — Runtime HTTP Probe

### Probe template

```bash
API_KEY=$(cat /tmp/admin_key.txt 2>/dev/null || docker exec "$CONTAINER" sh -c 'echo -n "$ADMIN_API_KEY"')
curl -s -X POST 'http://127.0.0.1:4000/chat/completions' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer *** \
  -d '{"model":"test","messages":[{"role":"user","content":"'"$1"'"}],"stream":false,"user":"'"$2"'"}'
```

### Evidence extraction

Parse the response `_cloe` block for:
- `grounding.intent` — should be the expected structured intent (e.g. `campaign_action`, not `GENERAL_QUERY`)
- `safety.provider_write` — false unless a mutation was attempted
- `safety.execute_called` — false unless execution pipeline was invoked
- `verdict` — system classification

**Key indicator:** If `intent` is `GENERAL_QUERY` for ALL campaign prompts, the structured pipeline is NOT wired. The response is an LLM-generated refusal, not pipeline output.

### Recommended probe set (7 probes)

| # | Prompt | User | Tests |
|---|--------|------|-------|
| 1 | "what time is it" | test-001 | Baseline |
| 2 | "crée une campagne" | test-001 | Singular FR campaign |
| 3 | "lance des campagnes" | test-001 | Plural FR campaign |
| 4 | "crée une campagne" | (empty) | Missing conversation identity |
| 5 | "create a campaign called Test" | test-003 | English campaign |
| 6 | Same as 2, "l'offre est Example" | test-001 | Multi-turn continuity |
| 7 | "annule" | test-001 | Harmless cancellation |

## Phase 4 — Evidence Matrix

| Link | Source present | Feature branch | Runtime present | Production caller | Runtime proof | Verdict |
|------|:-------------:|:--------------:|:---------------:|:-----------------:|:-------------:|---------|

Verdicts: `WIRED_AND_RUNNING`, `WIRED_ON_BRANCH_NOT_RUNTIME`, `SOURCE_PRESENT_NOT_CALLED`, `TEST_ONLY`, `CONTRACT_MISMATCH`, `MISSING`, `UNKNOWN`

## Phase 5 — Safety counters

Always report at end: campaigns created = 0, ExoClick calls = 0, memory writes = 0, deployments = 0, runtime restarts = 0, Draw Things calls = 0.

## Known pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **Flat vs prefixed path** | `docker exec cat /app/lah-openclaw-mvp/src/foo.js` fails even though file exists in git | Check container structure first: `find /app -maxdepth 2 -type d`. Files at `/app/src/` (flat), not `/app/lah-openclaw-mvp/src/`. |
| **GIT_COMMIT=unknown** | Cannot identify which commit runs | Use SHA-256 content hashes against multiple branches |
| **Wrong auth header** | Probe returns `AUTH_REQUIRED` | Use `Authorization: Bearer `, not `x-api-key`. Check `auth-resolver.js` to confirm. |
| **Wrong endpoint path** | `Cannot POST /v1/chat/completions` | Use `/chat/completions` (no `/v1/` prefix) |
| **LLM fallback masquerading as pipeline** | Response says "cannot do it" in fluent text | Check `_cloe.grounding.intent` — `GENERAL_QUERY` means LLM fallback, not structured pipeline |

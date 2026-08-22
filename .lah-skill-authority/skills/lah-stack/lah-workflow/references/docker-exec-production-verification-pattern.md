# Docker Exec Production Verification Pattern

## When to use

When you need to verify a production deployment of a Node.js application by exercising its code directly — without exposing an HTTP route, without writing a test harness, and without curl/wget in the container. Use after a controlled deployment (Phase 5 of PRODUCTION_ACTIVATION) to prove write, read, and retrieval work through the actual production module imports.

## Prerequisites

- Container running and accessible via `docker exec`
- Node.js (18+ with `await import`) inside the container
- Any needed env vars already set in the container's runtime environment (via .env, docker-compose environment, etc.)

## Pattern

### Phase 1 — Verify module availability

```bash
docker exec -i <container> node -e "
const { writeCampaignMemory } = await import('/app/src/services/campaign-memory-writer.js');
console.log('campaign-memory-writer: AVAILABLE');
"
```

This proves the merged code is actually present in the running container (not just in the source repository).

### Phase 2 — Bounded write through application authority

```bash
docker exec -i <container> node -e "
const { writeCampaignMemory } = await import('/app/src/services/campaign-memory-writer.js');
const result = writeCampaignMemory({
  metadata: {
    type: 'observation',
    content: 'Synthetic activation test...',
    source: 'production-activation',
    campaign_id: 'CAMP-TEST-YYYYMMDD',
    creative_id: 'CR-ACTIVATION-001',
    result: 'failure'
  },
  mode: 'write',
  env: { MEMORY_APPEND_WRITE: 'true' }
});
console.log(JSON.stringify(result, null, 2));
"
```

Key verification points in the receipt:
- `ok: true` — validation passed
- `governance: "written"` — write gate was active and triggered persistence
- `write_activated: true` — actual disk write occurred
- `written: true` — file persisted
- Tags are correctly structured (campaign:, offer:, geo:, etc.)
- No secrets in content

### Phase 3 — Read back through application authority

```bash
docker exec -i <container> node -e "
const { queryCampaignMemory } = await import('/app/src/brain/campaign-memory-reader.js');
const byCampaign = queryCampaignMemory({ query: { campaign_id: 'CAMP-TEST-YYYYMMDD' } });
console.log(JSON.stringify({ retrieval_status: byCampaign.retrieval_status, result_count: byCampaign.result_count, id: byCampaign.results[0]?.id }, null, 2));
"
```

Expected: `retrieval_status: "available"`, `result_count: 1`, id matches write receipt.

### Phase 4 — Summary verification

```bash
docker exec -i <container> node -e "
const { readAllCampaignEvents } = await import('/app/src/brain/campaign-memory-reader.js');
const { summarizeCampaignMemory } = await import('/app/src/brain/campaign-memory-summary.js');
const allEvents = readAllCampaignEvents();
const summary = summarizeCampaignMemory(allEvents);
console.log(JSON.stringify(summary, null, 2));
"
```

Verify: campaign appears, creative listed as failed/success, no fabricated results, no causal claims.

### Phase 5 — Unknown query proves no fabrication

```bash
docker exec -i <container> node -e "
const { queryCampaignMemory } = await import('/app/src/brain/campaign-memory-reader.js');
const unknown = queryCampaignMemory({ query: { campaign_id: 'CAMP-NONEXISTENT-999999' } });
console.log(JSON.stringify(unknown, null, 2));
"
```

Expected: `retrieval_status: "empty"`, `result_count: 0`. No fabricated memory.

## Key rules

1. **Always use synthetic data** — test campaign IDs, test creatives, no real financial/affiliate information
2. **Verify the write gate** — confirm `governance: "written"` not `"read_only_dryrun"` or `"duplicate_skipped"`
3. **Verify file persistence on host** — check `ls -la <host_mount_path>/memory-events/` after write
4. **Record content hash** — `sha256sum <file>` for persistence proof
5. **Container restart test** — restart container, re-query, verify same event ID and hash
6. **No duplicate proof** — count files in memory-events/ after restart to prove no duplicate was created
7. **Tag the event unambiguously** — use a date-specific campaign ID (`CAMP-ACTIVATION-YYYYMMDD`) so it's clearly a test artifact in production

## Relationship to other patterns

| Pattern | When to use | Difference |
|---------|-------------|------------|
| `http-route-runtime-proof-pattern.md` | Testing Express HTTP routes with real `http.request()` | Requires an exposed route; more realistic for API endpoints |
| `cross-process-proof-pattern.md` | Testing idempotency across separate OS processes | Uses `spawn()` with file-based IPC |
| `container-api-probe-pattern.md` | Probing external HTTP APIs from inside the container | Probes OTHER services, not the container's own code |
| **This pattern** | Exercising the container's own code modules directly | No HTTP route needed; tests actual module imports in production |

## Pitfalls

| Trap | Symptom | Fix |
|------|---------|-----|
| Write goes to test dir instead of production | Event not in production memory-events/ after restart | Verify `docker exec` is on the production container (not a test one) |
| Duplicate detection fires on re-run | `governance: "duplicate_skipped"` instead of `"written"` | The dedup window is 24h; use a different content or campaign_id for a fresh write, OR clean data/memory-events/ if this is a test run |
| `import()` path wrong | `ERR_MODULE_NOT_FOUND` | Check the working directory inside the container (`WORKDIR` in Dockerfile); paths are relative to `/app` (or whatever WORKDIR is) |
| `docker exec` (no `-i`) | Node script fails silently | Always use `docker exec -i` for piped Node.js -e scripts to keep stdin connected |

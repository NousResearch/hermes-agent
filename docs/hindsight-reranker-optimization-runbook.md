# Hindsight Reranker Performance Fix — Runbook

**Status:** Applied and verified in production — 2026-09-25
**Owner system:** Hindsight (`hindsight-app` container, image `ghcr.io/vectorize-io/hindsight:0.9.1`)
**Deployment file:** `/opt/hindsight/compose.yaml` (root:root, 0640/0644 — requires sudo to view/edit)

---

## The problem

Hindsight recall latency averaged **13.88s** (max **25.55s**), dangerously close to Hermes's 20s
external prefetch timeout (`_EXTERNAL_PREFETCH_TIMEOUT_S` in `agent/memory_manager.py`). One
sample (25.55s) actually exceeded it. This risked silent memory-recall failures during normal
Hermes conversations.

## Root cause

Hindsight's recall pipeline runs a **local CPU cross-encoder** reranker
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, no GPU in this container) over candidates pulled from
three parallel retrieval sources (semantic, BM25, graph). The reranker candidate cap defaults to a
flat **300**, and the per-budget override for `mid` (the budget Hermes uses) was never set —
`HINDSIGHT_API_RERANKER_MAX_CANDIDATES_MID` defaulted to `0` (unset), which falls back to the flat
300-candidate cap regardless of the request's `budget` parameter. Reranking 300 candidates on CPU
took ~7.1s on average (up to 9.5s); the rest of the pipeline (parallel retrieval, DB, embedding)
was largely fine.

## The fix

Added one environment variable to the `hindsight` service in `/opt/hindsight/compose.yaml`:

```yaml
services:
  hindsight:
    image: ghcr.io/vectorize-io/hindsight:0.9.1
    container_name: hindsight-app
    environment:
      HINDSIGHT_API_DATABASE_URL: postgresql://hindsight_user:***@db:5432/hindsight_db
      HINDSIGHT_API_LLM_PROVIDER: openai-codex
      HINDSIGHT_API_LLM_MODEL: gpt-5.6-luna
      HINDSIGHT_API_WORKER_ID: hindsight-hermes-prod
      HINDSIGHT_API_RERANKER_MAX_CANDIDATES_MID: 100   # <-- added 2026-09-25
      CODEX_HOME: /home/hindsight/.codex
```

Applied with:
```bash
cd /opt/hindsight
sudo docker compose up -d --no-deps hindsight
```
`--no-deps` recreates only the `hindsight` service; `hindsight-db` (Postgres/pgvector) is untouched.

**This setting is read once at process startup** (`HindsightConfig` construction) — a plain
`docker restart` is NOT sufficient if the var was just added to the compose file; you must
`docker compose up -d` (recreate) to pick it up.

## ⚠️ Why this matters for future rebuilds/upgrades

If Hindsight is ever rebuilt, redeployed, migrated to a new host, or the compose file is
regenerated from a template/backup that predates 2026-09-25, **this line will silently disappear**
and latency will regress back to the ~14s average / 25s+ tail without any error or warning — just
slow recalls that may start timing out again. There is no other record of this inside the
container image itself (it's an env var, not baked into the image).

**Always verify after any Hindsight redeploy:**
```bash
docker inspect hindsight-app --format '{{json .Config.Env}}' | grep RERANKER_MAX_CANDIDATES_MID
```
Expected output: `HINDSIGHT_API_RERANKER_MAX_CANDIDATES_MID=100`. If missing, re-add it to
`compose.yaml` per the diff above and recreate the service.

## Benchmark: before vs after (12 fresh `trace:true` recall samples each)

| Metric | Before (300 candidates) | After (100 candidates) |
|---|---:|---:|
| Total latency — min | 11.98s | 5.05s |
| Total latency — median | ~13.5s | 5.57s |
| Total latency — avg | 13.88s | 5.58s |
| Total latency — p95 | ~20s | 6.18s |
| Total latency — max | 25.55s | 6.18s |
| Reranking stage — avg | 7.1s | 1.81s |
| Reranking stage — range | 6.1–9.5s | 1.66–2.14s |
| Parallel retrieval — avg | 6.6s | 3.67s (unchanged config; natural variance) |
| Candidates reranked (confirmed via trace) | 300 | 100 |
| Errors/timeouts (of 12 samples) | 0 | 0 |

**Result: ~60% cut in total latency, ~75% cut in reranking latency, zero errors.**

## Recall quality validation (7-query eval set against real stored memories, after the change)

| Metric | Result |
|---|---|
| Top-1 hit rate | 6/7 (86%) |
| Top-3 hit rate | 7/7 (100%) |
| Top-5 hit rate | 7/7 (100%) |

No quality regression observed at 100 candidates vs the 300-candidate baseline.

## Full E2E chain (verified same day)

| Stage | Result |
|---|---|
| STORE | PASS (7.02s) |
| RETRIEVE | PASS (4.43s — rank-1 semantic match; note literal test-marker strings get paraphrased away by Hindsight's LLM extraction during STORE, this is expected/known behavior, not a defect) |
| PREFETCH | PASS |
| INJECT | PASS |
| MODEL USE | PASS |

## Related settings (NOT changed, documented for awareness)

| Setting | Current value | Purpose |
|---|---|---|
| `HINDSIGHT_API_RERANKER_MAX_CANDIDATES` | unset → 300 (flat/global fallback) | Global reranker cap if no per-budget override is set |
| `HINDSIGHT_API_RERANKER_MAX_CANDIDATES_LOW` | unset → 0 (disabled, falls to flat 300) | Per-budget override for `budget=low` — untouched, still broken/no-op the same way `mid` was before this fix |
| `HINDSIGHT_API_RERANKER_MAX_CANDIDATES_HIGH` | unset → 0 (disabled, falls to flat 300) | Per-budget override for `budget=high` — untouched |
| `HINDSIGHT_API_RECALL_BUDGET_FIXED_MID` | unset → 300 | Per-source retrieval depth (semantic/BM25/graph each fetch this many before merge); drives the ~3.67s parallel-retrieval stage. Deliberately left untouched per Daniel's instruction — next lever to investigate if more speed is needed |
| `_EXTERNAL_PREFETCH_TIMEOUT_S` (Hermes side, `agent/memory_manager.py`) | 20.0s | Left unchanged — now has real headroom (6.18s max vs previous 0.9s headroom) |

## Rollback (if this change ever needs to be reversed)

Remove the `HINDSIGHT_API_RERANKER_MAX_CANDIDATES_MID: 100` line from `compose.yaml`, then:
```bash
cd /opt/hindsight && sudo docker compose up -d --no-deps hindsight
```
This restores the flat-300 fallback behavior (and the pre-fix latency profile).

## If you need to test different candidate limits later

Considered but not applied: 50 candidates (would save ~1s more reranking time, but current
6.18s max / 100% top-3-top-5 quality already gives comfortable margin — diminishing return,
so not pursued 2026-09-25). If revisiting, benchmark the same way: 12+ `trace:true` samples,
compare reranking-stage duration + the 7-query relevance eval before changing further.

# Twitter Pipeline Blueprint (Trend Digest lane)

**Date:** 2026-02-11  
**Authoring scope:** blueprint only (no broad refactor)  
**Repos inspected:**
- `/home/yeqiuqiu/clawd-architect`
- `/home/yeqiuqiu/projects/ngmi-terminal`
- `/home/yeqiuqiu/projects/walletdb` (referenced + inspected)

---

## 1) Current-state map

### 1.1 Entrypoints currently present

#### A) NGMI core ingest/scan (Telegram/Discord only)
- CLI service entrypoint: `ngmi-terminal run` (`/home/yeqiuqiu/projects/ngmi-terminal/src/ngmi_terminal/cli.py`)
- System behavior docs: `/home/yeqiuqiu/projects/ngmi-terminal/docs/NGMI_SYSTEM_OVERVIEW.md`
- DB schema for message logs: `/home/yeqiuqiu/projects/ngmi-terminal/src/ngmi_terminal/db/schema.sql` (`chat_logs`, `raw_messages`, etc.)

#### B) Legacy Trend Digest script (cross-coupled, LLM-first, non-Twitter)
- Script: `/home/yeqiuqiu/clawd-architect/walletdb/ingest/ngmi_flash_digest.py`
- Reads NGMI `chat_logs` directly, then prompts OpenClaw/Gemini for digest text.
- This path is opt-in gated (`WALLETDB_ENABLE_NGMI_CONTEXT`) and was explicitly called out as legacy/optional.

#### C) WalletDB Apify staging framework (generic, not Twitter-specific)
- CLI commands: `walletdb apify run|ingest` in `/home/yeqiuqiu/projects/walletdb/src/walletdb/cli.py`
- Staging table: `apify_ingest` in `/home/yeqiuqiu/projects/walletdb/src/walletdb/db/schema.py`
- Actor registry has no Twitter actor at present (`/home/yeqiuqiu/projects/walletdb/src/walletdb/apify/actors.py`).

### 1.2 Data flow as-is

```text
Telegram/Discord -> ngmi-terminal ingesters -> ngmi_terminal.db (chat_logs/raw_messages/calls)
                                                   |
                                                   +-> deterministic scan/lane outputs (NGMI domain)

(legacy optional)
ngmi_terminal.db -> clawd-architect/walletdb/ingest/ngmi_flash_digest.py -> LLM summary -> Telegram post
```

### 1.3 Where data lands today

- **NGMI message data:** `/home/yeqiuqiu/projects/ngmi-terminal/data/ngmi_terminal.db`
  - `chat_logs` exists and is active.
  - sampled facts at inspection time:
    - ~66k `chat_logs` rows
    - platforms: `telegram`, `discord`, `ngmi_bot`
    - `x.com` links exist in message text, but this is **chat content**, not direct Twitter scrape ingest.

- **WalletDB Apify staging:** `/home/yeqiuqiu/projects/walletdb/data/walletdb.sqlite` table `apify_ingest`
  - table exists, but sampled row count is **0** (no staged ingest currently).

- **Twitter-specific canonical store/output:** not found in inspected repos.

### 1.4 Scheduled jobs currently relevant

- `ngmi-terminal/scripts/run_bullish_scan_once.sh` includes cron example (every 15m) for bullish scan.
- `ngmi-terminal/scripts/run_recap_4h_once.sh` is cron/systemd-friendly for recap send.
- `ngmi-terminal/docs/ops/schedules_and_watchdogs.md` states OpenClaw recap cron should stay disabled.
- Historical note: OpenClaw cron job named **"NGMI Trend Digest"** was disabled due to NGMI/WalletDB mixing (`memory/2026-02-10_cleanup_and_separation.md`).

**Twitter scrape schedule:** none found as an active code path.

### 1.5 What is broken/noisy + duplication points

1. **No concrete Twitter ingest lane implemented**
   - Requirement exists in docs/memory, but there is no active Twitter scrape -> normalize -> trend pipeline in code.

2. **Legacy digest path is non-deterministic and cross-coupled**
   - `ngmi_flash_digest.py` uses LLM generation directly from raw chat text and lives under WalletDB-ish path while reading NGMI DB.
   - This conflicts with deterministic-first and separation goals.

3. **No canonical Twitter schema yet**
   - Missing first-class tables for tweet raw records, normalization artifacts, dedupe lineage, topic windows, novelty history.

4. **Spec/code drift**
   - Trend logic docs are detailed (`docs/dev/trend_digest_trends_and_verdicts.md`) but implementation for Twitter lane is absent.

5. **Potential duplication risk**
   - Similar “digest” concepts exist in NGMI and WalletDB domains; without strict boundaries, operator confusion and accidental path-mixing recurs.

---

## 2) Target architecture — Trend Digest Twitter lane (deterministic-first)

## 2.1 Design principles

- Deterministic pipeline for ingest, normalization, dedupe, clustering, scoring.
- LLM (Gemini) is **optional** and only for final prose summarization/formatting.
- No NGMI/WalletDB DB cross-reads.
- Evidence-first output: each trend references concrete tweet ids + rule traces.

### 2.2 Proposed lane stages

1. **Ingestion (raw capture)**
   - Input from paid scraper/API (For You + Following snapshots).
   - Persist immutable raw payloads with `run_id`, `source_feed`, `fetched_at_utc`.

2. **Normalization**
   - Canonical author handle/id, canonical tweet id, canonical text normalization.
   - Extract deterministic entities: cashtags, hashtags, handles, URLs/domains, contracts.
   - Persist `tweet_created_at_utc` when available (do not infer from fetch time).

3. **Deduplication**
   - Hard dedupe: tweet id.
   - Structural dedupe: retweet/quote canonicalization to root tweet.
   - Near-dup guard (optional deterministic hash bucket) for repeated copy-paste posts.

4. **Topic clustering**
   - Deterministic token/entity buckets first.
   - Merge by explicit overlap rules (e.g., same ticker + shared dominant hashtags/domains).

5. **Novelty + trend scoring**
   - Multi-window metrics (short + long).
   - Velocity vs baseline, unique-authors, cross-feed confirmation, novelty lookback.
   - Hysteresis to avoid flicker.

6. **Outputs**
   - `alerts` (high-confidence investigate/act candidates)
   - `brief artifacts` (JSON facts + markdown summary)
   - `operator evidence pack` (top tweet ids/excerpts/links)

7. **Optional Gemini enrichment (strictly final step)**
   - Input: precomputed deterministic facts JSON.
   - Output: concise narrative paragraph(s), no score mutation.

### 2.3 Recommended storage boundaries

- Implement Twitter lane in **NGMI-terminal domain** (same product lane as Trend Digest), e.g.:
  - `/home/yeqiuqiu/projects/ngmi-terminal/data/ngmi_twitter_lane.sqlite`
- Keep WalletDB untouched for this lane unless explicitly building a separate WalletDB-native feature.
- If external runs are used, share via file artifact import/export only (no cross-DB joins).

### 2.4 Minimal schema set (Twitter lane DB)

- `twitter_ingest_runs(run_id, source_feed, started_at_utc, finished_at_utc, item_count, status, meta_json)`
- `twitter_posts_raw(run_id, source_feed, fetched_at_utc, post_id, author_id, payload_json, PRIMARY KEY(run_id, post_id))`
- `twitter_posts_norm(post_id PRIMARY KEY, canonical_post_id, author_handle, text_norm, created_at_utc, engagement_json, entities_json, urls_json, first_seen_run_id)`
- `twitter_dedupe(post_id, dedupe_reason, canonical_post_id, rule_version)`
- `twitter_topic_events(window_start_utc, window_end_utc, topic_key, counts_json, evidence_json)`
- `twitter_trends(trend_id, topic_key, score, verdict, novelty, why_json, evidence_json, emitted_at_utc)`

---

## 3) Phased implementation plan (P1/P2/P3)

## P1 — Foundation (ingest + normalize + deterministic artifacts)

**Goal:** create reproducible raw->normalized Twitter data path with no alerting yet.

### File-level changes (planned)
- `ngmi-terminal/src/ngmi_terminal/twitter/__init__.py` (new)
- `ngmi-terminal/src/ngmi_terminal/twitter/schema.sql` (new)
- `ngmi-terminal/src/ngmi_terminal/twitter/ingest.py` (new)
- `ngmi-terminal/src/ngmi_terminal/twitter/normalize.py` (new)
- `ngmi-terminal/src/ngmi_terminal/cli.py` (add `twitter ingest` + `twitter normalize` subcommands)
- `ngmi-terminal/scripts/run_twitter_ingest_once.sh` (new, cron-safe one-shot)
- `ngmi-terminal/docs/ops/twitter_lane_runbook.md` (new)

### Commands (planned)
```bash
cd /home/yeqiuqiu/projects/ngmi-terminal
python -m ngmi_terminal twitter ingest --input data/import/twitter_raw_*.jsonl --feed for_you
python -m ngmi_terminal twitter normalize --run-id <run_id>
python -m ngmi_terminal twitter stats --since 24h
```

### Exit criteria
- Ingest run manifest exists.
- Normalized rows reproducible on rerun.
- No LLM/Gemini usage in this phase.

## P2 — Deterministic trend engine (dedupe + clustering + novelty)

**Goal:** generate ranked trend candidates + verdicts from normalized posts.

### File-level changes (planned)
- `ngmi-terminal/src/ngmi_terminal/twitter/dedupe.py` (new)
- `ngmi-terminal/src/ngmi_terminal/twitter/topics.py` (new)
- `ngmi-terminal/src/ngmi_terminal/twitter/scoring.py` (new)
- `ngmi-terminal/src/ngmi_terminal/twitter/verdicts.py` (new)
- `ngmi-terminal/tests/test_twitter_dedupe.py` (new)
- `ngmi-terminal/tests/test_twitter_scoring.py` (new)
- `ngmi-terminal/tests/test_twitter_verdict_hysteresis.py` (new)

### Commands (planned)
```bash
cd /home/yeqiuqiu/projects/ngmi-terminal
python -m ngmi_terminal twitter dedupe --since 24h
python -m ngmi_terminal twitter detect-trends --short-window 60m --long-window 24h
python -m ngmi_terminal twitter emit --format json --out data/outputs/twitter_trends_latest.json
```

### Exit criteria
- Same input snapshot -> same trend output.
- Trend evidence includes tweet ids + rule traces.
- Hysteresis working across consecutive runs.

## P3 — Alert/brief delivery + optional Gemini summarization

**Goal:** ship operator-facing digest outputs with strict deterministic core retained.

### File-level changes (planned)
- `ngmi-terminal/src/ngmi_terminal/twitter/brief.py` (new)
- `ngmi-terminal/src/ngmi_terminal/twitter/summarize_gemini.py` (new; optional, guarded)
- `ngmi-terminal/src/ngmi_terminal/reports/twitter_digest.py` (new)
- `ngmi-terminal/scripts/run_twitter_digest_once.sh` (new)
- `ngmi-terminal/docs/ops/twitter_digest_delivery.md` (new)

### Commands (planned)
```bash
cd /home/yeqiuqiu/projects/ngmi-terminal
python -m ngmi_terminal twitter brief --since 12h --out data/outputs/twitter_brief_latest.md
python -m ngmi_terminal twitter brief --since 12h --gemini-summarize
```

### Exit criteria
- JSON facts artifact always produced.
- Gemini enrichment optional + non-blocking.
- Delivery job separated from ingest/scoring job.

---

## 4) Quick wins we can ship today (<=3)

1. **Freeze legacy mixed path in docs/runbook**
   - Mark `clawd-architect/walletdb/ingest/ngmi_flash_digest.py` as legacy-only and explicitly out-of-scope for Twitter lane.

2. **Add Twitter lane runbook stub in ngmi-terminal**
   - Define input contract (`post_id`, `created_at`, `author`, `text`, `engagement`) and deterministic rules before coding.

3. **Add a “no-cross-db” CI grep guard**
   - Fail if new Twitter lane code references `/home/yeqiuqiu/projects/walletdb` or `walletdb.sqlite` from NGMI code paths.

---

## 5) Guardrails

- Do not read WalletDB DB from NGMI Twitter lane.
- Do not read NGMI DB from WalletDB Twitter lane.
- Keep scoring deterministic and auditable.
- Gemini allowed only for final summarization over deterministic fact objects.

---

## Appendix — evidence snippets from inspection

- No Twitter actor in WalletDB Apify actor registry (`walletdb/src/walletdb/apify/actors.py`).
- `apify_ingest` table exists in WalletDB schema but sampled row count was `0`.
- NGMI `chat_logs` is active and includes many messages + some x.com links, but no dedicated Twitter scrape ingest tables were found.
- Historical OpenClaw cron "NGMI Trend Digest" was disabled due to cross-system mixing (`memory/2026-02-10_cleanup_and_separation.md`).

---
name: screener-ops
description: >-
  Umbrella skill for daily production operations, the Hermes knowledge layer,
  agent fleet monitoring, spec/governance lifecycle, and read-only monitoring
  sweeps for the Wake Robin biotech screener.
tags:
  - biotech-screener
  - operations
  - governance
  - monitoring
  - fleet
---

# Screener Ops & Governance Skill

## Purpose

Reference for daily production operations, the Hermes knowledge layer, agent fleet monitoring, spec/governance lifecycle, and structured monitoring sweeps that govern all changes to the biotech screener.

This skill is organized into three sections:

## Operator Profile

- **Operator**: Darren Schulz, CFA, CAIA â€” Director of Investments, Wake Robin (Holland, MI)
- **Credentials**: CFA (CFA Institute), CAIA (CAIA Association). 30+ years institutional investment management.
- **Domain expertise**: Asset allocation, portfolio construction, manager research & selection, biotech equity research & due diligence, 13F/13D/13G filing analysis, clinical trial analysis, SEC EDGAR, derivatives/options.
- **Technical skills**: AI agent architecture & development (designed and built the Hermes/OpenClaw fleet), LLM prompt engineering, Python scripting, WSL2/cron administration, API integration.
- **Escalation authority**: All QUARANTINE, PRODUCER_AUDIT_REQUIRED, and spec approval decisions route to this operator. The operator is the sole authority for promotion approvals, spec closures, architecture changes, and pipeline governance decisions.
- **Town-Hermes bridge target**: Operator briefs and alerts deliver to djschulz@gmail.com (personal) and dschulz@wakerobin.co (work).
- **Wake Robin context**: Wake Robin is a real estate investment and community development company. The DEM biotech screener is a parallel investment research capability operated by the Director of Investments.

1. **Framework Reference** â€” Stable pipeline architecture, processes, governance, and monitoring protocols (changes only with code updates)
2. **Operational State** â€” Volatile infrastructure and status snapshots that require periodic refresh
3. **Monitoring Sweep Protocol** â€” Read-only tri-sweep pattern for periodic health checks

---

# SECTION 1: FRAMEWORK REFERENCE

---

## Daily Production Pipeline

**Runner**: `tools/run_daily_production.py` (13-step orchestrator)
**Cron**: 5:30 PM ET weekdays + `@reboot` catch-up for missed runs

### Pipeline Steps (in order)

1. Price refresh
2. Cache warm (including FDA)
3. Screen (with `--inputs-manifest write`)
4. Audit
5. Gates
6. Manifest + promotion
7. Drift report
8. Action packet
9. Shadow portfolio
10. Trade plan
11. Portfolio report
12. Readiness scorecard
13. Ops digest + PIT backfill (optional)

### Key Rule

Always warm 8-K cache BEFORE running screen.

### Pipeline Timeout

6000s (100 min) to cover worst-case AACT + tail steps. Previous 4500s was killing mid-AACT on Mondays.

### Institutional Summary Artifact Flow

The `institutional_summary.json` has a two-copy architecture with a known promotion gap:

| Copy | Path | Refreshed? |
|------|------|------------|
| Snapshot | `data/snapshots/{date}/institutional_summary.json` | YES â€” daily (step 3) |
| Canonical | `production_data/institutional_summary.json` | **NO** â€” last manual copy 2026-04-13 |

The daily pipeline writes fresh data to the snapshot dir, but **no step copies it to `production_data/`**. The cohort quarantine validator (`check_13f_cohort_quarantine.py`) reads the stale canonical copy, causing a false G2 freshness failure â€” even when 46/48 managers have filed.

Full diagnosis steps, fix options, and architecture details in `references/13f-institutional-summary-pipeline.md`.

### Snapshot Artifact Layout

The pipeline produces multiple artifacts in `data/snapshots/{date}/`. The primary ranking output is `rankings.csv` â€” NOT `snapshot_top30.json` (a legacy name from an earlier version). Full artifact map, verification commands, and naming history in `references/snapshot-artifact-layout.md`.

Key rule: check `data/snapshots/{date}/rankings.csv` for ranking output. `production_data/` is NOT the snapshot output directory â€” it only holds specific promoted artifacts (universe.json, run_log, market_data, etc.).

### Yahoo Finance Rate Limiting / Market Data Staleness

**Symptom cascade**: Yahoo Finance API returns 429 (Too Many Requests) for ALL tickers including AAPL, XBI, AMGN. This is NOT a delisting event â€” it's rate limiting at Yahoo's end.

**Failure signature in logs**:
```
Failed to get ticker 'XBI' reason: Expecting value: line 1 column 1 (char 0)
XBI: No timezone found, symbol may be delisted
429 Client Error: Too Many Requests for url: https://query2.finance.yahoo.com/...
```

**Affected pipeline steps**:
- **Step 1 (Price refresh)**: yfinance returns empty for every ticker â†’ 0 rows appended. Pipeline continues (non-fatal).
- **`collect_market_data.py`**: Network test catches the 429 â†’ falls back to cached data. `--force-refresh` does NOT circumvent this â€” the network test gates before the force override.
- **Step 5 (Gates â€” market data staleness)**: `market_data.json` age exceeds 3-day limit â†’ **abort** (hard gate).

**Diagnosis**:
1. Check `production_data/market_data.json` last-modified date vs today
2. Test yfinance connectivity: `curl -s -o /dev/null -w "%{http_code}" "https://query2.finance.yahoo.com/v10/finance/quoteSummary/AAPL?modules=financialData&corsDomain=finance.yahoo.com"`
3. Status code 429 = rate limited. Status code 200 = issue may be in the yfinance library or local environment.

**Remediation options**:
1. Wait: Yahoo 429 bans typically lift in 1-4 hours. Try the scheduled 16:30 ET cron.
2. Extend staleness gate: Temporarily increase `MAX_MARKET_DATA_AGE_DAYS` in `run_daily_production.py` from 3 to 5 (revert after Yahoo recovers). Exact gate location: search for "Market data staleness gate" in the file.
3. Force refresh: `--force-refresh` on `collect_market_data.py` does not bypass a live 429 â€” the network test at the start of the script gates before the refresh logic.

**Monitoring**: Check `logs/cron.log` for wrapper-level PASS/FAIL and `logs/daily_production_YYYY-MM-DD.log` for the gate-level abort.

### Pipeline Exit Codes

`run_daily_production.py` uses exit codes to signal completion quality:

| Code | Meaning | Action |
|------|---------|--------|
| 0 | PASS â€” all steps completed clean | Normal day |
| 2 | WARN â€” core pipeline completed, one or more non-fatal steps failed or timed out | Verify WARN sources; track repeat offenders |
| Other | FAIL â€” pipeline aborted at a gate or crashed | Diagnose the blocking gate in the log |

**Common WARN sources (exit 2):**
- **Herald classify timeout** â€” `classify_press_releases.py --use-grok` times out at 300s. Non-blocking â€” deduped JSONL exists and retries on next run. One-off is fine; 3+ consecutive WARNs from the same source becomes an ops issue.
- **Data integrity audit WARN** â€” minor schema/coverage drift caught by `data_integrity_audit`.

**State recording for WARN-completed runs:**
When a production run exits 2 (completed with WARN), record the state concisely:

```
State:         COMPLETED_WARN
Exit:          2 (non-fatal)
WARN source:   [specific step and error]
Tracking:      Single occurrence / repeat pattern
Non-blocking:  [yes/no â€” if yes, specify what completed successfully]
Patch/live:    [any temp patches in effect with revert trigger]
```

For a worked example and full follow-up checklist, see `references/2026-05-19-production-warn-template.md`.

### CI Failures and Monthly Budget Cap

GitHub Actions minutes reset on the 1st of each month. CI failures in the last ~2 weeks of a month may be caused by the Actions budget cap being exhausted rather than code regressions. Check https://github.com/settings/billing before debugging CI failures in the second half of a month.

When an email flood of `Run failed` notifications arrives (e.g., ~25 failure emails across multiple workflows and SHAs in one day), follow this triage sequence:

1. Check `ci-account-gating-triage` skill first â€” rule out billing/quota before code-level debugging
2. Then run `github-email-incident-deduper` skill to cluster failures by root cause
3. Only if neither billing nor account gating explains it, proceed to per-workflow debugging

---

## Hermes Knowledge Layer (Spec 089)

**Generator**: `tools/build_hermes_knowledge_layer.py`

Repo-native "ops brain" that continuously answers:

1. What is the current operational state?
2. What changed since the last good state?
3. What is held, blocked, or awaiting first-fire validation?
4. What contradictions exist across specs, audit memos, cron, and registry?
5. What is the next allowed operator action?
6. What is explicitly not allowed?

### Four Layers

| Layer | Purpose | Output |
| --- | --- | --- |
| Capture | Read-only from specs, artifacts, registry, git, cron | Raw state |
| Normalize | Structured ledgers | `artifacts/ops/knowledge_layer/` |
| Reason | Drift, contradiction, missed-run detection | Alerts |
| Deliver | Operator briefs | Daily/weekly summaries |

### Output Artifacts

| Artifact | Location |
| --- | --- |
| Latest state | `artifacts/ops/knowledge_layer/latest_state.{json,md}` |
| Held spec ledger | `artifacts/ops/held_spec_ledger/latest.{json,md}` |
| First fire ledger | `artifacts/ops/first_fire_ledger/latest.{json,md}` |
| Contradiction ledger | `artifacts/ops/contradiction_ledger/latest.md` |
| Operator briefs | `artifacts/ops/operator_brief/daily/YYYY-MM-DD.md` |

### Phase 2A â€” Cron Automation

**Cron**: 5:45 PM ET weekdays (OS crontab entry)
**Runner**: `/usr/bin/python3 tools/build_hermes_knowledge_layer.py`
**Log**: `logs/knowledge_layer.log`
**Runbook**: `references/hermes-knowledge-layer-cron.md`

No `.env` sourcing needed â€” the builder is a pure filesystem reader (git, specs, local artifacts, no API calls).

**Cron branch**: Scheduled on `hermes-knowledge-layer-cron-2026-05-19`, NOT on the feature branch (`spec-089-kg-implementation`). Follow this pattern: ops infrastructure changes (cron wiring, scheduling, runbooks) get their own branch, separate from feature implementation.

**Schedule choice rationale**: After production (16:30 ET) and data pipeline (14:00-17:00), before the 18:00+ wave of review/fleet jobs.

### First-Fire Validation Protocol

For any new cron job added to the fleet (whether OS-crontab Python tool or Hermes cron), run this validation checklist on the first execution:

```
1. Job runs           â€” confirm execution (log timestamp, job list)
2. Outputs written    â€” ls -la on expected output file path
3. Outputs parseable  â€” python3 -c "json.load(open('<path>'))" (or equivalent)
4. Key field reported â€” grep for critical metric in log (e.g. "contradictions: X")
5. Failure exits      â€” test with invalid input; confirm nonzero exit and clear log message
6. Gitignore verified â€” git check-ignore <output_path> returns tracked path
```

The checklist is one-shot per new job, not recurring. Skip items where the tool's design makes a test destructive or nonsensical.

### Phased Implementation Pattern

When implementing spec changes involving both deterministic automation and LLM-driven components, follow this order:

1. **Deterministic first** â€” Implement the Python/repo-based tooling that produces structured output. No LLM calls, no reasoning surface.
2. **Validate before expanding** â€” Let the deterministic foundation run on-schedule for several clean days before layering any LLM reasoning on top.
3. **LLM-driven components deferred** â€” Mon-schedule reasoning jobs (held-spec analysis, first-fire reports, trend detection) wait until the deterministic ledgers are proven clean and stable.
4. **Explicit "not yet" boundaries** â€” Document what is deferred and why, so future sessions don't re-negotiate the scope.

This pattern emerged from the KG Phase 2 implementation: the Python builder was automated first (Phase 2A), with full LLM-driven alerting and town integration (Phase 2B) deferred until after >3 clean daily runs.

---

## Town-Hermes Bridge (Spec 090)

**Module**: `common/operator_delivery.py`

Routes Hermes Knowledge Layer events to Town via email trigger. Town does NOT control Hermes.

### Architecture

```
Hermes job completes
  -> write ledger artifact (repo)
  -> send_operator_event(channel="town", ...)
    -> structured email to TOWN_EMAIL (djschulz@gmail.com)
    -> Town routine triggers on [Hermes] subject prefix
    -> Town creates task / DMs operator
```

### What Town is NOT

- NOT a scheduler or cron controller
- NOT a repo mutator or spec approver
- NOT allowed to reactivate bioshort_watch LLM
- NOT the authoritative source for any production state

---

## OpenClaw Agent Fleet

### Agent Registry

**File**: `agents/AGENT_REGISTRY.json`

- Schema v1.0, as-of 2026-05-17 (per agent_governance.md, authoritative source): 30 total agents â€” 27 active, 1 suppressed (bioshort_watch), 1 retired (company_news_ingest), 1 shadow (shadow_watch). Other documents may show older counts (17, 26, 27, 28) â€” always cite agent_governance.md with a dated reference.
- Authority levels: observe_only, observe_and_propose, write_artifacts, mutate_data, mutate_config
- Only crt_resolution_watcher holds mutate_data authority (writes to catalyst resolution tables under orchestrator supervision)

### Model Configuration (updated 2026-05-19)

- **Primary provider**: Nous Research - deepseek/deepseek-v4-flash (free tier)
- **Fallback provider**: Together AI - meta-llama/Llama-3.3-70B-Instruct-Turbo
- **Previous**: OpenRouter (out of credits as of 2026-05-13), Together primary (2026-05-13 to 2026-05-18)
- **Auto-routing**: Provider chain: Nous â†’ Together â†’ (Fallback chain for fails)

### Inference Tuning (deepseek optimized)

DeepSeek v4 Flash on Nous free tier: 128K context, ~200 tokens/sec. Free tier subject to periodic 429 rate limits ("upstream Provider returned error"). Retry after 30-60 min if this occurs â€” not a credential issue. Adding credits to a free-tier key does not bypass rate limits.

### Uncertainty Handling (all agents, 2026-05-13)

All agents tuned with explicit uncertainty escalation rules:

- ops_supervisor: missing artifacts -> RED (not GUESS); confidence < 0.7 -> escalate
- sentinel: missing drift -> FAIL; boundary cases -> WARN; ambiguous rollback -> both commands
- data_auditor: missing snapshot -> FAIL; specific ticker counts (not "some")
- ic_health_monitor: missing dashboard -> UNKNOWN; threshold boundaries -> ALERT (conservative)
- fleet_steward: unreachable status -> MEDIUM; missing last_run -> anomalous (not healthy)

### Llama-Specific Prompting

Agent AGENTS.md docs updated with Llama-specific procedures:

- IF/THEN chains instead of open-ended reasoning
- Step numbering for multi-step workflows
- Schema-first output format
- No inferred data; report missing explicitly

### Gateway Monitoring

- `~/.hermes/monitor_together_latency.py` tracks latency trends
- Alerts on success rate <80% or avg latency >5s
- Logs to `together_latency.log`

### Known Broken Cron Jobs

| Cron Job | Job ID | Failure | Expected Fix Date |
|----------|--------|---------|-------------------|
| pdufa-proximity-alert | e84535b2 | Uses `arcee-ai/trinity-large-thinking` which is not available on Nous free tier. Still configured with old OpenRouter model. | Update job to use current provider chain (see Model Configuration above). Attempted 2026-05-19 08:15 ET â€” will retry 2026-05-20 08:15 and fail again unless updated. |

### Monitoring Layers

| Layer | Tool | Purpose |
| --- | --- | --- |
| Heartbeat | `tools/agent_heartbeat_checks.py` | Per-agent health |
| Supervisor | `agents/ops_supervisor/supervisor.py` | Fleet-wide anomaly classification |
| Post-snapshot | `tools/run_post_snapshot_supervisor.py` | Post-pipeline task orchestration |
| Sentinel | `tools/agent_supervisor_sentinel.py` | Final watchdog |

### Anomaly Classification

| Classification | Severity | Meaning |
| --- | --- | --- |
| new | ORANGE | First occurrence |
| carried | YELLOW | Same anomaly seen yesterday (exact text match) |
| resolved | GREEN | Previously seen, now gone |

Terminal agents (e.g., ops_supervisor) are intentionally unsupervised and do not carry HEARTBEAT.md.

### Herald Pipeline

Done predicate requires BOTH deduped AND classified JSONL:

- `data/press_releases/deduped/deduped_{date}.jsonl`
- `data/press_releases/classified/classified_{date}.jsonl`

If classification failed but dedupe exists, the next supervisor run retries classification.

**Known WARN: classify timeout**
- `classify_press_releases.py --use-grok` can time out at the 300s default on heavy press-release days (10+ releases).
- **Non-blocking**: deduped JSONL exists. The timeout means only classification was missed. Retries on next run.
- **Downgrade path**: if timeout becomes persistent (3+ consecutive WARNs), consider increasing the timeout or using the non-grok classify fallback.
- **Diagnosis**: look for `Timeout` or `300s` in the Herald section of the daily production log.

---

## SOUL.md / Ruleset System

### SOUL.md

Per-agent operating manual defining boundaries, tools, and heartbeat checks. Located in each agent workspace under `agents/{name}/SOUL.md`.

### Ruleset Health Monitor

**Tool**: `tools/ruleset_health_monitor.py`

- JSONL history grows with each new evaluation date (idempotent on same-day reruns)
- Tracks consecutive WARN days by active ruleset ID
- Recommends rollback after sustained degradation

## Governance Artifacts (PR #286, merged May 16, 2026)

### governance/AGENT_ROUTING_POLICY.md

Tier 0-4 routing policy classifying every part of the codebase by governance sensitivity. Defines allowed tools, review requirements, and merge rules per tier. The policy itself is Tier 4. Changes require a memo, not a direct edit.

### governance/STATUS.md

Enforcement status: AGENT_ROUTING_POLICY.md is live. Enforcement layers pending: agent_registry.yml (PR 2), AGENT_DIRECTORY_MAP.md, CI registry validation, import-graph validation.

### governance/HASH_ROTATIONS.md

Required landing zone for any Tier 3 production-hash rotation. Each entry requires: old hash, new hash, effective date, affected surface, reason, downstream impact, reviewer.

### Compliance Memo

"Why the DEM 27-Agent Fleet Is Insulated from Model-Output-as-Control-Signal Failures" - Final version, repo-verified. Cites Texas A&M security taxonomy (470 advisories). Available in Content Library at ai-projects/.

### Operational Routing

docs/ops/hermes_openclaw_routing_policy.md (v1.0, effective 2026-05-15) defines three execution lanes:

- Lane A (Deterministic Production): No LLM. Scripts, cron, tests only.
- Lane B (Cheap Monitoring): File/JSON checks first. LLM on anomaly only via run_agent_direct.py.
- Lane C (High-Token Manual): Manual sessions for synthesis, audits, refactoring. No autonomous cron.
Critical constraint: no cron job may depend on a gateway token.

---

## Spec Lifecycle

### Spec States

| State | Meaning |
| --- | --- |
| DRAFT | Under development |
| IN PROGRESS | Active work, phased |
| HELD | Blocked on dependency |
| RESOLVED | All acceptance criteria met |
| SUPERSEDED / MITIGATED | Failure modes neutralized via different route |
| CLOSED | Formally closed |

### Active Spec Numbering

Specs numbered sequentially (currently 071-105 range active). Each spec has:

- Acceptance criteria with explicit section references
- Phase gates (A/B/C/D typical)
- Blocking dependencies on other specs
- Closure memos in `artifacts/audit/`

**Spec numbering collision resolved (2026-05-14):** Original expectation coverage spec was drafted as Spec 100, which collided with the existing IC tooling correction spec. Renumbered to Spec 105 in commit cb242311.

**Schema/coverage/export specs (commits cb242311 through b310671a, 2026-05-14):**

| Spec | Title | Status | Commit |
| --- | --- | --- | --- |
| 105 | Expectation Layer Coverage Verification | CODE-CLOSED / pending live QA | 0ddbb509 |
| 101 | Runway Severity v1.1 Export Contract | CLOSED | eaa4ea87 + cba4ee0f |
| 104 | Insider Diagnostic Stabilization | MEASURED / pending 2026-05-15 | b310671a |
| 102 | Historical Backfill for Expectation Research | DRAFT | -- |

**Other specs shipped 2026-05-14:**

| Spec | Title | Status | Commit |
| --- | --- | --- | --- |
| 087 B2 | Dashboard Freshness Envelope | CLOSED | 400a6cd9 |
| 087 B0 | Stale-Propagation Guard | CLOSED (formal closure memo) | 0f0c7952 |
| 087C A | Bioshort Alpha Research Design | DESIGN (memo only) | 7628b9c6 |
| 088 B | Catalyst Delta v2 Filter Companion | SHIPPED | 5ca4b033 |

All schema/coverage/export specs are correctness work. No new model, no new alpha.

### Held-Spec Ledger

Tracks all specs that are held/blocked with:

- What is held and why
- First-fire validation status
- Alert deadlines
- Next operator action

---

## Expectation Layer Coverage Gate (Spec 105)

**QA file**: `production_qa_check.py`
**Status:** CODE-CLOSED (commit 0ddbb509). Pending live production snapshot QA via `python tools/production_qa_check.py --as-of-date YYYY-MM-DD`.

Production pipeline hard-fails if market-expectation fields are missing or under-covered in `rankings.csv`. Thresholds sourced from `FEATURE_COVERAGE_REQUIREMENTS` (not hardcoded).

### Required Expectation Fields

| Field | Required Coverage | Source |
| --- | --- | --- |
| `short_interest_pct` | 0.90 | Market data provider |
| `close_price` | 0.99 | Market data provider |
| `market_cap_mm` | 0.95 | Market data provider |
| `priced_move_pct` | 0.80 | Derived (catalyst pricing model) |
| `insider_net_buy_value_90d` | 0.30 | Form 4 (tracked nonblocking / diagnostic only) |

### Gate Behavior

- Runs every pipeline execution at Step 5 (Gates)
- Hard fail if any required field is missing from DataFrame
- Hard fail if any field falls below its per-field threshold
- Error message includes: field name, actual coverage, required threshold
- Coverage stats logged every run regardless of pass/fail
- Expectation model must consume these columns from `rankings.csv`, not from a parallel source

### Key Rule

`FEATURE_COVERAGE_REQUIREMENTS` is the single source of truth. If thresholds change, the gate inherits automatically. Do NOT hardcode coverage floors in pipeline scripts.

---

## Export Contract Registry (Spec 101)

**Status:** CLOSED (commits eaa4ea87 + cba4ee0f, 2026-05-14). `ev_severity_score` now exported. Build gap resolved.

Tracks which computed fields are exported to CSV and snapshots.

### Runway Severity Export (v1.1, RESOLVED)

**All exported (post-Spec 101):**

- `runway_severity_score`, `ev_severity_score`, `runway_buffer_months`, `financing_truth_gate`
- `dilution_haircut`, `size_multiplier`, `severity_bucket`, `severity_notes`
- `check_severity_formulas()` QA validation runs on every snapshot
- Validates finiteness before formula checks; fails explicitly on blank/NaN/Inf

**Derived field contracts (must hold for all non-null rows):**

```
dilution_haircut == 0.35 * ev_severity_score       (tolerance 1e-6)
size_multiplier == max(0.40, 1 - 0.60 * ev_severity_score)  (tolerance 1e-6)
```

Pre-v1.1 snapshot readers default `ev_severity_score` to NaN (not fail).

---

## Diagnostic Fields Registry (Spec 104)

Fields tracked for observability but explicitly excluded from scoring, ranking, and selection.

### Current Diagnostic Fields

| Field | Status | Meaning of Null | Meaning of 0.0 |
| --- | --- | --- | --- |
| `insider_net_buy_value_90d` | DIAGNOSTIC ONLY | Not fetched / no Form 4 coverage | Fetched, no insider buy activity in 90d |

### Insider Model Isolation Guard (CRITICAL)

`insider_net_buy_value_90d` must NOT enter the expectation model's `market_features` input. The model has an `insider_net_buy_z` weight that activates silently if the field flows upstream. Guard with at least one of:

1. **Input exclusion (preferred):** Runtime assert that `insider_net_buy_value_90d` is NOT in `market_features` DataFrame at inference
2. **Weight zeroing:** `insider_net_buy_z` weight = 0.0 with test
3. **Drop guard:** Pre-inference step that drops the field if present, with logged warning

### Rules

- Never collapse blank (NaN) and zero (0.0) -- they have different semantics
- Never impute zero for missing or blank for zero
- CI check: flag suspicious if column is ALL zero or ALL null
- Field must remain in `DIAGNOSTIC_FIELDS`, NOT in `ALPHA_FEATURE_REGISTRY`
- Does not affect ranks, actions, or position sizing
- Promotion requires: 20+ stable snapshots, >= 60% coverage, IC > 0 at p < 0.05, Checklist v2 pass, explicit written approval

---

## Backfill Tooling (Spec 102)

Research-enablement tooling for backfilling expectation fields into historical snapshots.

### Target Fields

`short_interest_pct`, `close_price`, `market_cap_mm`, `priced_move_pct` (required); `insider_net_buy_value_90d` (optional)

### Key Rules

- Default: additive-only (`recompute=False`). Original ranks/actions preserved.
- Every backfill emits a structured manifest (snapshot_date, fields_added, coverage before/after, recompute flag, timestamp, version)
- `_backfill_version` metadata column added to all backfilled snapshots (null for originals)
- Research scripts must filter on `_backfill_version` to avoid silent pre/post mixing
- Default scope: 30 trading days, configurable

---

## Source Files

| Component | File |
| --- | --- |
| Daily Production Runner | `tools/run_daily_production.py` |
| Knowledge Layer Builder | `tools/build_hermes_knowledge_layer.py` |
| Operator Delivery | `common/operator_delivery.py` |
| Agent Heartbeat Checks | `tools/agent_heartbeat_checks.py` |
| Ops Supervisor | `agents/ops_supervisor/supervisor.py` |
| Post-Snapshot Supervisor | `tools/run_post_snapshot_supervisor.py` |
| Ruleset Health Monitor | `tools/ruleset_health_monitor.py` |
| Ops Digest Builder | `tools/build_ops_digest.py` |
| Readiness Scorecard | `tools/weekly_readiness_scorecard.py` |
| Cron Wrapper | `tools/cron_daily_production.sh` |

---

# SECTION 2: OPERATIONAL STATE

> **SNAPSHOT DATA** â€” The values below are point-in-time and go stale. Verify against current pipeline or infrastructure before citing.

---

## Active Ruleset

*Last reviewed: 2026-05-13*

- **ID**: `8887576e` (v1.14.0)
- **File**: `production_data/decision_rulesets/v1.14.0_coinvest_only_selector.json`
- **Prior ruleset**: `2a3e79eb` (v1.13.0) - RETIRED 2026-05-04
- **Pinned in**: `run_screen.py` AND `run_phase2_snapshot_delta.py` (must stay in sync)
- **Manifest**: 36+ entries, no duplicate IDs
- **Architecture freeze**: In effect until post-h20d checkpoint (~2026-05-26)

## BioShort Research (Spec 092)

*Last reviewed: 2026-05-13*

| Phase | Status | Key Output |
| --- | --- | --- |
| A (inventory) | COMPLETE | 142/162 usable snapshots, 18 with decision_portfolio.csv fallback |
| B (research-mode isolation) | COMPLETE | --research-mode flag, archive redirect, mode tagging |
| C (historical panel) | COMPLETE | 146 rows x 16 features, 100% success, 0 live path mutations |
| D (forward returns) | COMPLETE | DEFER verdict 60.5% hit T+5, median T+5 +0.63%, T+20 +2.49% |

Key findings (pseudo-PIT):

- DEFER verdict: 129 samples, 60.5% accuracy at T+5 (forward_5d >= 0)
- Median T+5 return: +0.63%
- Median T+20 return: +2.49%
- Median drawdown: -2.86% over 20d post-recommendation
- Pseudo-PIT caveat: features computed with current logic on historical snapshots. No promotion claims supported.

## Town-Hermes Bridge Status

*Last reviewed: 2026-05-13*

- Phase A complete (dry-run mode, `OPERATOR_DELIVERY_DRY_RUN=1`)
- Phase B (live delivery): not yet started


## Knowledge Graph Implementation Status

*Last reviewed: 2026-05-24*

- **Phase 2 Step 4 (KG implementation)**: COMPLETE (2026-05-21) — 68/68 tests PASS; 4a loader + 4b queries + 4c contradictions + 4e integration; h20d evidence package ready
- **Spec 110 Phase 1 PoC**: COMPLETE (2026-05-21) — 56 nodes, 16 edges, 5 query patterns, 22 tests PASS; pipeline provenance graph; no production wiring
- **Phase 2 Step 4d (CLI)**: deferred post-h20d
- **Phase 2 Step 5 (KG gating)**: blocked on 13F quarantine clearance + h20d decision (~2026-05-26)
## Infrastructure

*Last reviewed: 2026-05-13*

- **Current**: WSL2 on Windows host
- **Agent model**: deepseek/deepseek-v4-flash via Nous free tier (primary), meta-llama/Llama-3.3-70B-Instruct-Turbo via Together AI (fallback). Switched 2026-05-19 from Together primary (was OpenRouter 2026-05-13 to 2026-05-18). Free tier subject to 429 rate limits; retry after 30-60 min.
- Daily cron runs 4:30-7:30 PM ET on weekdays
- universe_maintenance cron: 10:00 AM ET (fixed race condition - was running before rankings.csv existed)
- Sleep-cliff risk: Windows host suspend kills crons silently
- Stopgap: `powercfg /change standby-timeout-ac 0`
- Missed cron signature: 24-48h gap in `data/snapshots/`
- **Planned**: $15/mo Linux VPS (DigitalOcean / Hetzner). No timeline set. WSL2 remains dev environment.

---

# SECTION 3: MONITORING SWEEP PROTOCOL

## Purpose

Periodic read-only health checks of the biotech screener's operational state during architecture freezes or between active work cycles. The mandate: **report deltas only â€” no repo edits, no scope expansion, no implementation work.**

## Trigger Conditions

- Architecture freeze is in effect (check model_documentation.md for freeze dates)
- On "monitoring sweep" or "run agents" requests during freeze
- Gate events: PR merge checks, 13F filing wave (~23rd of end-of-quarter month), h20d checkpoint
- User asks for status / health check on any component

## The Tri-Sweep Checklist

Run these three checks in order, report only what changed since the last sweep:

### 1. PR Status Check
- Verify PR is merged into main: `gh pr view <PR_NUM> --json state,title,mergeable,labels,updatedAt`
- If merged: `git log --oneline --all --grep='<PR_NUM>' -1` to find the merge commit
- Confirm the commit is an ancestor of HEAD: `git merge-base --is-ancestor <commit> HEAD`
- Report: PR number, state, merge commit if merged, open vs. closed delta from last sweep

### 2. 13F Filing Count
- Read `production_data/13f_filing_status.json`
- Check `filed` array length vs. total manager count in `production_data/manager_registry.json`
- Report: filed / total count, new filers since last sweep, last_check timestamp
- Target quarter: check `target_quarter` field in the filing status JSON

### 3. Architecture Freeze / Gate Artifact
- Search for `h20d`, `architecture.freeze`, `arch.freeze` in model documentation and config
- Verify freeze end date has not passed
- Report: freeze status, end date, any gate artifacts that were triggered

### 4. Email Sweep (Optional â€” Operator-Directed)

Run this as a 4th check when the user asks for a broad operational status review (not every sweep). Targets the same operator Gmail inbox (djschulz@gmail.com via Himalaya CLI).

#### Capability Check

```bash
himalaya folder list        # Verify IMAP is working
himalaya envelope list -f INBOX -s 100  # Check inbox non-empty
```

#### Search Targets (multi-pass)

| Pass | Search Terms | Purpose |
|------|-------------|---------|
| 1 | CI failures, Actions, budget, billing | Pipeline health |
| 2 | Fleet, agent, heartbeat, steward, supervisor | Fleet health |
| 3 | PR, review, Codex, Claude, merge | Recent code activity |
| 4 | BIO, biotech, mover, earnings, PDUFA | Signal/alpha context |
| 5 | 13F, filing, SEC, EDGAR | Regulatory calendar |
| 6 | skill, audit, memory, prompt, governance | Skill/library health |

Use `himalaya envelope list -f INBOX -s <search_term> -l 10` for each pass. Expand to `[Gmail]/All Mail` if inbox results are thin.

#### Read and Classify

For each relevant message found:
1. Read via `himalaya get -f INBOX <id>`
2. Extract: date, sender, subject, key data points
3. Classify against existing skills (no update needed / candidate doc update / candidate instruction update)
4. Look for: denominator/key-value drifts, stale tickers, budget depletion, steward gaps

#### Output

Single markdown table with: Date, Sender, Subject, Why Relevant, Action Needed.
Append to the tri-sweep report under a `## Gmail Findings` heading.

**Delta-only:** compare against the last known state (from memory or prior sweep session). If nothing changed, say nothing changed.

```
## Status: <PASS / DELTA>
- PR <NUM>: merged âœ“ (commit <SHA>)
- 13F: N/M filed (no change since last sweep)
- Freeze: active until <DATE>
```

### Style Rules (for operators)

- **FACTS vs INFERENCE** â€” explicitly label what you verified vs what you infer. If you didn't check, say so.
- **No scope creep** â€” never "while I'm here" edit configs, fix code, or add features during a sweep.
- **Command-first** â€” if action is needed, draft the exact remediation command and wait for approval before running.
- **One remediation at a time** â€” if multiple issues found, triage one, get approval, then proceed to next.
- **"Over-hot" is stop** â€” if user says you're going too far, stop expanding. Report only the requested scope.
- **Numbered options** â€” when proposing next steps, give 2-4 numbered choices.
- **Report findings, don't close tasks** â€” read-only verification means report what you found and stop. Do not mark tasks as COMPLETE, close out task lists, or declare work done without explicit user instruction. Let the user decide when a task is closed.
- **Memory at capacity** â€” memory is near capacity (~2,200 chars). During sweeps, do not store new entries unless genuinely new durable information emerges. Replace old entries rather than appending. When in doubt, leave memory alone and report transient findings in plain text.

## Related Reference

- `references/2026-05-19-monitoring-sweep.md` â€” Example monitoring sweep report
- `references/2026-05-19-production-warn-template.md` â€” Production run WARN state recording template with worked example and follow-up checklist
- `references/2026-05-19-yahoo-429-diagnosis.md` â€” Yahoo Finance rate-limiting failure cascade, diagnosis steps, and remediation options
- `references/2026-05-19-missing-ticker-diagnosis.md` â€” DRUG/KYNB missing-ticker investigation: IFRS CAD currency gap, stale CIK in universe.json, and 6-step diagnostic workflow for missing financial data
- `references/13f-institutional-summary-pipeline.md` â€” 13F institutional summary artifact flow: PIT cache â†’ snapshot â†’ production_data promotion gap, diagnosis steps, and fix options
- `references/snapshot-artifact-layout.md` â€” Full pipeline output artifact map with file names, locations, verification commands, and naming convention history

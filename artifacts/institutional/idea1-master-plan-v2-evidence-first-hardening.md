# BTC Vol Desk Master Plan v2 — Evidence-First Hardening

> **For Hermes:** Use subagent-driven-development / TDD for implementation slices. Every item requires tests, artifact rebuild, packet verification, browser-visible QA, and an explicit done/not-done/blocker ledger.

**Date:** 2026-05-18 CST  
**Scope:** Frontend/static investor packet and dashboard, backend monitor/pipeline/CLI, strategy/backtest/signals/candidate triage, quote evidence, legal/evidence gates, packet/docs/artifacts, cron automation, tests, security/secrets, and commercial positioning.

**Evidence status:** `SCREEN-ONLY · NOT EXECUTABLE`  
**External use:** `BLOCKED` until licensed historical sources, two-counterparty real quote evidence, and counsel-approved legal/business wrapper are all verified.

---

## 1. Audit inputs and verification proof

### My audit lane

Verified locally:

- Focused BTC Vol test suite: `135 passed in 2.06s`.
- Investor packet verifier: `ok=true`, 13 artifacts checked, packet SHA matched.
- Source-intake skeleton validator: `ready=false`, `covered_source_groups=0/6`, correctly blocks placeholder licensed-source package.
- Cron job read-back: `BTC Vol daily point-in-time snapshot`, job ID `9a4c6cc52559`, enabled, script-only, next run `2026-05-18T14:30:00-07:00` = `4:30 PM CST`.
- Browser/visual audit of packet site: page renders cleanly, no executable controls visible, but dollar-value disclaimer adjacency and dense readability need improvement.

Key current packet state from `site-data.json`:

- Institutional readiness gate: `0/4 readiness gates passed`, `not-ready`.
- Quote evidence ledger: `0` valid records, `0` quote-verified candidates.
- Legal wrapper: `approved_by_counsel=false`, `draft-blocked`.
- Source-intake validation: `ready=false`, `0/6` licensed historical source groups ready.
- Current source availability: `6/6 current source groups available`, but this is current/screen/cached availability, not licensed historical readiness.

### Claude Opus 4.7 separate audit lane

Verified execution:

- Claude auth: Claude Max account.
- Model usage included `claude-opus-4-7`.
- Separate read-only adversarial audit produced P0/P1/P2 findings.
- No files modified by Claude.

Claude’s strongest verified findings were independently spot-checked in source:

- `packet_verify.py` verifies hashes/control language but does not enforce legal-wrapper approval semantics.
- `quote_evidence.py` lacks operator/counsel attestation fields and exposes raw counterparty strings in summaries.
- `site_data.py` has a fixture-vs-covered inconsistency between the source coverage matrix and tracker.
- `site_data.py` silently falls back to `btc_spot=80000` for case studies if spot is missing.
- `history.py` appends JSONL without file locking.
- `quality.py` penalizes zero-dislocation runs, which is backwards for quiet markets.
- `backtest_report.py` displays PnL numbers even when the sample gate is insufficient.
- Browser visual audit confirmed disclaimer-adjacency weakness around large case-study dollar values and backtest metrics.

---

## 2. Accepted audit findings

### P0 — Boundary, legal, and evidence-integrity fixes

#### P0.1 — Legal-wrapper enforcement is advisory, not a hard verifier gate

**Evidence:** `packet_verify.py` checks packet hash, publishability string, artifacts, secret scan, CTA scan, and control language. It does not parse `legal_wrapper_json` or enforce `approved_by_counsel` status.

**Important nuance:** For the current internal packet, `approved_by_counsel=false` should not make internal verification fail. Instead the verifier must explicitly output an external-use gate status:

- internal packet integrity: can pass;
- external readiness: must fail until counsel approval exists.

**Required fix:** Add structured verifier fields:

- `legal_wrapper_present`
- `legal_wrapper_approved`
- `external_use_ok`
- `external_use_blockers`

`external_use_ok` must be false unless all required external gates pass.

#### P0.2 — Quote evidence promotion lacks operator/counsel attestation

**Evidence:** `quote_evidence.REQUIRED_FIELDS` requires RFQ/evidence basics but no `promoted_by_operator`, `promotion_timestamp`, `legal_review_ref`, or `promotion_basis`.

**Required fix:** Add promotion lifecycle controls:

- no skip from `screen-only` to `trade-verified`;
- `quote-verified` requires two distinct real external counterparties plus operator attestation;
- `trade-verified` requires execution record, operator attestation, and legal/compliance reference;
- demo/manual placeholders can never become quote-verified.

#### P0.3 — Raw counterparty names can flow to publishable surfaces

**Evidence:** `build_quote_evidence_summary()` exposes `counterparties` from raw records.

**Required fix:** Introduce counterparty pseudonymization:

- raw ledger can store actual counterparty name internally;
- rendered site/memo/tearsheet/packet can only expose `counterparty_pseudonym` or deterministic labels like `Counterparty A/B`;
- verifier scans rendered artifacts for raw counterparty names supplied in the ledger test fixture.

#### P0.4 — Fixture-only historical sources are counted as covered in one matrix

**Evidence:** `_source_coverage_matrix()` says `2/6 required source groups covered`; `_source_coverage_tracker()` correctly says `0 covered · 4 missing · 2 fixture-only`.

**Required fix:** Coverage matrix must support `covered | fixture-only | missing`; summary must read like `0/6 licensed source groups covered · 2 fixture-only · 4 missing`.

#### P0.5 — Large dollar outputs need claim-local disclaimers

**Evidence:** Browser visual audit showed case-study and backtest dollar values visually dominate while disclaimers are section-level or small.

**Required fix:** Every large monetary output in case studies and backtest cards must include an adjacent inline pill:

- `ILLUSTRATIVE MODEL OUTPUT`
- `NOT A QUOTE`
- `NOT EXECUTABLE`
- `NOT INVESTOR-PUBLISHABLE`

The warning must live in the same DOM/card/container as the monetary value.

---

### P1 — Reliability, reproducibility, and model-honesty fixes

#### P1.1 — Run manifest append is not locked

**Evidence:** `history.append_run_manifest()` opens JSONL in append mode with no lock.

**Fix:** Add `fcntl.flock` around append on macOS/Linux. Add concurrency regression.

#### P1.2 — Site JSON is not deterministic

**Evidence:** `write_site_data()` uses `json.dumps(..., indent=2)` without `sort_keys=True`, while generated packets hash `site-data.json`.

**Fix:** Use sorted deterministic JSON for all hashed/generated JSON artifacts. Add reproducibility test: build same packet twice on same inputs, assert hashes match.

#### P1.3 — Multiple `now()` calls create timestamp drift

**Evidence:** `run_monitor.py` captures timestamps at separate stages.

**Fix:** Capture one run-level `as_of` at start, thread through source captures, report, freshness, manifest, and site.

#### P1.4 — Missing BTC spot silently becomes `$80,000`

**Evidence:** `site_data.py:694` uses `float(latest.get("btc_spot") or 80000)`.

**Fix:** Remove fallback. If spot missing, suppress case studies and render `spot unavailable — case studies suppressed`.

#### P1.5 — Quote-evidence dummy path masks unavailable state

**Evidence:** `site_data.py` passes `__no_quote_evidence_for_latest_run__.jsonl` when no quote path exists.

**Fix:** Let `load_quote_evidence_ledger(None)` return a typed `unavailable` result. Render unavailable distinctly from `empty but available`.

#### P1.6 — Backtest sample gate still displays performance-like PnL

**Evidence:** Backtest reports mark sample gate insufficient but still display gross PnL and trade PnL.

**Fix:** When `sample_gate_ready=false`, suppress headline PnL and scenario PnL with `—`; keep diagnostics and controls only.

#### P1.7 — Effective cost is computed but not reflected per scenario

**Evidence:** `effective_cost` appears in robustness metrics, while scenario dict records only input `cost_per_trade`.

**Fix:** Add `effective_cost_per_trade` to scenario outputs and UI labels.

#### P1.8 — Quality score penalizes quiet markets

**Evidence:** `quality.py` subtracts points when `dislocations == 0`.

**Fix:** Remove that penalty. Add penalty only for suspicious candidate explosions or source/parsing degradation.

#### P1.9 — Fetchers need bounded retry/backoff and source-level failure classification

**Evidence:** Current fetch paths are mostly single-attempt calls.

**Fix:** Add retry helper: 3 attempts, exponential backoff with jitter, 4xx fail-fast, 5xx/network retry. Record retry counts and final failure class in run manifest.

#### P1.10 — Packet verifier trusts builder-provided missing-artifacts metadata

**Evidence:** Verifier errors if `missing_artifacts` is non-empty, but does not independently assert required labels are represented.

**Fix:** Add `EXPECTED_PACKET_LABELS`; verifier fails if each required label is not either present in artifacts or explicitly listed as missing.

---

### P2 — Polish and maintainability

- Replace hardcoded wording constants with central control-language constants where copy is reused.
- Add explicit parse-rejection logs in historical ingest.
- Align UTC/CST timestamp fields across Databento/current sources.
- Add units to backtest tables: vol pts, USD, synthetic PnL.
- Make vendor endpoints/config overridable while preserving safe defaults.
- Add dashboard/site HTML tests for forbidden CTAs, raw counterparty leaks, and required control labels.
- Move the investor ask higher in the packet narrative, above deep operational evidence sections.
- Increase font size/contrast for dense tables/hashes/source diagnostics.

---

## 3. Disputed / narrowed Claude findings

These are useful but should not be implemented literally without nuance:

1. **“Fail packet verifier if counsel not approved.”**  
   Narrowed: internal packet integrity should still pass; external-use readiness must fail explicitly.

2. **“Fixture coverage is a P0.”**  
   Accepted as P0 for investor/commercial interpretation, not because the backtester currently treats fixtures as live. The product already has a readiness gate, but the matrix headline is misleading.

3. **“Raw zip CTA scan skip may hide forbidden language.”**  
   Keep skip for third-party raw captures, but add a separate contamination report for raw captures so vendor marketing text does not fail packet integrity while still being visible.

---

## 4. New master implementation plan

### Phase A — Hard evidence/legal gates first

**Exit criterion:** Internal packet verification still passes, but external-use readiness is a first-class machine-readable failure until legal, quote, and licensed-source gates are complete.

1. Add external-use gate fields to `packet_verify.py`.
2. Add tests for missing legal wrapper, draft wrapper, approved wrapper, and external-use blocked state.
3. Add `EXPECTED_PACKET_LABELS` enforcement.
4. Rebuild packet and verify `internal_integrity_ok=true`, `external_use_ok=false`.
5. Browser QA Legal / Readiness sections.

### Phase B — Quote evidence lifecycle and pseudonymization

**Exit criterion:** No raw counterparty name appears in rendered site/packet/memo/tearsheet; promotion requires operator/legal attestation.

1. Extend quote schema with promotion attestation and pseudonym fields.
2. Add state-machine validation tests.
3. Add same-counterparty duplicate tests and two-real-counterparty promotion tests.
4. Update site data and renderers to expose pseudonyms only.
5. Add packet verifier raw-name leak regression.

### Phase C — Source coverage truth and case-study safety

**Exit criterion:** Fixture-only never reads as licensed coverage, and no case-study dollar value can appear without adjacent model-output/non-executable warning.

1. Fix `_source_coverage_matrix()` fixture classification.
2. Update source-diagnostics UI badges and copy.
3. Remove `$80,000` spot fallback.
4. Suppress case studies when spot unavailable.
5. Add inline control pills to every large case-study dollar value.
6. Visual/browser QA.

### Phase D — Backtest honesty hardening

**Exit criterion:** Insufficient sample gates cannot produce headline performance-like PnL in packet/site.

1. Suppress PnL when sample gate fails.
2. Add `effective_cost_per_trade` per scenario.
3. Add units and synthetic-fill labels in every table/card.
4. Bind input manifest hash into backtest JSON.
5. Add reproducibility and zero/single/all-win/all-loss tests.

### Phase E — Reproducibility and audit trail integrity

**Exit criterion:** Two builds on identical inputs are byte-identical for hashed artifacts; concurrent manifest appends are safe.

1. Deterministic JSON dump everywhere hashed.
2. Atomic/locked JSONL manifest append.
3. One run-level timestamp threaded through all outputs.
4. Add packet rebuild reproducibility test.
5. Add concurrent append regression.

### Phase F — Operational resilience

**Exit criterion:** Transient source failures are retried and classified without leaking secrets or producing ambiguous failures.

1. Add retry/backoff helper for external fetches.
2. Store per-source retry count/failure class in run manifest.
3. Add secret-redacted error wrapper around credential-bearing fetchers.
4. Add tests for retryable 5xx, fail-fast 4xx, and redacted exception output.
5. Manual live run and evidence bundle verification.

### Phase G — Frontend/packet polish pass

**Exit criterion:** Packet reads like an institutional evidence memo, not a dense internal dashboard, while preserving controls.

1. Move investor ask and executive narrative higher.
2. Increase dense-section font size and contrast.
3. Rename generic green statuses to precise labels like `source file present` or `configured-source healthy`.
4. Add a sticky/persistent top warning banner.
5. Add browser visual QA screenshots to completion proof.

---

## 5. Verification plan for every implementation slice

Each slice must run:

1. Focused RED/GREEN pytest for the changed behavior.
2. Full BTC Vol focused suite: `python -m pytest --tb=short -q tests/test_btc_vol*`.
3. Live/internal monitor run where applicable: `python -m institutional_btc_vol.cli run artifacts/institutional/data`.
4. Packet rebuild: `python -m institutional_btc_vol.cli build-packet artifacts/institutional/data artifacts/institutional/investor-packet`.
5. Packet verification: `python -m institutional_btc_vol.cli verify-packet artifacts/institutional/investor-packet`.
6. Static site browser QA: open `artifacts/institutional/investor-packet/site/index.html`, inspect console, visual screenshot/vision check.
7. Final ledger: Done / Not done / Blocked / Proof.

---

## 6. Current status after audit

**Done:**

- Complete dual audit run: my audit + separate Claude Opus 4.7 audit.
- Test baseline verified green: `135 passed`.
- Packet verifier baseline verified green for internal integrity.
- Browser visual audit completed.
- New master plan written here.

**Implementation progress after plan creation:**

- Phase A started and first verifier slice completed:
  - `packet_verify.py` now reports legal-wrapper presence/approval and a separate `external_use_gate`.
  - Internal packet integrity can pass while external-use readiness remains blocked.
  - Verifier now enforces required packet artifact labels including legal wrapper artifacts and quote evidence.
  - `investor_packet.py` now treats legal wrapper JSON/Markdown as required packet artifacts.
- Phase B started and first quote-evidence lifecycle slice completed:
  - `quote_evidence.py` now requires operator/legal promotion attestation for `quote-verified` and `trade-verified` evidence.
  - Quote evidence summaries now pseudonymize counterparties for rendered surfaces and set `raw_counterparties_redacted=true`.
  - `quote_templates.py` now includes promotion/legal-review fields.
  - `investor_site.py` renders pseudonymized counterparty labels when available.

**Not done:**

- Remaining Phase B state-machine depth and packet-level raw-counterparty leak verifier are not complete.
- P0.4/P0.5 and all P1/P2 fixes remain to be implemented.

**Blocked / external gates:**

- Licensed historical source readiness remains blocked until real licensed/replay-ready files replace skeleton placeholders.
- Quote verification remains blocked until real two-counterparty indicative quotes are captured.
- External use remains blocked until counsel approves the legal/business wrapper.
- Everything remains `SCREEN-ONLY · NOT EXECUTABLE`.

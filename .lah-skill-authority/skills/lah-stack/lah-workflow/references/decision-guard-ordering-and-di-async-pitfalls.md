# Decision-engine guard ordering + DI-async pitfalls (from CLOE_EXOCLICK_ZONE_AUTONOMOUS_MONITORING_AND_FAST_CUT_WIRING_V1, 2026-08-01)

Three real defects surfaced during certification of the zone fast-cut wiring. All three are
class-level traps for any MIXED mission that builds a decision engine with operator
simulations and DI'd persistence — not ExoClick-specific.

## 1. Decision-guard ordering: stale/per-zone-cap before the CUT path

**Symptom:** New decision engines that evaluate `STATS_STALE` / `PER_ZONE_LOSS_CAP` /
`ABNORMAL_SPEND_VELOCITY` BEFORE the full CUT path return REVIEW instead of CUT for an
eligible zone; stale data can produce a monetary signal instead of WATCH.

**Observed:** `zone-decision-engine.js` (Codex lot 2) checked `ABNORMAL_SPEND_VELOCITY`
then `PER_ZONE_LOSS_CAP` before the CUT block. Operator simulations S1/S4/S20 failed:
`spend=50` hit the 5000-cent per-zone cap exactly (spend × 100 = cap), so the zone was
`REVIEW_ZONE PER_ZONE_LOSS_CAP` instead of `CUT_ZONE`. S12 failed because stale data was
also masked by the cap check → `REVIEW` instead of `WATCH STATS_STALE`.

**Fix — order guards fail-closed:**
1. missing identifiers → protected zone/campaign → hard stops (total loss cap, tracking
   broken + spend, impossible statistics)
2. STALE check (never cut stale — data age must gate everything monetary)
3. window too short → insufficient sample → conversions unavailable → tracking degraded
4. CUT path (all conditions ANDed: min spend, min clicks, explicit zero conversions,
   healthy tracking, window ok, fresh, grace passed, unprotected)
5. abnormal velocity / per-zone-cap signals (REVIEW, escalation)
6. grace-not-passed → KEEP

**Simulation-fixture calibration:** scenarios that expect a CUT must use values that
deliberately clear the caps — spend below the per-zone cap (e.g. 40 USD when cap = 5000
cents), `last_event_at` in the past (grace period passed), fresh `source_timestamp`. The
failures looked like engine bugs but were fixture misalignment; diagnose by reading the
actual decision output before patching.

**Invariant test to add:** stale data can never produce a cut; protected zone/campaign can
never produce a cut; missing conversions (null) never treated as zero when tracking is
broken.

## 2. Injected async dependency not awaited

**Symptom:** A DI'd writer (e.g. `deps.writeCampaignMemory`) is called without `await`.
The module returns `{ ...receipt, record }` where `receipt` is a Promise → object spread
of a Promise yields `{}` → receipt silently lost. Unit tests pass by coincidence when they
only assert the side effect (`written` variable set) and never the return value.

**Observed:** `campaign-memory-zone-events.js` (Codex lot 4): `const receipt =
deps.writeCampaignMemory({...})` missing `await`; the memory-wiring test asserted `written`
(truthy) so it passed, while the return receipt was empty. Caught by invariant I10 which
asserted `record.ok === true` on the RETURNED value.

**Fix:**
- Make the caller `async` and `await` every DI'd callback that returns a Promise.
- Tests must assert the RETURNED receipt shape (`result.ok`, `result.record`), not just
  the side effect.
- Static invariant tests (secret redaction, forbidden-call scans, return-shape checks)
  are the net that catches this class of silent data loss.

## 3. Lesson: Codex TASK_PASS is a self-report, not a verdict

Five Codex lots returned TASK_PASS in this mission, yet certification still found three
real defects (above) that the lot-authored tests did not catch. The `node --test` run in
`required_commands` only proves what the lot's own tests assert. Independent verification
is mandatory:
- re-run every lot's test file yourself,
- import every new module and exercise the REAL entry points (shadow cycle, gate, executor),
- add invariant tests Codex won't write for itself (static forbidden-call scan with
  comment-stripping, redaction check, protected-zone/campaign never-cut, stale-never-cut),
- review security-critical code (live gates, mutation endpoints, token handling) line by line.

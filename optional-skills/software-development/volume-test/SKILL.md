---
name: volume-test
description: Check web app behavior as stored datasets grow.
version: 1.0.0
author: neto (zqovark), Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [qa, testing, data, performance]
    category: software-development
    related_skills: [dogfood]
---

# Volume Test Skill

Check a web application's functional correctness and timings at increasing stored-data volumes while keeping workload and concurrency constant. This complements `dogfood`; it does not perform concurrent-user load tests, find a breaking point, fix the target, or benchmark infrastructure.

## When to Use

Use for a local application or an explicitly designated test environment, with source access, understood data contracts, and permission to create and remove run-owned data. A URL alone is insufficient for this version.

## Prerequisites

- Python 3 standard library, application repository, target URL, and an understood seeder or documented batch API.
- Use `terminal`, `read_file`, `search_files`, `write_file`, and `patch` for inspection and bounded helpers. Use `browser_exec` for UI completion checks, screenshots, and console/network evidence when available.
- Determine entities, absolute volumes, seed, run ID, output directory, request timeout, maximum records, duration budget, and functional criteria. Discover these from code where possible; ask only for missing prerequisites.
- Linux is the validated platform. No mandatory cloud, paid service, Docker, or load-testing package. Browser and Hermes model availability are separate prerequisites for agent/UI validation.

## How to Run

Ask: “Use volume-test on this local web application and repository, at 100, 1,000 and 10,000 total records with seed 42 and concurrency one; preserve measurements and clean up this run.”

Read `references/methodology.md` before preparing data. Use the existing application's documented preparation/read/cleanup paths; the bundled script is not a universal application adapter. Keep application-specific integration outside the skill runtime.

## Quick Reference

| Artifact | Purpose |
| --- | --- |
| `plan.json` | Contracts, absolute volumes, criteria, queries, limits, cache policy |
| `manifest.json` | Run ownership, confirmed writes/counts, reconciliation and cleanup |
| `samples.json` | Raw request/UI timings with explicit units, errors and timeouts |
| `report.md` | Comparison, correctness findings, coverage, limitations and cleanup |
| `scripts/volume_support.py` | Deterministic integers, atomic checkpoints, ordered row comparison, metrics |
| `templates/volume-test-report.md` | Report structure |

## Procedure

1. **Inspect.** With `read_file` and `search_files`, identify entities, relationships, migrations, seeders, endpoints, UI completion signals, stable sort keys and export behavior. Read applicable repository instructions. Do not inspect unrelated private repositories or reuse credentials/data from other projects.
2. **Plan.** Save the plan before creating records. Specify the counted entity, absolute totals, data distributions, unchanged query parameters, page size, concurrency, repeats, limits and run-scoped cleanup. If creation, verification or cleanup is ambiguous, stop at a plan and report the blocker instead of inventing SQL or an adapter.
3. **Baseline.** Confirm the initial persisted counts and relationships, separating unrelated sentinel records. Prepare the first absolute volume and run reference operations. Separate preparation time from request timings.
4. **Grow and verify.** Preserve the logical prefix when increasing to each absolute total. Use deterministic values by seed/index and a separate run identity. Use bounded seeder/API batches, checkpoint confirmed effects after each batch, and reconcile partial/ambiguous writes with reads before retrying. Never blindly retry a write. Confirm actual entity and relationship counts; an incomplete setup is not a completed stage.
5. **Exercise.** Use the same sequential operations at each stage: list, boundary/final pages, search, filters, stable sorting with tie-breaks, totals, and full filtered export where implemented. Generate expected results independently of production queries. Compare complete identifier multisets, order, fields and totals, not just HTTP status. Traverse pages to detect missing/duplicate records.
6. **Measure.** Label the first request separately; call it cold only if the relevant caches were actually controlled. Warm up, then collect at least ten warmed measurements per operation. Persist raw milliseconds, sample counts, errors, timeouts and correctness results as they happen. `summarize` includes failures in its timing population; use median/min/max and do not overinterpret small-sample percentiles.
7. **UI.** When a browser is available, keep viewport and parameters fixed. Measure time to an independently verified correct UI state with a DOM/result condition, not page-load alone or a fixed sleep. Capture screenshots and relevant console/network errors. Separate UI timings from HTTP timings; if unavailable, mark UI unrun, not passed.
8. **Compare and report.** Use `templates/volume-test-report.md`. Report observed changes and ratios against baseline without inventing an SLA. Distinguish observation from hypothesis; growth alone does not identify a database/network/memory cause. Preserve partial results on interruption or limits; mark unexecuted operations explicitly. Do not create a ticket for every finding.
9. **Cleanup.** In a finalization path, remove only run-owned records through the understood application contract. Verify residual counts and unrelated sentinels; preserve manifests and evidence. If cleanup fails, record the precise residuals and safe next action. Never truncate, reset a shared database, or remove unrelated records.

## Pitfalls

- Totals are not increments: growing 100 → 1,000 means adding 900, not 1,000.
- Product/entity counts and relationship/table-row counts are distinct.
- A successful HTTP response does not prove complete exports or correct pagination.
- Seed reproducibility must not depend on UUIDs, batch size, or hash randomization.
- Do not fabricate a slowdown with sleeps or classify an injected defect as spontaneous.
- Stop growing when the record/duration/error budget is reached. Cleanup still needs a bounded attempt; retain evidence even when that attempt fails.
- Do not claim a script run proves Hermes used this skill. Record actual agent invocation/loading separately; missing model/browser validation remains a gap.

## Verification

A finished report has actual counts at every completed stage, raw measurements and correctness evidence for fixed operations, explicit cache/sample policy, coverage and gaps, a reproducible configuration, and verified scoped cleanup. Uncontrolled cache, unavailable browser/agent runs and externally blocked operations must remain explicit. Local results describe that application and environment only.

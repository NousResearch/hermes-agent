# Debt Dimensions and the Ledger Contract

What each dimension looks for, how findings are graded, and how a rerun reconciles. Detector commands
count only when their output is observed.

## Orientation first

- **Stack truth** — the manifests (package/build/dependency files), never memory of the tree.
- **Churn ranking** — `scripts/churn_rank.py`: the most-changed files are where debt costs the most.
- **Size ranking** — the largest tracked files by line count (same script).
- **Gate inventory** — the test suites, type checkers, linters and CI entry points that already
  exist. Debt a gate already catches is a gate-configuration finding, not new debt.

High churn + high size + low test coverage is the audit's priority corner.

## The nine dimensions

| Dimension (id prefix) | What to look for |
| --- | --- |
| Architectural decay (`arch`) | cyclic imports, god modules, bypassed layers, boundary leaks between packages |
| Consistency rot (`cons`) | competing patterns for one job — two HTTP clients, three error styles, mixed naming |
| Type and contract gaps (`type`) | untyped public surfaces, `any`-equivalents, implicit schemas parsed in many places |
| Test debt (`test`) | untested high-churn paths, skipped or stub tests, assertions that cannot fail |
| Dependency and configuration debt (`dep`) | unpinned or abandoned dependencies, drifted config copies, secrets handled by convention |
| Performance and resource debt (`perf`) | unbounded caches and queues, N+1 access patterns, sync work on hot paths |
| Error-handling and observability debt (`err`) | swallowed exceptions, bare retries, failures invisible to logs or metrics |
| Security hygiene (`sec`) | credentials in the tree, injectable string building, permissive defaults |
| Documentation drift (`doc`) | READMEs and comments contradicting the code, dead runbooks, stale generated artifacts |

Per-stack detectors (linters, dead-code finders, dependency and coverage audits) are chosen from the
manifests read in orientation.

## Grading

- **Severity**: `critical` (active correctness or security risk), `high` (costs every change in the
  area), `medium` (costs some changes), `low` (cosmetic or contained).
- **Effort**: `S` (one bounded change), `M` (a few files, one review), `L` (needs phased work).
- **Top fixes** rank by severity first; **quick wins** are severity ≥ medium with effort `S`.
- A `critical`/`L` finding gets a phased plan; the ledger never recommends a rewrite.

## The ledger

One row per finding: `id`, `dimension`, `file:line`, `severity`, `effort`, `recommendation`. Stable
`id` = dimension prefix + path + short slug (`test.src-routing-chat.untested-fallback`), so reruns
match rows without archaeology. After the table: top fixes, quick wins, and the mandatory **looks bad
but is actually fine** section.

## Rerun reconciliation

With a previous ledger present, reconcile before writing:

- **RESOLVED** — the cited evidence is gone at the cited location; name the commit or the observed
  absence.
- **CARRIED** — still present; keep the id, increment its age. A finding carried three audits is a
  prioritization finding about the ledger itself.
- **NEW** — not matched by any prior id.

Prior ids that no longer match the tree are mapped by dimension + path before anything is declared
RESOLVED.

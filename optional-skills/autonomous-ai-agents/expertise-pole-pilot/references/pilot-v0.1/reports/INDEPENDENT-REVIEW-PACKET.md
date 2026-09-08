# Independent review packet — SEO pilot v0.1

- **Status:** `PREPARED — not executed`
- **Scope:** Domain / SEO pilot
- **Owner:** Human steward until an independent reviewer is named
- **Maturity:** Experimental — review-enabling artifact
- **Last reviewed:** 2026-09-08
- **Freshness rule:** Refresh after every material pilot, source, or evaluation change
- **Evidence requirement:** Reviewer declaration, inspected artifact IDs, reproducible test output, and a completed defect table
- **Exit criterion:** A distinct reviewer delivers a signed-by-role verdict with severity, evidence, corrective owner, and retest result
- **Escalation:** Any request for source retrieval, credentials, spend, live-site access, publication, or production action goes to Vincent
- **Linked records:** `INDEPENDENT-REVIEW.md`, `MATURITY-DECISION.md`, `TRACEABILITY-MATRIX.md`, `support-pole/templates/contradictory-review-template.md`

## Declaration of independence

The reviewer must declare their role and date, then affirm all of the following before reviewing:

1. they did not design or author the artifact under review;
2. they are not the workflow orchestrator or a decision approver for the same run;
3. they received the input set and the rubric before forming a verdict;
4. they will record uncertainty and cannot silently substitute a new source, live observation, or recommendation.

If any statement is false or unknown, the result is a useful local check but **not** an independent review. This packet must not state that a review was executed until this declaration and an actual completed defect table exist.

## Review input set

Review the following local artifacts as a fixed package:

- `support-pole/` — twelve canonical governance documents and four templates;
- `domains/seo/` — nine canonical SEO documents, six profiles, three skill contracts, one workflow, fixtures, runs, and templates;
- `reports/PHASE-0-INVENTORY.md`, `COVERAGE-REPORT.md`, `TRACEABILITY-MATRIX.md`, `RESEARCH-COMPARISON-STATUS.md`, `LOCAL-ADVERSARIAL-CHECK.md`, `INDEPENDENT-REVIEW.md`, and `MATURITY-DECISION.md`;
- `domains/seo/evaluations/runs/RUN-001.md` plus `CASE-01.md` through `CASE-06.md`;
- `evaluations/audit_pack.py` and `tests/test_audit_pack.py`.

The reviewer may run only the deterministic local test unless Vincent explicitly authorizes a different action:

```bash
python3 -m unittest discover -s tests -p 'test_audit_pack.py' -v
```

Expected pre-review state: 7 tests pass after this packet is added. The test proves document contracts and the packet’s readiness, not SEO performance or review independence.

## Mandatory attacks

Test the documented system against these failure modes, with a cited file and line/section for every finding:

1. duplicate responsibility, generic “does everything” role, or ownerless capability;
2. a missing, stale, contradictory, or falsely authoritative source claim;
3. confusion between fixture evidence and a live-site observation;
4. a recommendation that exceeds its contract, lacks owner/approval/rollback, or invites production action;
5. a workflow state, handoff, or escalation that cannot be verified from its inputs and outputs;
6. over-automation, confirmation bias, hidden assumption, or impossible quality gate;
7. a rubric that would miss the injected error in `CASE-06`;
8. promotion language that exceeds the evidence currently available.

## Non-objectives

- Do not perform a live SEO audit, crawl, Search Console action, site change, publication, account creation, message, credential use, or spend action.
- Do not replace the blocked primary-source research with memory, snippets, assumptions, or invented references.
- Do not change the pack while reviewing; record findings first so the designer can make attributable corrections.
- Do not award `promote` based solely on document completeness or a passing structural test.

## Expected reviewer deliverables

1. A completed copy of `support-pole/templates/contradictory-review-template.md` with an explicit declaration of independence.
2. One row per defect: ID, severity, claim challenged, evidence, correction requested, owner, status, and post-correction result.
3. A verdict: `retain experimental`, `quarantine`, or `promote` only if all documented promotion gates are actually met.
4. Residual risks and any request requiring Vincent, especially source research, baseline execution, or access to a real corpus.
5. A handoff to the integrator that identifies exactly which test or review must be rerun after each correction.

## Closure protocol

The integrator copies the reviewer’s completed output into `reports/INDEPENDENT-REVIEW.md`, preserves the original reviewer declaration, links each correction to the defect ID, and reruns the deterministic regression suite. `MATURITY-DECISION.md` may change only after the independent reviewer’s verdict and retest evidence are present.

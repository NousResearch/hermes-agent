---
name: source-spike-artifact-checker
description: Use when closing or reviewing Spearhead source-spike artifacts. Provides a closure-summary template and checker for provenance, license/access, verdict, routing, downstream decision, CLOSE_READY, and Notion closure hygiene.
version: 1.0.0
author: Gond / Project Spearhead
license: private
metadata:
  hermes:
    tags: [spearhead, source-spike, provenance, closure, checker]
    related_skills: [gond-reliable-worker, kanban-worker, third-party-import-redflag-scan]
---

# Source-Spike Artifact Checker

## Overview

Use this skill for Spearhead source-spike closure artifacts: repository/library/article extraction tasks that produce `source-spike.md` and `closure-summary.md` under a durable artifact directory.

The goal is boring consistency. Every closure summary must be parseable enough for EMA/Gond to decide whether a source can be closed, routed to a specialist, or turned into a follow-up implementation/research card without reopening the whole spike. The enforced field list is also captured in `references/source-spike-artifact-schema.md`. When third-party automation content is importable or recommended for adoption, closure summaries should also carry the local import-audit sidecar fields from `third-party-import-redflag-scan --summary-json`: scanner command, exit code, counts, audit status/risk, credential/env declarations and mismatches, external services, network-write/API mutation findings, and accepted/rejected findings.

## When to Use

Use this when:

- closing a source-spike Kanban card;
- reviewing a source-spike closure from another worker;
- deciding whether `CLOSE_READY` is justified;
- migrating older source-spike summaries into a more consistent shape;
- creating a downstream specialist card from an extraction.

Do not use this as a substitute for actual source inspection. The checker validates artifact hygiene, not technical truth.

## Required Artifact Files

Preferred source-spike directory shape:

```text
<artifact-dir>/
  source-spike.md
  closure-summary.md
```

`closure-summary.md` is the handoff gate. It must contain the minimum fields below. `source-spike.md` may contain richer analysis, excerpts, comparisons, and provenance details, but the closure summary must stand alone.

## Required Closure Fields

A closure summary is acceptable only if it covers these gates:

1. Provenance
   - source URL or source identifier;
   - retrieval method (API/raw/git clone/browser/etc.);
   - immutable revision when available: commit hash, tag, release, paper date, or explicit `not available` rationale.
2. License/access
   - license conclusion for the inspected source;
   - access status: public, auth required, paywalled, local/private, etc.;
   - explicit note if third-party links/dependencies have separate license review risk;
   - if the source recommends importing external skills, MCP configs, workflow files, agent rules, hooks, or executable snippets, run or cite the `third-party-import-redflag-scan` gate and summarize any `warn` / `error` findings.
3. Verdict
   - one of: `ADOPT`, `ADOPT SELECTIVELY`, `SPIKE`, `MONITOR`, `NO_ADOPT`, `REJECT`, or a clear equivalent;
   - whether this is pattern extraction, implementation candidate, or no-op closure.
4. Specialist routing
   - `NO_HANDOFF` if no downstream worker is needed; or
   - specialist lane/profile/domain and why: Gond implementation, research, security, UX, data, Notion/admin, etc.
5. Downstream decision
   - exact next action: close, create follow-up card, needs approval, monitor, or backlog;
   - if a follow-up is recommended, state scope and approval gate.
6. CLOSE_READY
   - explicit `CLOSE_READY: yes` or `CLOSE_READY: no`;
   - if `no`, name the blocking missing evidence.
7. Notion closure hygiene
   - explicit statement of Notion status: no Notion writes performed, Notion update required, Notion already updated, or Notion deliberately out of scope;
   - source-spike workers must not bulk-edit Notion unless separately approved.

## Recommended Template

Use `templates/closure-summary-template.md` as the canonical starting point. Keep headings stable; stable headings make the checker and future migration safer.

## Checker

Run the checker from any shell with Python 3:

```bash
python skills/software-development/source-spike-artifact-checker/scripts/source_spike_checker.py /path/to/artifact-dir
```

You can pass either an artifact directory or a `closure-summary.md` path. For multiple artifacts:

```bash
python skills/software-development/source-spike-artifact-checker/scripts/source_spike_checker.py \
  /path/to/spike-a /path/to/spike-b /path/to/spike-c
```

Exit codes:

- `0`: every checked artifact passes required gates;
- `1`: at least one artifact has missing/weak gates;
- `2`: usage or file-access error.

The output is intentionally text-first so it can be pasted into Kanban comments.

## Review Policy

If the checker fails:

- do not mark the source-spike as DONE/CLOSE_READY;
- either patch the summary with the missing fields or block with a precise reason;
- if the failure is due to a checker false positive, add the missing synonym/pattern to the checker and rerun against at least three existing summaries.

If a summary passes the checker but the artifact content is untrustworthy, trust the human/engineering review over the checker. The checker is a floor, not a certification authority.

## Common Pitfalls

1. `CLOSE_READY yes` without a downstream decision. Closure readiness requires a stated next state, not just a status word.
2. License omitted because the source is public. Public access is not a license grant.
3. Routing omitted because no one is needed. Say `NO_HANDOFF`; silence is ambiguous.
4. Notion omitted. Say whether Notion was untouched, updated, required, or out of scope.
5. Commit omitted for GitHub repos. If the source has revisions, capture one. If not available, state why.
6. Follow-up recommendations without scope. A downstream card needs bounded work, non-goals, and evidence requirements.

## Verification Checklist

- [ ] `source-spike.md` exists or absence is explained.
- [ ] `closure-summary.md` exists and passes the checker.
- [ ] Provenance includes URL/source, retrieval method, and revision or rationale.
- [ ] License/access is explicit.
- [ ] If external skill/MCP/workflow/importable automation is recommended, the `third-party-import-redflag-scan` gate is cited and `warn` / `error` findings are summarized.
- [ ] Verdict is explicit.
- [ ] Specialist handoff is `NO_HANDOFF` or names a lane/profile/domain.
- [ ] Downstream decision/next action is explicit.
- [ ] `CLOSE_READY` is explicit and justified.
- [ ] Notion closure status is explicit.
- [ ] Checker output is attached to the Kanban handoff.

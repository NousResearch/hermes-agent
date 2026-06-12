---
name: maintainer-sweep
description: "ClawSweeper/Clownfish-style GitHub maintenance for Hermes: read-only sweeps, durable reports, proposal-first workers, and deterministic apply gates."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [github, maintenance, agents, safety, ci, pr-review]
    related_skills: [github-pr-workflow, github-code-review, github-issues, codex, hermes-agent]
---

# Maintainer Sweep

Use this when the user asks Hermes to maintain a GitHub repo, clean up a backlog, review PRs/issues in bulk, run a guarded merge train, or make ClawSweeper/Clownfish-style agentic development safer.

The pattern is: **LLMs propose; deterministic code applies.** Do not let a model comment, close, push, or merge directly from its own judgment.

## Core invariants

1. **Proposal first** — every issue/PR gets a durable report before any public comment or mutation.
2. **One durable record per item** — include source URL, snapshot hash, evidence, recommendation, proposed comment/action, and gate status.
3. **Worker without write credentials** — Codex/Hermes/Claude workers may inspect, classify, and produce artifacts, but must not hold GitHub write tokens.
4. **Deterministic applicator** — a separate script/tool re-fetches live GitHub state and owns comments, labels, closes, branch pushes, and merges.
5. **Live-state recheck** — immediately before every mutation, verify target repo, item number, current head SHA/branch/base, labels, author/permission, CI, mergeability, and snapshot freshness.
6. **Explicit allow gates** — no mutation unless the specific capability is opened (`ALLOW_COMMENT`, `ALLOW_CLOSE`, `ALLOW_FIX_PR`, `ALLOW_MERGE`, etc.). Keep gates closed by default and reset them after the execution window.
7. **Security out of scope** — secrets, auth bypasses, vulnerability reports, supply-chain issues, exploitability, privacy leaks, and broker/live-trading changes go to human/security review.
8. **Audit ledger** — append every report, promotion, block, and mutation decision to a machine-readable ledger.

## Hermes-native first pass

From a repo checkout, run a read-only sweep:

```bash
python scripts/maintainer_sweep.py --repo OWNER/REPO --state-dir .hermes/maintainer --limit 50
```

For tests or offline runs, use a fixture:

```bash
python scripts/maintainer_sweep.py --repo OWNER/REPO --source-file /tmp/items.json --state-dir .hermes/maintainer --json
```

Outputs:

```text
.hermes/maintainer/repos/OWNER__REPO/
  items/issue-123.md
  items/pr-456.md
  ledger.jsonl
  dashboard.md
  summary.json
```

The shipped `scripts/maintainer_sweep.py` is deliberately read-only. It writes `mutation_allowed: false` and `action_state: proposal` so later lanes cannot accidentally treat a report as permission to act.

## Recommended development loop

1. **Sweep**: generate read-only reports and dashboard.
2. **Cluster**: group reports by implementation surface/risk boundary.
3. **Plan**: create a small fix/cleanup plan from one safe cluster.
4. **Worker**: delegate implementation to a worker without write credentials, preferably on a worktree/branch.
5. **Verify**: run tests, inspect diff, and request review if risk is non-trivial.
6. **Apply**: only deterministic code or the operator performs public comments/closes/merges after live-state gates pass.
7. **Ledger**: record why each item was applied, blocked, skipped, or escalated.

## Promotion rules

Safe candidates for automation:

- docs typo or broken link
- deterministic test repair
- narrow CI/config fix
- stale generated dashboard/report update
- duplicate/superseded close with exact canonical reference and no fresh target-side activity

Always human-gated:

- security/privacy/supply-chain/auth findings
- secrets or credential handling
- destructive data migrations
- infrastructure permissions or deployment changes
- financial/trading/broker/live execution changes
- ambiguous product decisions or active maintainer disagreement

## Merge gate checklist

Before any automerge or guarded merge train:

- base branch is expected
- source branch prefix is allowlisted
- PR is not draft
- mergeable is clean/mergeable
- required CI is green at the exact head SHA
- no requested changes or unresolved threads
- no unreviewed security/infra/broker side effects
- live PR head SHA matches reviewed SHA
- explicit merge gate is open for this run

## Reporting style

Keep summaries terse and evidence-centric:

- what was swept
- report/dashboard path
- candidate clusters
- applied actions, if any
- blocked/skipped reasons
- next safe action

Do not say “done” for proposed reports alone. Say “read-only sweep complete” unless the applicator actually mutated and verified GitHub state.

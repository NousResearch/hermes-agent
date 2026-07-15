---
name: provider-validation-harness
description: Use the Hermes provider candidate evaluator.
version: 2.0.0
author: Drew Schuyler (github.com/drewschuyler); Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [providers, evaluation, validation, receipts]
    related_skills: [hermes-agent]
---

# Provider Validation Harness Skill

Use this workflow to produce local, receipt-backed evidence for a configured
provider path. It does not call a provider by itself, change routing, install a
model, edit user configuration, or grant a promotion decision.

## When to Use

Use `validate` for a fast six-case tier-0 compatibility smoke. Use `evaluate`
for a pinned candidate-versus-incumbent `cli-full-v1` screening run. Use
`score` when a run already has saved receipts and must be rescored offline.

## Prerequisites

- A read-only evaluation config from
  `docs/examples/candidate-evaluation-cli-full-v1.yaml`.
- Complete, clean candidate and incumbent stack manifests. They must identify
  weights, runtime, template/parser, decoding, context, Hermes revision/config,
  hardware, resolved `hermes-cli` tool schemas, and rollback readiness.
- A rollback artifact with a current route, tested recipe, and human owner.
- For implementation or CI, use a deterministic local fake boundary; do not
  call a live or paid provider.

## How to Run

Tier-0 smoke:

```bash
hermes providers validate --provider PROVIDER --model MODEL \
  --suite agent-readiness --toolsets file --out /tmp/hermes-tier0
```

Dry-run candidate evaluation (the default):

```bash
hermes providers evaluate \
  --candidate-manifest candidate-manifest.json \
  --incumbent-manifest incumbent-manifest.json \
  --evaluation-config evaluation.yaml \
  --out /tmp/hermes-candidate-run
```

An operator may explicitly use `--execute` for an approved local run. The
command always uses separate Hermes-home snapshots, the same fixture, real
SessionDB persistence, the declared `hermes-cli` schema, and no
`--ignore-rules`.

Offline score:

```bash
hermes providers score --run-dir /tmp/hermes-candidate-run
hermes providers suites list
```

## Quick Reference

| Surface | Meaning |
| --- | --- |
| `validate` | Six-case tier-0 compatibility smoke; never qualification. |
| `evaluate` | Frozen 27-case, three-repetition interleaved CLI comparison. |
| `score` | Receipt-only deterministic re-score; no provider client. |
| `suites list` | Lists the pinned suite/scorer identity. |

The evaluator reports only `GATE-FAILED`, `REJECT`, `HOLD`, or `SCREEN-PASS`.
`PROMOTE-CANDIDATE` is reserved for a later preregistered policy with at least
100 cases. Screening intervals are descriptive/non-confirmatory.

## Procedure

1. Freeze both manifests, the standalone evaluation config, suite digest, and
   seed. Do not use the active user config as an implicit evaluation spec.
2. Confirm that the full lane selects `hermes-cli`, not the tier-0 `file`
   toolset, and that both manifests contain the same suite/scorer/weights and
   local-only policy.
3. Run the dry-run and inspect missing prerequisites. It must not create a
   provider client or modify `config.yaml`, `.env`, sessions, routing, or
   credentials.
4. For an approved execution, let the incumbent A/A pilot complete first.
   The pilot requires 81 paired observations, exact receipt integrity, zero
   online/offline scorer disagreement, bounded false non-ties, a zero-including
   mean delta interval, and a bounded order effect.
5. Inspect `receipts.jsonl` as the source of truth. Each of the 27 cases has
   three repetitions per arm; repetitions aggregate within case before HFS.
6. Review hard gates, seven dimension means, HFS, paired deltas/intervals,
   win/loss/tie counts, and optional archive rank. Archive rank is informational
   and never overrides a gate or screening status.
7. Run offline `score` and compare `offline-summary.json` with `summary.json`.
   Editing a summary cannot make a tampered or incomplete receipt set eligible.

## Pitfalls

- The tier-0 `file` toolset is not the full lane and must not be described as
  production coverage.
- Do not add `--ignore-rules`: it removes context, memory, and skills that the
  lane is explicitly measuring.
- A timeout, missing session, malformed role sequence, unsafe side effect,
  fabricated artifact, or incomplete pair is evidence of a gate failure, not a
  missing observation to discard.
- A different hardware/runtime path is not a same-policy speed comparison.
- External network, browser, gateway/platform, delegation, auto-routing, and
  automatic promotion are outside PR-1.

## Verification

- Confirm the result directory contains `run.json`, normalized config, both
  manifests, `schedule.jsonl`, `receipts.jsonl`, pair results, summaries, raw
  artifacts, and `checksums.sha256`.
- Confirm all receipt hashes and raw-file hashes validate.
- Confirm every summary carries `lane_id`, `suite_version`, `scorer_version`,
  `weights_version`, A/A outcome, rollback readiness, and
  `promotion_applied: false`.
- State the exact lane and screening status. Never call it global Hermes
  qualification or authorize routing from this skill.

---
name: profile-collaboration
description: Route substantive work to relevant Ares profiles.
version: 1.2.0
author: Josh Stevenson / RecursiveIntell Ares
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [profiles, collaboration, evidence, publication, routing]
    category: productivity
    related_skills: [hermes-agent-skill-authoring, requesting-code-review]
---

# Profile-Separated Ares Collaboration

## Overview

This skill is the durable operating contract for collaborating with the named
Ares profiles. `ares_runtime.specialist_routing` owns deterministic need
routing; this skill owns the repeatable procedure; `scripts/run_panel.py` owns
bounded invocation and receipt capture; `scripts/verify_receipt.py` owns
mechanical artifact verification. Profile reports remain advisory until Ares
independently checks them against current source and runtime state.

Policy presence, a spawned process, a returned report, and a controller-verified
panel are different evidence states. Never collapse them into one claim.

## When to Use

Use only when a specialist lane is relevant to the current task and its expected
information gain or risk reduction exceeds the latency, cost, and session noise.
Do not consult profiles for routine mechanical work, facts available from current
source, or as ceremony. Select the smallest sufficient specialist set. Escalate
to the full panel only when the task is genuinely cross-domain, high-stakes,
release/publication-critical, or the operator explicitly requests it.

## Rule

Actual profile-separated Ares bots are preferred over generic delegated
subagents when specialist consultation is justified. Route only to named lanes
that can materially improve the result. Give selected profiles the same
self-contained brief, integrate their outputs, preserve dissent, and
independently verify decisive claims. A profile cannot replace the operator's
authority or widen its own scope. Omitted irrelevant profiles are not failed
lanes; record the selected set and routing reason in the receipt.

## Profile routing

- `public` — public claims, README/docs, release notes, commits/pushes,
  deployment statements, portfolio/application material, benchmarks, talks,
  posts, and external communication.
- `job-scout` — job/career research: finding current roles that fit the
  operator's verified skills, discovery leads, posting verification, and
  application-target shortlists (read-only; never applies on the operator's
  behalf).
- `explorer` — competing designs, novel combinations, cheap falsification, and
  kill criteria.
- `longmemeval-bench` — source-of-truth, evidence lineage, schemas, receipts,
  and reproducibility.
- `statistician` — estimands, uncertainty, quantitative comparison, and
  evidence strength.
- `ml-evaluation-researcher` — benchmark design, controls, rubrics, and
  model/evaluation claims.
- `cognitive-scientist` — coordination, dissent, reconciliation, authority,
  and recovery constructs.
- `psychometrician` — measurement validity, assessment governance, and
  discriminant validity.
- `inbox-manager` — correspondence, follow-up, commitments, and communication
  evidence.

## Deterministic need routing

Before invoking profiles, compile explicit uncovered evidence obligations. If
the installed `ares_runtime.specialist_routing` exposes the documented CLI, run
it with `terminal`:

```bash
python -m ares_runtime.specialist_routing --request /path/to/request.json
```

Some installed revisions provide this module as a library only (no CLI entry
point). In that case, record `ROUTER_CLI_UNAVAILABLE` and select from the
explicit obligation-to-profile map below; do not treat a silent zero-exit module
invocation as a routing decision. The request must name the uncovered question,
why direct deterministic evidence is insufficient, materiality, independence
need, expected information gain, cost, latency, and a specialist cap.
Obligation classes map to profiles:

- `public_claim` → `public`
- `competing_design` → `explorer`
- `source_lineage` → `longmemeval-bench`
- `quantitative_inference` → `statistician`
- `model_evaluation` → `ml-evaluation-researcher`
- `coordination_recovery` → `cognitive-scientist`
- `measurement_validity` → `psychometrician`
- `communication_commitment` → `inbox-manager`

The router preserves one-executor default behavior, suppresses ceremonial
fan-out when direct evidence is sufficient or expected value is non-positive,
and emits `selected`, `no_specialist_needed`, `degraded`, or `blocked`. Advisory
`SpecialistBidV1` artifacts are shadow-only and cannot grant authority or alter
selection. All eight profiles require `full_panel_explicit: true`.

## Operating procedure

1. Resolve the current source/runtime identity, run deterministic need routing,
   and write one self-contained
   brief containing the goal, constraints, evidence cutoff, allowed scope,
   and explicit read-only boundary when applicable.
2. Select the smallest sufficient set and use the reusable runner:
   ```bash
   python3 "$HOME/.ares/skills/productivity/profile-collaboration/scripts/run_panel.py" \
     --workspace /absolute/workspace \
     --profiles public,explorer \
     --brief 'The same complete brief for every selected profile.' \
     --max-workers 2
   ```
   Use `--full-panel` only as an explicit escalation. The runner refuses an
   implicit all-profile invocation, isolates each selected `HERMES_HOME`, uses
   bounded concurrency and process-group cleanup, requests automatic archival
   of automation-owned oneshot sessions, and writes a machine-readable receipt
   under `~/.ares/profile-collaboration/receipts/`.
3. Run the dry-run first for a new routing shape. It must enumerate exactly the
   explicitly selected profiles, in canonical order, with no duplicates. A
   full-panel dry-run must enumerate all eight. Do not put credentials, tokens,
   passwords, or connection strings in the brief or receipt.
4. Wait for every selected profile. A running child is not a completed
   consultation. A nonzero exit, timeout, empty report, controller error, or
   failed session archival is a failed or blocked selected lane, not implicit
   approval. Automation-owned transcripts remain durable but should be archived
   from the normal Desktop Sessions projection.
5. Verify the receipt without changing it:
   ```bash
   python3 "$HOME/.ares/skills/productivity/profile-collaboration/scripts/verify_receipt.py" \
     --receipt /absolute/path/to/receipt-directory
   ```
   Pass the **directory containing `panel.json`**, not the `panel.json` path itself;
   the verifier writes its separate `verification.json` projection inside that
   directory. This checks profile identity/order, runtime revision, artifact existence,
   byte counts, and SHA-256 digests. It deliberately leaves semantic review
   to the controller.
6. Read each returned report, extract evidence/uncertainty/dissent/next gate,
   preserve disagreements, and independently verify decisive claims against
   current files and live commands. Only then may the controller mark the
   panel closeout verified.
7. For anything public or externally visible, obtain the `public` review
   before the effect and again at closeout. Publication/release authority is a
   separate operator decision.

## Evidence states

Use these distinctions in receipts and responses:

- `policy_present` — SOUL/skill text exists and is loaded or observed on disk;
- `invoked` — a profile process was started;
- `returned` — the profile exited with a report;
- `execution_artifacts_verified` — the controller verified the receipt files,
  identities, and hashes;
- `controller_verified` — the controller reviewed the reports and current
  source/runtime evidence;
- `not-applicable`, `unavailable`, `failed`, or `blocked/unknown` — explicit
  lane outcomes that must remain visible.

A runner receipt can prove execution mechanics. It cannot by itself prove that
all reports are correct, that a desktop is healthy, or that a public claim is
safe.

## Support files

- [scripts/run_panel.py](scripts/run_panel.py) — bounded eight-profile
  invocation with per-profile stdout/stderr artifacts and `panel.json`.
- [scripts/verify_receipt.py](scripts/verify_receipt.py) — mechanical receipt
  and hash verification; writes a separate `verification.json` projection.

## Common pitfalls

- Treating a memory entry, SOUL paragraph, or skill file as proof that the
  panel actually ran.
- Calling eight profiles with unconstrained concurrency after an OOM or
  renderer incident. The default runner uses three workers; the desktop hard
  admission budget allows four local profile backends, but reserve the fourth
  for a user-selected/foreground profile or explicitly justified burst.
- Treating a generic approval or empty response as a domain result.
- Allowing one profile's credentials, configuration, workspace, or authority
  to leak into another profile.
- Replacing a failed panel with a generic delegated subagent without recording
  the unavailable-profile exception.
- Sending a profile a huge staged diff as a publication brief. First validate the
  profile with a bounded one-shot; then ask for a narrow claim-boundary review
  against the staged stat, selected high-risk paths, and named validation receipts.
  A panel timeout caused by unbounded review scope is not a profile-health failure.
- Mutating profile homes during a read-only consultation.
- A profile can exit zero yet be operationally blocked when every tool call fails before execution (observed runtime error: `DaemonThreadPoolExecutor object has no attribute _initializer`). Treat the lane as `failed`/`blocked`, not `returned` approval; preserve the receipt, report the runtime revision, and retry only after the executor/runtime owner is repaired or with a verified alternate runtime.
- Overwriting an old receipt or deleting failed evidence. Receipts are
  append-only run artifacts; new attempts get new directories.

## Verification checklist

- [ ] The current runtime and source identity are recorded.
- [ ] The brief is identical across all required profiles and contains no
      secrets.
- [ ] The specialist set is the smallest sufficient set and its routing reason is recorded.
- [ ] Dry-run profiles match the explicit selection in canonical order.
- [ ] Every selected profile has an explicit outcome and exit status.
- [ ] Receipt paths remain inside the run directory and their hashes verify.
- [ ] No orphaned panel process remains after completion or timeout.
- [ ] Automation-owned oneshot sessions are archived and absent from the normal Sessions projection.
- [ ] Reports are reviewed for evidence, uncertainty, dissent, and next gates.
- [ ] `public` is consulted before any external-facing effect.
- [ ] Desktop/runtime recovery is tested separately from collaboration.
- [ ] Remaining proof debt and rollback/quarantine are visible.

## Closeout

Report the receipt path, runtime revision, profiles consulted, each profile's
outcome, preserved dissent, controller verification commands, unresolved
blockers, and whether the public gate passed, failed, or was not applicable.
Do not call the feature complete merely because the runner returned zero: the
mechanical verification and semantic controller review must both pass.

---
name: bmad-investigate
description: Forensic case investigation with evidence-graded findings, calibrated to the input. Use when the user asks to investigate a bug, trace what caused an incident, walk through unfamiliar code, or build a mental model of a code area before working on it.
---

# Investigate

## Overview

Reconstruct what's happening, or what an unfamiliar area does, from the available evidence. Produce a structured case
file another engineer can pick up cold. Calibrate continuously between defect-chasing (symptom-driven) and
area-exploration (no symptom); the same discipline applies on both ends.

**Args:** A ticket ID, log file path, diagnostic archive, error message, code area name, problem description, or a path
to an existing case file. The last form resumes a prior investigation; everything else opens a new case.

**Output:** `{implementation_artifacts}/{workflow.case_file_subdir}/{workflow.case_file_filename}`. Reference inputs
are recorded; raw content is not read into the parent context until an outcome calls for it.

`{slug}` is the ticket ID when one is provided, otherwise a short descriptive name agreed with the user, sanitized to
lowercase alphanumeric with hyphens. On collision with an existing case file at the resolved path, ask whether to
rename to `slug-YYYY-MM-DD.md` or resume the existing file (resuming routes to Outcome 0).

After every outcome, present what was learned and pause for the user before continuing.

## Principles

- **Evidence grading.**
  - **Confirmed.** Directly observed; cite `path:line`, log timestamp, or commit hash.
  - **Deduced.** Logically follows from Confirmed evidence; show the chain.
  - **Hypothesized.** Plausible but unconfirmed; state what would confirm or refute it.
- **Stronghold first.** Anchor in one Confirmed piece of evidence and expand outward. Never start from a theory and
  hunt for support. When evidence is sparse, switch to evidence-light mode (Outcome 1 branch).
- **Challenge the premise.** The user's description is a hypothesis, not a fact. Verify independently; if evidence
  contradicts, say so.
- **Follow the evidence, not the narrative.** When evidence contradicts the working theory, update the theory — never
  the other way around. Resist confirmation bias even when the user is convinced.
- **Hypotheses are never deleted.** Update Status (Open / Confirmed / Refuted) and add a Resolution. Wrong turns are
  part of the deliverable.
- **Missing evidence is itself a finding.** Document the gap, what it would resolve, and how to obtain it.
- **Write it down early.** Initialize the case file as soon as the slug is agreed; it is the persistent state across
  interruptions.
- **Path:line citations** use CWD-relative format, no leading `/`, so they're clickable in IDE-embedded terminals.
- **Delegation discipline.** When a step requires reading 5+ files or any file >10K tokens, delegate to a subagent
  that returns structured JSON only. Cite `path:line` from the result; don't re-read in the parent.
- **Issue independent operations in parallel** (multi-grep, multi-read, parallel inventories) — one message, multiple
  tool calls.
- **Communication.** Evidence-first language ("the evidence shows", "unconfirmed, requires X to verify"). No hedging,
  no narrative.

## On Activation

### Step 1: Resolve the workflow block

Run: `python3 {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow`

If the script fails, stop and surface the error.

### Step 2: Execute prepend steps

Run each entry in `{workflow.activation_steps_prepend}` in order.

### Step 3: Load persistent facts

Treat each entry in `{workflow.persistent_facts}` as foundational context. `file:` prefixes are paths or globs under
`{project-root}` (load contents); other entries are facts verbatim.

### Step 4: Load config

Load `{project-root}/_bmad/bmm/config.yaml` and resolve `{user_name}`, `{communication_language}`,
`{document_output_language}`, `{implementation_artifacts}`, `{project_knowledge}`. If `{implementation_artifacts}` is
unresolved, fall back to `./investigations/` and surface the fallback before initializing.

### Step 5: Greet

Greet `{user_name}` in `{communication_language}`.

### Step 6: Execute append steps

Run each entry in `{workflow.activation_steps_append}` in order.

Activation is complete. If `activation_steps_prepend` or `activation_steps_append` were non-empty, confirm every entry was executed in order before proceeding. Do not begin the main workflow until all activation steps have been completed.

### Step 7: Acknowledge and route

Acknowledge the input as a reference (record paths and IDs; don't read raw content). Path to an existing case file →
Outcome 0. Otherwise → Outcome 1.

## Procedure

### Outcome 0: Existing case is loaded and surfaced

Read the case file. Surface, in order: open hypotheses (Status = Open) with their confirm/refute criteria; open
backlog (Status ≠ Done); missing-evidence rows; last Conclusion with confidence. Ask which thread to pull. New
evidence opens a new `## Follow-up: {YYYY-MM-DD}` block (append `#2`, `#3` on same-day reentry). Pause for user with the recap above; wait for direction.

### Outcome 1: Scope and stronghold are established

Acknowledge each input shape — record location, scope, time window only; bulk reads happen in Outcome 2.

- **Issue tracker ticket.** Fetch full details via available MCP tools.
- **Diagnostic archive.** Record path, file count, time window.
- **Log file or stack trace.** Record path and time window; only the stack frame already in the user's message is in
  scope here.
- **Free-text description.** Capture verbatim; treat as hypothesis.
- **Code area name** (no symptom). Record entry point.
- **Recent commit area.** Record commit range.

If the user arrived with a hypothesis, register it as Hypothesis #1. Find the stronghold *independently*; the user's
hypothesis is one of the things the stronghold validates or refutes.

Find a stronghold: a Confirmed piece of evidence (error message, function name, HTTP route, config parameter, test
case). Anchor here.

**Initialize `{case_file}` before branching.** The path is
`{implementation_artifacts}/{workflow.case_file_subdir}/{workflow.case_file_filename}` with `{slug}` substituted (slug
and collision rules in Overview). Create the file from `{workflow.case_file_template}` and fill Hand-off Brief
(rough), Case Info, Problem Statement, initial Evidence Inventory.

**Evidence-light branch.** When no Confirmed evidence is reachable: mark the case evidence-light in the Hand-off
Brief; populate the Investigation Backlog with prioritized data-collection items; record "to make progress, I need one
of: …"; pause for the user to provide evidence or authorize Outcome 2 to scan more broadly.

Otherwise present scope, stronghold, file path, proposed approach. Pause for user with the recap above; wait for direction.

### Outcome 2: Evidence perimeter is mapped

Survey the scene: inventory available evidence in parallel across these independent categories: diagnostic archives;
issue tracker; version control; test results; static analysis; source code. For any category exceeding ~10K tokens,
delegate to a subagent that returns a JSON manifest (paths, sizes, time windows, key fragments cited as `path:line`).

Classify each Available, Partial, or Missing — Missing is itself a finding. Update Evidence Inventory and Investigation
Backlog. Pause for user with the recap above; wait for direction.

### Outcome 3: Cause is reasoned about with discipline

- **Trace causality.** Symptom-driven: trace backward from the symptom to producing conditions and the state that
  emerged. Exploration: trace backward from outputs (returns, side effects, messages sent) to producing conditions.
  Same technique, different anchor.
- **Reconstruct the timeline** by cross-referencing logs, system events, version control, user observations.
- **Form and test hypotheses.** State, identify confirming/refuting evidence, search, grade
  (Confirmed / Refuted / Open). Update Status. Never delete.
- **Refutation pass.** Each time a hypothesis transitions toward Confirmed, actively look for refuting evidence first.
  Record the attempt in Resolution.
- **Verify the user's premise.** If evidence contradicts, say so explicitly.
- **Add discovered paths to the backlog.** Stay focused on the current thread.

Update Confirmed Findings, Deduced Conclusions, Hypothesized Paths, Backlog, Timeline. Highlight contradictions to the
original premise. Pause for user with the recap above; wait for direction.

### Outcome 4: Source has been traced where it matters

Issue these first-pass scans as parallel tool calls in one message: grep for exact error strings; glob the affected
directory for parallel implementations; `git log` for recent changes.

Then sequentially: read the surrounding code; follow the caller chain; watch for language and process boundary
crossings (compiled→scripts, IPC, host→device, configuration flow).

Lean by case type:

- **Exploration:** I/O mapping (triggers, outputs, dependencies); frequent-terms scan; control-flow filtering
  (branches, loops, error handling, state-machine transitions).
- **Symptom-driven:** depth assessment — is the root cause reachable from local context, or is a broader area model
  required? Surface escalations; never silently expand scope. Trivial-fix assessment — off-by-one, missing null check,
  swapped argument → one-line code suggestion or draft diff in the report; non-trivial → stop at the root cause area.

Investigation stops at the diagnosis; implementation is out of scope. Update Source Code Trace (Error origin, Trigger,
Condition, Related files; area model when broader). Pause for user with the recap above; wait for direction.

### Outcome 5: Report is finalized and the hand-off is clean

Update `{case_file}`:

- **Hand-off Brief** rewritten to final form (3 sentences, 15-second read).
- **Final Conclusion** with confidence: **High** (Confirmed root cause, deterministic repro), **Medium** (Deduced;
  minor uncertainty), **Low** (Hypothesized; clear data gap).
- **Fix direction** when applicable (categorize by mechanism if multiple combine).
- **Diagnostic steps** if uncertainty remains.
- **Reproduction Plan** when applicable, or a verification plan for exploration cases.
- **Status:** Active / Concluded / Blocked on evidence.

Present the conclusion, then a concrete next-steps menu: trivial fix → `bmad-quick-dev`; scope/plan adjustment →
`bmad-correct-course`; tracked story → `bmad-create-story`; fresh review → `bmad-code-review`. Recommend the
highest-value action. Mitigations and workarounds are generated only on explicit request — investigation stops at the
diagnosis. Execute `{workflow.on_complete}` if non-empty. Pause for user with the recap above; wait for direction.

## Follow-up Iterations

Continue work by appending to `{case_file}` under a new `## Follow-up: {YYYY-MM-DD}` block (`#2`, `#3` on same-day
reentry). The investigation is complete when:

- Root cause is Confirmed.
- Root cause is Hypothesized with a clear data gap.
- The mental model is sufficient for the user's stated goal (exploration cases).
- The backlog contains only items requiring unavailable evidence.
- The user explicitly concludes.

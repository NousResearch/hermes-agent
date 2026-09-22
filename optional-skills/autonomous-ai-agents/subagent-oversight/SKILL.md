---
name: subagent-oversight
description: Use when dispatching or supervising delegated subagents.
version: 1.0.0
author: Hermes Agent contributors
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [orchestration, delegation, monitoring, verification, safety]
---

# Subagent Oversight

Supervise delegated work as a lifecycle: **authorize → dispatch → monitor → steer or stop → verify → hand off**. Delegation is not complete when a child reports success; the parent must verify the resulting state before relying on it or reporting completion.

This skill is capability-aware. Use the orchestration and inspection tools exposed by the current Hermes surface. Do not assume that every surface provides live child control, transcript access, process control, or durable scheduling.

## Operating principles

- Keep the parent accountable for the final result. A child’s status or self-report is evidence, not verification.
- Give each child a bounded, self-contained assignment with a clear completion condition.
- Preserve authorization boundaries. A child may use only the tools, credentials, workspaces, and approval scope granted by the parent and current runtime.
- Prefer reversible intervention. Steer first when a child is merely unclear or slow; stop when it is unsafe, duplicative, blocked, or pursuing the wrong objective.
- Make timing configurable. Use the runtime’s configured polling interval, inactivity window, maximum run duration, and retry policy; never encode a universal elapsed-time rule in the procedure.
- Treat missing control or observation capabilities as a limitation, not as proof that work is progressing or stopped.

## Authorization before dispatch

Before creating a child, establish:

1. The user-approved objective and any actions that require confirmation.
2. The allowed workspace, files, tools, network access, credentials, and side effects.
3. The parent’s verification bar and the conditions that require escalation, pause, or stop.
4. Whether the child may delegate further, and any configured concurrency or depth limits.

If the runtime cannot expose or confirm the required authorization state, do not broaden access implicitly. Narrow the assignment, request confirmation through the available approval mechanism, or decline the delegation.

## Dispatch

Use the native delegation capability when available, such as `delegate_task` or an equivalent Hermes tool.

Record the returned child handle or identifier and any result, status, log, transcript, or artifact references. Give the child:

- A precise goal and definition of done.
- Relevant source-of-truth files and exact workspace locations.
- Required inputs, fixed decisions, constraints, and a do-not-change list.
- Expected outputs and where they must be written.
- Verification commands or checks the child must run.
- A reporting format that distinguishes completed work, blocked work, assumptions, and unverified claims.

For independent tasks, dispatch in a batch only within the runtime’s configured concurrency and authorization limits. For dependent tasks, wait for upstream verification before dispatching downstream work.

## Monitor

Use every observation channel the current surface actually provides:

- Native child status or result inspection, if exposed.
- Child logs, transcripts, tool events, or progress messages, if accessible.
- Direct inspection of expected artifacts and their metadata.
- Runtime, process, or scheduler status, when the parent has permission to inspect it.

Evaluate progress from evidence rather than elapsed time alone. Look for meaningful tool activity, changed artifacts, resolved blockers, and movement toward the completion condition. Distinguish normal silence during a long operation from an inactivity window that the configured policy defines as requiring attention.

Do not infer that a child has stopped merely because its status is unavailable. Mark the state as unknown, use an available fallback observation, and report the uncertainty.

## Steer

If the child is stalled, ambiguous, drifting, or missing a required check, send a targeted steering message through the native control action when available. Steering should be delivered at a safe runtime boundary; do not claim that a currently running operation was interrupted unless the runtime confirms interruption.

A useful steer contains:

- The specific correction or next action.
- The artifact, evidence, or decision that resolves the ambiguity.
- A concise list of work that is already correct and must not regress.
- The revised verification requirement.
- An instruction to pause and report if the required input or authorization is unavailable.

If steering is unavailable, use a supported parent-side channel such as a follow-up message, shared control file, or supervisor mechanism. If no such channel exists, do not silently edit the child’s deliverable or assume the message was received; record the limitation and verify independently.

## Stop

Stop a child through the native stop/cancel capability when it is:

- Acting outside its authorization or workspace.
- Repeating work without progress.
- Producing unsafe, destructive, or materially incorrect changes.
- Blocked on unavailable input and unable to make safe progress.
- No longer needed because the objective or upstream state changed.

A stop request may be queued or may leave partial output. Confirm the runtime’s reported state, preserve useful partial work, and inspect the workspace for unintended changes. If native stopping is unavailable, use the least destructive supported supervisor or process-control mechanism; otherwise mark the child as uncontrolled and prevent downstream reliance on its output.

## Verify

Verify before downstream dispatch, merge, publication, or a completion report.

1. Confirm the expected artifact exists in the allowed workspace and inspect its size, timestamps, and type where relevant.
2. Read the artifact itself; never rely only on a child’s summary.
3. Check structure and substance: required headings or fields, key content, references, generated assets, and cross-file consistency.
4. Run the project-appropriate tests, linters, parsers, build checks, or other acceptance checks available in the environment.
5. Compare claims against the source of truth and distinguish verified facts from assumptions.
6. Inspect for collateral changes, secrets, unauthorized files, and scope creep.

When the parent writes or rewrites a deliverable, verify that write too. Preserve another child’s artifact rather than appending parent decisions directly to it. Put parent rulings, review notes, or handoff instructions in a separate control document unless a deliberate read-modify-write is required.

For generated Markdown, HTML, or similar artifacts, validate the parsed structure as well as raw text. Presence of a file or a passing superficial check does not prove that headings, sections, links, citations, or assets survived generation.

## Handoff and completion

Hand off only verified artifacts. Include:

- The exact artifact location and child identifier, if applicable.
- Checks that passed and checks that were unavailable or failed.
- Open risks, assumptions, partial work, and authorization decisions.
- Any follow-up action needed by the next child or the user.

If verification finds a defect, pause dependent work, correct or steer the responsible child, and repeat the relevant checks. Never report completion based solely on a successful dispatch, a green child status, or an uninspected file.

## Capability-neutral fallback

When a Hermes surface lacks a capability, degrade explicitly:

| Missing capability | Safe fallback |
|---|---|
| Live child listing or status | Track the returned handle and inspect available results, logs, processes, and artifacts. Mark status unknown when evidence is insufficient. |
| Steering | Use a supported follow-up or shared control channel; otherwise do not claim the child received direction. |
| Stopping | Use supported cancellation or process supervision; if neither exists, isolate the output and block downstream use until independently verified. |
| Transcript or tool-event access | Inspect artifacts and runtime state; lower confidence and use the configured observation policy. |
| Durable background execution | Keep the parent active, use an available scheduler/supervisor, or tell the user the work is process-bound. |
| Automated authorization prompts | Require explicit parent/user confirmation before sensitive actions; do not infer approval from tool availability. |
| Automated tests or parsers | Perform the strongest available structural and manual checks and report what could not be executed. |

The fallback is part of the result: state what was observed, what was not observable, and what remains unverified.

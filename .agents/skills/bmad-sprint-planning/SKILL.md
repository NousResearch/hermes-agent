---
name: bmad-sprint-planning
description: 'Check that planning is complete enough to implement, then generate the sprint status file from the epics. Can also summarize sprint progress and validate or repair the tracking file. Use when the user says "run sprint planning", "generate sprint plan", "check implementation readiness", "show sprint status", "validate sprint status", or "fix sprint status"'
---

# Overview

You are a senior developer about to commit to this plan. Two moves, in order: first scrutinize the planning the way a skeptic reads a handoff — gaps found now are cheap, gaps found mid-build are not. Then hand the mechanical work to the script: parsing epics, deriving keys, merging statuses, and writing `sprint-status.yaml` are deterministic jobs, not judgment calls. Your judgment goes where the script can't: deciding which files are epics, weighing readiness, and reconciling anything the script flags.

## On Activation

1. Resolve customization: `uv run {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --project-root {project-root} --key workflow`. On failure, read `{skill-root}/customize.toml` directly and use defaults.
2. Execute each entry in `{workflow.activation_steps_prepend}` in order.
3. Treat every entry in `{workflow.persistent_facts}` as foundational context for the rest of the run. Entries prefixed `file:` are paths or globs under `{project-root}` — load the referenced contents as facts. All other entries are facts verbatim.
4. Run `uv run {project-root}/_bmad/scripts/resolve_config.py --project-root {project-root}` and resolve `{user_name}`, `{communication_language}`, `{document_output_language}`, `{project_name}`, `{planning_artifacts}`, `{implementation_artifacts}`, `{project_knowledge}` (skip gracefully if unset), and `{date}`. Stay in `{communication_language}` for every turn, not just the greeting.
5. Greet `{user_name}`, detect intent, and load only what that intent needs:
   - **readiness** — check implementation readiness only: load `references/readiness-gate.md`, run the gate, report, stop
   - **sprint-planning** — the full flow (also the refresh path for an existing `sprint-status.yaml`): load `references/readiness-gate.md`, then on PASS `references/generate-tracking.md`
   - **status** — "show sprint status", "where are we": skip the gate, load `references/status-view.md`
   - **validate** — check the tracking file's format: load `references/validate.md`
   - **fix** — repair or rebuild a broken `sprint-status.yaml`: load `references/fix-sprint-status.md`

   If interactive and unclear, ask; for headless behavior see `## Headless Mode`.

Execute each entry in `{workflow.activation_steps_append}` in order.

Activation is complete. If `activation_steps_prepend` or `activation_steps_append` were non-empty, confirm every entry was executed in order before proceeding.

## If the Script Fails

This rule covers every intent: when `sprint_plan.py` errors or the file is in a state it cannot handle, do not stop at the error and do not guess silently. Read the files yourself, deliver the same outcome by best judgment, tell the user the deterministic path failed and why, and offer the fix flow (`references/fix-sprint-status.md`) to restore a file the script can work with.

## On Completion

Whatever the intent, close out in `{communication_language}` per the loaded reference, then run `{workflow.on_complete}` if non-empty; treat a string scalar as one instruction and an array as a sequence.

## Headless Mode

When invoked headless, do not ask. Run the gate and, unless intent was readiness-only, generate tracking. Ambiguity the interactive flow would resolve by asking (duplicate epic versions, unreconciled orphans, an unconfirmed fix) halts with a `blocked` status instead of guessing. End with a JSON response:

```json
{
  "status": "complete",
  "intent": "sprint-planning",
  "gate": "PASS",
  "status_file": "{implementation_artifacts}/sprint-status.yaml",
  "findings": [],
  "warnings": []
}
```

`gate` is `PASS`, `CONCERNS`, or `FAIL`; on `FAIL` include `findings` and the saved findings path if written, and omit `status_file`. `intent` is `"readiness"`, `"sprint-planning"`, `"status"`, `"validate"`, or `"fix"` — for status and validate intents, omit `gate` and pass the script's JSON through under a `report` key (not `status`, which names the run state).

## References

- `scripts/sprint_plan.py` — the deterministic parser/generator/merger; subcommands `generate`, `status`, `validate`. Its JSON output is the contract this skill reads; argparse errors are JSON too
- `references/readiness-gate.md` — the PASS/CONCERNS/FAIL gate: artifact inventory and the implementability question
- `references/generate-tracking.md` — epic discovery, the generate command, and acting on its JSON report
- `references/status-view.md` — the status view: counts, risks, open action items, next recommended action
- `references/fix-sprint-status.md` — rebuild a broken tracking file: evidence-gathering subagents, user confirmation, pristine regeneration
- `references/validate.md` — format validation of an existing `sprint-status.yaml`
- `sprint-status-template.yaml` — the documented file format and status vocabulary; the script embeds the same block and the test suite pins the two copies together

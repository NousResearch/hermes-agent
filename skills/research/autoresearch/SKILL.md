---
name: autoresearch
description: Run bounded AutoResearch cycles on a workspace-defined project. Inspect manifests, validate before running, respect editable bounds, and only publish interesting runs.
version: 1.0.0
author: Hermes Agent + Teknium
tags: [autoresearch, research, experimentation, evaluation]
---

# AutoResearch

Use this skill when a workspace defines AutoResearch manifests under `.hermes/autoresearch/` and the user wants Hermes to explore, evaluate, and summarize bounded research candidates.

## Workflow

1. Start with `autoresearch(action="list_projects")` or `autoresearch(action="inspect_project")`.
2. Run `autoresearch(action="validate_project")` before any research cycle.
3. Use `autoresearch(action="research_cycle", family_id="...")` to execute one bounded family.
4. Inspect results with `status`, `list_runs`, or `inspect_run`.
5. Only use `publish_summary` when the run is interesting or the user explicitly wants a summary anyway.

## Guardrails

- Respect the project's declared editable files and editable marker ranges.
- Do not mutate files outside the bounded mutable surface.
- Prefer the built-in `research_cycle` flow over ad hoc manual mutation.
- Treat evaluator commands and result JSON as the source of truth for selection.
- Reports are per interesting run, not per candidate.
- Use Hermes messaging tools for delivery. Do not invent a parallel Telegram workflow.

## Project Contract

AutoResearch workspaces use:
- `.hermes/autoresearch/project.yaml`
- `.hermes/autoresearch/families/*.yaml`
- `.hermes/autoresearch/runs/<run_id>/`
- `.hermes/autoresearch/workspaces/<run_id>/<candidate_id>/`
- `research/YYYY-MM-DD/<project-id>--<run-id>.md`

## Good Habits

- Summarize the project thesis and family thesis before running.
- Mention the primary metric and selector constraints in plain language.
- When a run fails validation or evaluation, explain that failure instead of guessing.
- If a report exists, cite the report path in your response.
- If the user wants recurring summaries, combine AutoResearch with `cronjob` and `send_message` rather than reimplementing scheduling.

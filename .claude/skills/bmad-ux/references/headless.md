# Headless Mode

Load this file when invoked headless. Follow it for the whole run.

## Detection

Headless when any of: caller sets `headless: true` (or harness equivalent); invocation is from another skill or non-interactive runner; `{workflow.activation_steps_prepend}` declares it; first message is an automation context pre-supplying inputs. Ambiguous → default interactive.

## Inputs

Free-form structured payload in the first message:

- `intent` — `"create"`, `"update"`, or `"validate"`. If absent, infer from the artifact set.
- **Create**: any source spec (PRD, brief, requirements list, design-thinking output, prior UX — text, path, or URL) plus brand / platform / accessibility notes; `doc_workspace` if a specific run folder is required.
- **Update**: existing workspace containing `DESIGN.md` + `EXPERIENCE.md` (or path to either) + change signal.
- **Validate**: existing workspace containing `DESIGN.md` + `EXPERIENCE.md` (or path to either). Workspace defaults to the spines' containing directory.

Inferences → `assumptions[]`. Gaps needing a human decision → `open_questions[]`. Do not invent persona, brand, accessibility, or scope detail.

Creative tools default off in headless. Caller can override; artifacts land in `.working/` and are not promoted unless the caller signals.

## Behavior

Do not ask. Do not greet. Complete the intent from what's provided, what exists in `{doc_workspace}`, or what you can discover. If intent stays ambiguous after inference, halt with `status: "blocked"` and a one-sentence `reason`.

`status`:
- `"complete"` — stands on its own.
- `"partial"` — artifact produced but `open_questions[]` non-empty or critical inputs inferred.
- `"blocked"` — no artifact produced.

End with JSON matching `assets/headless-schemas.md`. `intent` reflects detected intent. Omit keys for artifacts not produced.

## Mode-specific overrides

**Update.** Apply the change. Log it via `uv run {project-root}/_bmad/scripts/memlog.py append --workspace {doc_workspace} --type change --text "<change + rationale>"`. Surface conflicts in `conflicts_with_prior_decisions[]`.

**Validate.** Always write both `validation-report.html` and `validation-report.md` regardless of finding count. Always include `"offer_to_update": true`. Skip the browser-open step.

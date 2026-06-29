# Headless Mode

Load this file when bmad-prd is invoked headless (no interactive user). Follow it for the whole run.

## Detection

Headless mode is in effect when any of the following is true:

- the invoking caller sets a `headless: true` flag (or equivalent argument the harness exposes),
- the invocation is from another skill or a non-interactive runner (no TTY, no user message stream),
- `{workflow.activation_steps_prepend}` includes an entry that explicitly declares headless,
- the first message comes from an automation context that pre-supplies all inputs and asks for an artifact path back.

When ambiguous, default to interactive.

## Inputs the caller is expected to provide

The caller passes inputs in their first message (free-form structured payload; no fixed schema, but every field below should be present when applicable):

- `intent` — `"create"`, `"update"`, or `"validate"`. If absent, infer from the artifact set.
- For **Create**: a brief or product spec the LLM works from (plain text, file path, or URL), plus any user/scope notes; `doc_workspace` if a specific run folder is required (otherwise the workflow binds the default).
- For **Update**: the existing `prd.md` path (or a workspace path that contains one), and a change signal (the request: what to change and why).
- For **Validate**: the existing `prd.md` path (or workspace path), and optionally a checklist override path. Workspace defaults to the PRD's containing directory.

Anything the caller does not provide is either inferred from inputs/workspace or recorded as `assumptions[]` / `open_questions[]` in the JSON status. Do not invent user detail, success metrics, or scope decisions to fill gaps — record them.

## General

Do not ask. Complete the intent using what is provided, what exists in `{doc_workspace}`, or what you can discover yourself. If intent remains ambiguous after inference, halt with `status: "blocked"` and a `reason` field — do not prompt. Do not greet.

Populate `assumptions[]` with every value you inferred without direct caller confirmation; populate `open_questions[]` with every gap that needs a human decision. Use `status: "partial"` when the artifact was produced but `open_questions[]` is non-empty or critical inputs were inferred (Create with no brief; Update with a vague signal acted on best-effort; Validate that could not load the checklist). `complete` = stands on its own; `partial` = caller should review before downstream use; `blocked` = no artifact produced.

End with the JSON response (full schemas with examples in `assets/headless-schemas.md`). The `intent` field must match the detected intent. Omit keys for artifacts not produced.

## Mode-specific overrides

**Update.** Apply the change, log it via `uv run {project-root}/_bmad/scripts/memlog.py append --workspace {doc_workspace} --type change --text "<change + rationale>"`, and surface any conflict-with-prior-decision in `conflicts_with_prior_decisions[]` in the JSON status. Halt `blocked` if intent is ambiguous.

**Validate.** Always write both `validation-report.html` and `validation-report.md` to `{doc_workspace}` regardless of finding count. Always include `"offer_to_update": true` in the JSON status. Skip the browser-open step in `references/validate.md` — write the artifacts and return.

---
deferred_work_file: '{implementation_artifacts}/deferred-work.md'
---

# Step One-Shot: Implement, Review, Present

## RULES

- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`
- NEVER auto-push.

## INSTRUCTIONS

### Implement

Follow `./sync-sprint-status.md` with `{target_status}` = `in-progress`.

Implement the clarified intent directly.

### Review

Invoke the `bmad-review-adversarial-general` skill in a subagent with the changed files. The subagent gets NO conversation context — to avoid anchoring bias. Launch at the same model capability as the current session. If no sub-agents are available, write the changed files to a review prompt file in `{implementation_artifacts}` and HALT. Ask the human to run the review in a separate session and paste back the findings.

### Classify

Deduplicate all review findings. Three categories only:

- **patch** — trivially fixable. Auto-fix immediately.
- **defer** — pre-existing issue not caused by this change. Append to `{deferred_work_file}`.
- **reject** — noise. Drop silently.

If a finding is caused by this change but too significant for a trivial patch, HALT and present it to the human for decision before proceeding.

### Generate Spec Trace

Set `{title}` = a concise title derived from the clarified intent.

Write `{spec_file}` using `./spec-template.md`. Fill only these sections — delete all others:

1. **Frontmatter** — set `title: '{title}'`, `type`, `created`, `status: 'done'`. Add `route: 'one-shot'`.
2. **Title and Intent** — `# {title}` heading and `## Intent` with **Problem** and **Approach** lines. Reuse the summary you already generated for the terminal.
3. **Suggested Review Order** — append after Intent. Build using the same convention as `./step-05-present.md` § "Generate Suggested Review Order" (spec-file-relative links, concern-based ordering, ultra-concise framing).

Follow `./sync-sprint-status.md` with `{target_status}` = `review`.

### Commit

If version control is available and the tree is dirty, create a local commit with a conventional message derived from the intent. If VCS is unavailable, skip.

### Present

1. Open the spec in the user's editor so they can click through the Suggested Review Order:
   - Resolve two absolute paths: (1) the repository root (`git rev-parse --show-toplevel` — returns the worktree root when in a worktree, project root otherwise; if this fails, fall back to the current working directory), (2) `{spec_file}`. Run `code -r "{absolute-root}" "{absolute-spec-file}"` — the root first so VS Code opens in the right context, then the spec file. Always double-quote paths to handle spaces and special characters.
   - If `code` is not available (command fails), skip gracefully and tell the user the spec file path instead.
2. Display a summary in conversation output, including:
   - The commit hash (if one was created).
   - List of files changed with one-line descriptions. Any file paths shown in conversation/terminal output must use CWD-relative format (no leading `/`) with `:line` notation (e.g., `src/path/file.ts:42`) for terminal clickability — this differs from spec-file links which use spec-file-relative paths.
   - Review findings breakdown: patches applied, items deferred, items rejected. If all findings were rejected, say so.
   - A note that the spec is open in their editor (or the file path if it couldn't be opened). Mention that `{spec_file}` now contains a Suggested Review Order.
   - **Navigation tip:** "Ctrl+click (Cmd+click on macOS) the links in the Suggested Review Order to jump to each stop."
3. Offer to push and/or create a pull request.

HALT and wait for human input.

Workflow complete.

## On Complete

Run: `python3 {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow.on_complete`

If the resolved `workflow.on_complete` is non-empty, follow it as the final terminal instruction before exiting.

---
---

# Step 5: Present

## RULES

- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`
- NEVER auto-push.

## INSTRUCTIONS

### Generate Suggested Review Order

Read `{baseline_commit}` from `{spec_file}` frontmatter and construct the diff of all changes since that commit.

Append the review order as a `## Suggested Review Order` section to `{spec_file}` **after the last existing section**. Do not modify the Code Map.

Build the trail as an ordered sequence of **stops** — clickable `path:line` references with brief framing — optimized for a human reviewer reading top-down to understand the change:

1. **Order by concern, not by file.** Group stops by the conceptual concern they address (e.g., "validation logic", "schema change", "UI binding"). A single file may appear under multiple concerns.
2. **Lead with the entry point** — the single highest-leverage file:line a reviewer should look at first to grasp the design intent.
3. **Inside each concern**, order stops from most important / architecturally interesting to supporting. Lightly bias toward higher-risk or boundary-crossing stops.
4. **End with peripherals** — tests, config, types, and other supporting changes come last.
5. **Every code reference is a clickable spec-file-relative link.** Compute each link target as a relative path from `{spec_file}`'s directory to the changed file. Format each stop as a markdown link: `[short-name:line](../../path/to/file.ts#L42)`. Use a `#L` line anchor. Use the file's basename (or shortest unambiguous suffix) plus line number as the link text. The relative path must be dynamically derived — never hardcode the depth.
6. **Each stop gets one ultra-concise line of framing** (≤15 words) — why this approach was chosen here and what it achieves in the context of the change. No paragraphs.

Format each stop as framing first, link on the next indented line:

```markdown
## Suggested Review Order

**{Concern name}**

- {one-line framing}
  [`file.ts:42`](../../src/path/to/file.ts#L42)

- {one-line framing}
  [`other.ts:17`](../../src/path/to/other.ts#L17)

**{Next concern}**

- {one-line framing}
  [`file.ts:88`](../../src/path/to/file.ts#L88)
```

> The `../../` prefix above is illustrative — compute the actual relative path from `{spec_file}`'s directory to each target file.

When there is only one concern, omit the bold label — just list the stops directly.

### Mark Spec Done

Change `{spec_file}` status to `done` in the frontmatter.

Follow `./sync-sprint-status.md` with `{target_status}` = `review`.

### Commit and Open

1. If version control is available and the tree is dirty, create a local commit with a conventional message derived from the spec title.
2. Open the spec in the user's editor so they can click through the Suggested Review Order:
   - Resolve two absolute paths: (1) the repository root (`git rev-parse --show-toplevel` — returns the worktree root when in a worktree, project root otherwise; if this fails, fall back to the current working directory), (2) `{spec_file}`. Run `code -r "{absolute-root}" "{absolute-spec-file}"` — the root first so VS Code opens in the right context, then the spec file. Always double-quote paths to handle spaces and special characters.
   - If `code` is not available (command fails), skip gracefully and tell the user the spec file path instead.

### Display Summary

Display summary of your work to the user, including the commit hash if one was created. Any file paths shown in conversation/terminal output must use CWD-relative format (no leading `/`) with `:line` notation (e.g., `src/path/file.ts:42`) for terminal clickability — the goal is to make paths clickable in terminal emulators. Include:

- A note that the spec is open in their editor (or the file path if it couldn't be opened). Mention that `{spec_file}` now contains a Suggested Review Order.
- **Navigation tip:** "Ctrl+click (Cmd+click on macOS) the links in the Suggested Review Order to jump to each stop."
- Offer to push and/or create a pull request.

Workflow complete.

## On Complete

Run: `python3 {project-root}/_bmad/scripts/resolve_customization.py --skill {skill-root} --key workflow.on_complete`

If the resolved `workflow.on_complete` is non-empty, follow it as the final terminal instruction before exiting.

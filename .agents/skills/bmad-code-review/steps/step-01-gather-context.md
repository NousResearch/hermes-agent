---
diff_file: '' # set at runtime: path to the diff file
claims_file: '' # set at runtime (path or empty)
spec_file: '' # set at runtime (path or empty)
review_mode: '' # set at runtime: full or no-spec
story_key: '' # set at runtime when discovered from sprint status
---

# Step 1: Gather Context

## RULES

- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`
- The prompt that triggered this workflow IS the intent — not a hint.
- Writing `{diff_file}` and the claims file is the only change this step may make. Otherwise it is read-only.

## INSTRUCTIONS

1. **Find the review target.** The conversation context before this skill was triggered IS your starting point — not a blank slate. Check in this order — stop as soon as the review target is identified:

   **Tier 1 — Explicit argument.**
   Did the user pass a PR, commit SHA, branch, spec file, or diff source this message?
   - PR reference → resolve to branch/commit via `gh pr view`. If resolution fails, ask for a SHA or branch.
   - Commit or branch → use directly.
   - Spec file → set `spec_file` to the provided path. Check its frontmatter for `baseline_commit`. If found, use as diff baseline. If not found, continue the cascade (a spec alone does not identify a diff source).
   - Also scan the argument for diff-mode keywords that narrow the scope:
     - "staged" / "staged changes" → Staged changes only
     - "uncommitted" / "working tree" / "all changes" → Uncommitted changes (staged + unstaged)
     - "branch diff" / "vs main" / "against main" / "compared to <branch>" → Branch diff (extract base branch if mentioned)
     - "commit range" / "last N commits" / "<from-sha>..<to-sha>" → Specific commit range
     - "this diff" / "provided diff" / "paste" → User-provided diff (do not match bare "diff" — it appears in other modes)
   - When multiple keywords match, prefer the most specific (e.g., "branch diff" over bare "diff").

   **Tier 2 — Recent conversation.**
   Do the last few messages reveal what the user wants to be reviewed? Look for spec paths, commit refs, branches, PRs, or descriptions of a change. Apply the same diff-mode keyword scan and routing as Tier 1.

   **Tier 3 — Sprint tracking.**
   Look for a sprint status file (`*sprint-status*`) in `{implementation_artifacts}` or `{planning_artifacts}`. If found, scan for stories with status `review`:
   - **Exactly one `review` story:** Set `story_key` to the story's key (e.g., `1-2-user-auth`). HALT and give the user a choice:
     - **Review this story** — review the detected story `<story-id>` (status `review`).
     - **Choose another target** — pick a different review target.
     If the user chooses **Review this story**, use the story context to determine the diff source (branch name derived from story slug, or uncommitted changes). If they choose **Choose another target**, clear `story_key` and fall through.
   - **Multiple `review` stories:** Present them as numbered options alongside a manual choice option. Wait for user selection. If a story is selected, set `story_key` and use its context to determine the diff source. If manual choice is selected, clear `story_key` and fall through.
   - **None:** Fall through.

   **Tier 4 — Current git state.**
   If version control is unavailable, skip to Tier 5. Otherwise, check the current branch and HEAD. If the branch is not `main` (or the default branch), confirm: "I see HEAD is `<short-sha>` on `<branch>` — do you want to review this branch's changes?" If confirmed, treat as a branch diff against `main`. If declined, fall through.

   **Tier 5 — Ask.**
   Fall through to instruction 2.

   Never ask extra questions beyond what the cascade prescribes. If a tier above already identified the target, skip the remaining tiers and proceed to instruction 3 (construct diff).

2. HALT. Ask the user: **What do you want to review?** Present these options:
   - **Uncommitted changes** (staged + unstaged)
   - **Staged changes only**
   - **Branch diff** vs a base branch (ask which base branch)
   - **Specific commit range** (ask for the range)
   - **Provided diff or file list** (user pastes or provides a path)

3. Write the diff for the chosen source to `{diff_file}` — a uniquely-named file in the system temp directory, so concurrent reviews cannot collide. The review layers read that file; the diff text is never pasted into their prompts.
   - For **staged changes only**: run `git diff --cached > {diff_file}`.
   - For **uncommitted changes** (staged + unstaged): run `git diff HEAD > {diff_file}`.
   - For **branch diff**: verify the base branch exists, then run `git diff <base-branch>...HEAD > {diff_file}`. If it does not exist, HALT and ask the user for a valid branch.
   - For **commit range**: verify the range resolves, then run `git diff <range> > {diff_file}`. If it does not resolve, HALT and ask the user for a valid range.
   - For **provided diff**: validate the content is non-empty and parseable as a unified diff. If it is not parseable, HALT and ask the user to provide a valid diff. Write the validated diff to `{diff_file}`.
   - For **file list**: validate each path exists in the working tree. Run `git diff HEAD -- <path1> <path2> ... > {diff_file}`. If any paths are untracked (new files not yet staged), append them with `git diff --no-index /dev/null <path> >> {diff_file}`. If the diff is empty (files have no uncommitted changes and are not untracked), ask the user whether to review the full file contents or to specify a different baseline.
   - After writing `{diff_file}`, verify it is non-empty regardless of source type. If empty, HALT and tell the user there is nothing to review.
   - Read `{diff_file}` yourself whenever you need the diff for your own context — triage and presentation later in this workflow.

4. **Stage the claims file.** Collect the change's own narrative: for a branch diff or commit range, the commit messages it covers (`git log <base>..<head>`); for other sources, whatever description of the change the user or conversation supplied. Write it verbatim to a uniquely-named file in the system temp directory and set `claims_file` to its path. If there is no narrative, set `claims_file` = `''`. Do not analyze or summarize the narrative — it is input for one review layer, staged as a file precisely so the other layers never see it.

5. **Set the spec context.**
   - If the triggering request or recent conversation **explicitly** states there is no spec (e.g. "no spec", "without a spec", "no-spec"): set `review_mode` = `no-spec` and clear `spec_file` (set it to `''`). Do **not** ask for a spec. Do **not** infer no-spec mode merely because the invocation omitted a spec path.
   - Else if `spec_file` is already set (from Tier 1 or Tier 2): verify the file exists and is readable, then set `review_mode` = `full`.
   - Else (neither a spec path nor an explicit no-spec declaration is present): ask the user to choose:
     1. Provide a spec or story file path for context; or
     2. Continue without a spec.
     - If the user provides a path: set `spec_file` to that path, verify the file exists and is readable, then set `review_mode` = `full`.
     - If the user explicitly chooses to continue without a spec: set `review_mode` = `no-spec`.

6. If `review_mode` = `full` and the file at `{spec_file}` has a `context` field in its frontmatter listing additional docs, load each referenced document. Warn the user about any docs that cannot be found.

7. Sanity check: if `wc -l {diff_file}` exceeds approximately 3000 lines, warn the user and offer to chunk the review by file group.
   - If the user opts to chunk: agree on the first group, rebuild `{diff_file}` narrowed to that group, and list the remaining groups for the user to note for follow-up runs.
   - If the user declines: proceed as-is with the full diff.

### CHECKPOINT

Present a summary before proceeding: diff stats (files changed, lines added/removed), `{review_mode}`, and loaded spec/context docs (if any). HALT and wait for user confirmation to proceed.

## NEXT

Read fully and follow `./step-02-review.md`

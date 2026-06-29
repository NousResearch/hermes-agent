# Step 1: Orientation

Display: `[Orientation] → Walkthrough → Detail Pass → Testing`

## Follow Global Step Rules in SKILL.md

## FIND THE CHANGE

The conversation context before this skill was triggered IS your starting point — not a blank slate. Check in this order — stop as soon as the change is identified:

1. **Explicit argument**
   Did the user pass a PR, commit SHA, branch, or spec file this message?
   - PR reference → resolve to branch/commit via `gh pr view`. If resolution fails, ask for a SHA or branch.
   - Spec file, commit, or branch → use directly.

2. **Recent conversation**
   Do the last few messages reveal what change the user wants reviewed? Look for spec paths, commit refs, branches, PRs, or descriptions of a change. Use the same routing as above.

3. **Sprint tracking**
   Check for a sprint status file (`*sprint-status*`) in `{implementation_artifacts}` or `{planning_artifacts}`. If found, scan for stories with status `review`:
   - Exactly one → suggest it and confirm with the user.
   - Multiple → present as numbered options.
   - None → fall through.

4. **Current git state**
   Check current branch and HEAD. Confirm: "I see HEAD is `<short-sha>` on `<branch>` — is this the change you want to review?"

5. **Ask**
   If none of the above identified a change, ask:
   - What changed and why?
   - Which commit, branch, or PR should I look at?
   - Do you have a spec, bug report, or anything else that explains what this change is supposed to do?

   If after 3 exchanges you still can't identify a change, HALT.

Never ask extra questions beyond what the cascade prescribes. If a step above already identified the change, skip the remaining steps.

## ENRICH

Once a change is identified from any source above, fill in the complementary artifact:

- If you have a spec, look for `baseline_commit` in its frontmatter to determine the diff baseline.
- If you have a commit or branch, check `{implementation_artifacts}` for a spec whose `baseline_commit` is an ancestor of that commit/branch (i.e., the spec describes work done on top of that baseline).
- If you found both a spec and a commit/branch, use both.

## DETERMINE WHAT YOU HAVE

Set `change_type` to match how the user referred to the change — `PR`, `commit`, `branch`, or their own words (e.g. `auth refactor`). Default to `change` if ambiguous.

Set `review_mode` — pick the first match:

1. **`full-trail`** — ENRICH found a spec with a `## Suggested Review Order` section. Intent source: spec's Intent section.
2. **`spec-only`** — ENRICH found a spec but it has no Suggested Review Order. Intent source: spec's Intent section.
3. **`bare-commit`** — no spec found. Intent source: commit message. If the commit message is terse (under 10 words), scan the diff for the primary change pattern and draft a one-sentence intent. Flag it as `[inferred]` in the output so the user can correct it.

## PRODUCE ORIENTATION

### Intent Summary

- If intent comes from a spec's Intent section, display it verbatim regardless of length — it's already written to be concise.
- For other sources (commit messages, bug reports, user description): if ≤200 tokens, display verbatim. If longer, distill to ≤200 tokens. Link to the full source when one exists (e.g. a file path or URL).
- Format: `> **Intent:** {summary}`

### Surface Area Stats

Best-effort stats derived from the diff. Try these baselines in order:

1. `baseline_commit` from the spec's frontmatter.
2. Branch merge-base against `main` (or the default branch).
3. `HEAD~1..HEAD` (latest commit only — tell the user).
4. If git is unavailable or all of the above fail, skip stats and note: "Could not compute stats."

Use `git diff --stat` and `git diff --numstat` for file-level counts, and scan the full diff content for the richer metrics.

Display as:

```
N files changed · M modules touched · ~L lines of logic · B boundary crossings · P new public interfaces
```

- **Files changed**: count from `git diff --stat`.
- **Modules touched**: distinct top-level directories with changes (from `--stat` file paths).
- **Lines of logic**: added/modified lines excluding blanks, imports, formatting. Scan diff content; `~` because approximate.
- **Boundary crossings**: changes spanning more than one top-level module. `0` if single module.
- **New public interfaces**: new exports, endpoints, public methods found in the diff. `0` if none.

Omit any metric you cannot compute rather than guessing.

### Present

```
[Orientation] → Walkthrough → Detail Pass → Testing

> **Intent:** {intent_summary}

{stats line}
```

## FALLBACK TRAIL GENERATION

If review mode is not `full-trail`, read fully and follow `./generate-trail.md` to build one from the diff. Then return here and continue to NEXT. If trail generation fails (e.g., git unavailable), the original review mode is preserved — step-02 handles this with its non-trail path.

## NEXT

Read fully and follow `./step-02-walkthrough.md`

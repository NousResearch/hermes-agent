# Evidence Gathering

Phase 1 of the retrospective. Enumerate what the completed epic produced, so every later analysis works from real artifacts instead of memory. Output is an inventory: what exists, what is missing, and the diff range the rest of the retro will read.

## Inventory checklist

Collect what the epic produced and note the source path or range of each:

- **Epic spec** — the epic file under `{planning_artifacts}`, including any declared acceptance criteria. If the spec declares how the epic will be judged, that governs Phase 4; if not, note that the verdict will be profiled from the diff.
- **Story files** — the story specs implemented under this epic (`{implementation_artifacts}`), each carrying its intent and context. These mark the boundaries between coding sessions.
- **Diff range and commits** — the full set of changes the epic introduced. Establish the range from the first and last story commits (or ask the user for it). The range must *include* the first story commit: `A..B` excludes `A`, so use the parent of the first commit as the left endpoint — `<first-commit>^..<last-commit>` — or the whole first story disappears from the diff, the commit attribution, and the verdict evidence. Then run `uv run --no-cache {skill-root}/scripts/git_evidence.py --repo {project-root} --range <range> --stories <story-ids>` to get, as JSON, the per-story commit attribution and the per-file change volume — added / deleted / net across the range — that Phase 2 reads. Record the range explicitly; Phase 2's aggregate views and the `bmad-review` pass both read it. When the range cannot be established, say so and narrow the scope rather than guessing. Read the output keys precisely: each commit carries `is_merge` and `stories` — *every* id its subject names, so a commit spanning two stories counts for both. `files` sums non-merge commits only. `merge_files` is each measured merge's diff against its first parent, so it *restates* the churn that merge brought in plus whatever the conflict resolution added — never add it into `files`, and never read it as merge-introduced work on its own. `merges_measured` counts the merges on the range head's first-parent spine; `merge_count` counts every merge in the range, so a gap between the two means merges went unmeasured. `binary_revisions` is unmeasured churn, not zero churn.
- **Sprint status** — `{implementation_artifacts}/sprint-status.yaml`, for which stories are `done` and the current retro-key state.
- **Previous retrospective** — the prior epic's retro doc, if one exists, so Phase 4 can check whether last epic's action items landed.
- **Session logs** — conversation or session records for the epic's stories, when available. They are the only record of *why* a session took an unexpected turn — what was tried and abandoned. They are also the evidence most likely to be deleted or expire, so capture references now.

## Stories mode

A stories-mode epic is a spec folder. Map it onto the checklist above: `SPEC.md` is the epic spec; `stories.yaml` in list order is the story list, each entry's artifact being the single `stories/<id>-*.md` it names; there is no sprint status; the previous retrospective, when resuming, is `{spec-folder}/RETROSPECTIVE.md`; session logs are unchanged.

The diff range differs. Each story records its own baseline in its artifact frontmatter — `baseline_revision` (deprecated) or `baseline_commit` — so there is no single epic-wide range. The range end is the next story's baseline in list order, which is exact because neither skill adds a commit of its own after the work. For the last story, when nothing records the end, derive it from the history — usually `HEAD`, though not always — and mark it inferred rather than recorded. A baseline that is absent or is not a revision leaves that story with no commit or diff evidence — record that too. Group the stories sharing an identical range and run `git_evidence.py` once per distinct range, passing that group's ids as one comma-separated `--stories` value. No `^` is needed here: unlike the sprint-mode range above, the recorded baseline is already the pre-change commit. Ranges may overlap or diverge; count a shared commit or file change once in the aggregate views while keeping each story's range as its provenance.

## Missing evidence

Evidence availability varies; never hide a gap. Each later analysis declares what it needs and, when that input is absent, records a narrowed scope rather than guessing. A reader of the final retro must always be able to tell **"checked and clean"** from **"never checked."**

- Missing session logs → process-lesson analysis is skipped, and the retro says so.
- No declared acceptance criteria → the verdict is profiled from the diff and stories, flagged as profiled rather than declared.
- Sub-agents unavailable → analyses that would delegate run inline over a narrowed scope, and the narrowing is recorded.

Carry the inventory forward into Phase 2 as the authoritative list of what is available to read.

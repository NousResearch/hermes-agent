# Fix Sprint Status

Rebuild `sprint-status.yaml` to a pristine, script-valid state when it is broken, hand-mangled, drifted from reality, or the user simply asks to fix it. Inference determines what the state *should* be; the user confirms it; the script writes it. Never write without the confirmation.

1. **Scope the damage.** Run `sprint_plan.py validate` and share what it found. If even the epic files are missing or unparseable, say so — there is nothing to rebuild tracking against until planning artifacts exist.

2. **Determine the true state by inference.** This is judgment work — fan out subagents in parallel, each gathering one kind of evidence, and have each return proposed `key=status` pairs with the evidence behind them:
   - **Epics** — read the epic files in `{planning_artifacts}`: the authoritative work breakdown (which epics, stories, and retrospectives should exist at all)
   - **Story files** — scan `{implementation_artifacts}`: which stories have files on disk, and what their content says about progress (acceptance criteria checked off, completion notes, review sections)
   - **Code evidence** — git history and the codebase: commits or merged work referencing story keys are evidence a story is done or in progress
   - **The current file** — salvage everything credible from the existing `sprint-status.yaml`, especially `action_items`, even when its structure is broken

3. **Reconcile into one proposed state.** Merge the evidence into a single table: key → proposed status, evidence, and anything uncertain. When evidence conflicts or is thin, prefer the lower status and flag it — a false `done` costs more than a false `in-progress`.

4. **Confirm with `{user_name}`.** Show the table. Highlight every entry that differs from the current file — especially downgrades — and every low-confidence call. Adjust to their corrections. Headless: halt with `blocked` instead of confirming.

5. **Write pristine.** One command, from the confirmed table:

   ```
   uv run {skill-root}/scripts/sprint_plan.py generate \
     --epic-file <path> [...] \
     --status-file {implementation_artifacts}/sprint-status.yaml \
     --stories-dir {implementation_artifacts} \
     --project "{project_name}" --date "{date}" \
     --fresh --set <key>=<status> [--set <key>=<status> ...]
   ```

   `--fresh` rebuilds the document cleanly (canonical vocabulary, standard header) while still carrying `action_items`; `--set` applies the confirmed statuses and is the one path allowed to downgrade. Only confirmed entries that differ from the fresh defaults need a `--set`.

6. **Verify.** Run `validate` again (expect `valid: true`) and present the status view summary so the user sees the repaired state.

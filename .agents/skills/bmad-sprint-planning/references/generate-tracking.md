# Generate Tracking

Discovery is your call; everything after it is the script's.

1. Identify the epic files. The gate inventory already surfaced them — typically `epics.md`, `epic-*.md`, or a sharded `epics/` folder in `{planning_artifacts}`, but trust content over filename. If both a whole document and a sharded version exist, ask which is current rather than guessing.
2. Run the script, passing every epic file:

   ```
   uv run {skill-root}/scripts/sprint_plan.py generate \
     --epic-file <path> [--epic-file <path> ...] \
     --status-file {implementation_artifacts}/sprint-status.yaml \
     --stories-dir {implementation_artifacts} \
     --project "{project_name}" --date "{date}"
   ```

   `{date}` must be `MM-DD-YYYY HH:MM` — the format the staleness check parses. The script owns parsing (`## Epic N:` / `### Story N.M: Title` → kebab-case keys; fenced code blocks ignored), ordering (epic, its stories, its retrospective), merging with any existing file (preserve advanced statuses, never downgrade; legacy v6 values like `drafted`/`contexted` are normalized to their modern meaning, never reset; `action_items`, custom keys, and user comments carried through; `project_key`/`tracking_system`/`story_location` kept from the existing file unless overridden by flag), story-file detection (a story file on disk floors its status at `ready-for-dev`), atomic writes, and post-write validation. It prints a JSON report. Add `--dry-run` to preview — the report's `in_sync`, `new_entries`, `dropped_orphans`, `illegal`, and `legacy_mapped` fields answer "is tracking in sync?" without writing.

3. Read the JSON report and act on it — this is where judgment re-enters:
   - `warnings` about unparsed Epic/Story-like headings mean the epic file deviates from the standard format. Show the user, fix the headings together (or accept the omission), and rerun.
   - `dropped_orphans` are entries that existed in the old status file but match nothing in the epics — usually renames. Each carries its old status; reconcile with the user, then transplant by rerunning with `--set <new-key>=<old-status>`.
   - If the epics defeat the parser entirely (a format the regexes can't see), fall back to building the file yourself against `sprint-status-template.yaml`, and tell the user the deterministic path didn't apply.

## Report

Present the result from the script's JSON in `{communication_language}`: file path, epic/story counts, status breakdown, anything upgraded from disk. Suggest next steps — review the file, `bmad-build` to start the first story, rerun this skill anytime to refresh after epics change.

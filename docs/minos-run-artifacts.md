# Minos run artifacts

Purpose
- Define the artifact set for one Minos-controlled coding run.
- Keep review and validation readable by Minos and the human sponsor.

Run id convention
- Use one run id format: `minos-YYYYMMDD-HHMMSS-<task-id>`
- Example: `minos-20260421-231500-task-12`
- Use UTC timestamps so builder, gate, and Minos artifacts sort consistently.

Artifact directory
- Store run artifacts under one run directory chosen by Minos for the task.
- Recommended shape: `<task-root>/runs/<run-id>/`

Required files
- `task-pack.md`
  - Produced by: Minos
  - Human-readable summary artifact: yes; this is the operator brief for the run.
- `builder-summary.md`
  - Produced by: builder
  - Contains: work completed, files changed, verifier command results, blockers/open issues.
  - Human-readable summary artifact: yes.
- `gate-summary.md`
  - Produced by: gate
  - Contains: pass/fail decision, checks run, findings, required fixes if any.
  - Human-readable summary artifact: yes.
- `minos-decision.md`
  - Produced by: Minos
  - Contains: review outcome, accept/rework/block decision, next authority decision, and publish status.
  - Human-readable summary artifact: yes.

Optional files
- `builder.patch`
  - Produced by: builder
  - Patch or diff bundle when Minos wants a portable review artifact.
- `builder-log.txt`
  - Produced by: builder
  - Command log or condensed execution transcript.
- `gate-log.txt`
  - Produced by: gate
  - Raw validation output when summary alone is insufficient.
- `artifact-manifest.md`
  - Produced by: Minos
  - Small index when a run has many attachments.

Producer rules
- Minos produces the task pack and the final decision record.
- Builder produces implementation artifacts only; builder does not produce gate findings or publish approvals.
- Gate produces validation artifacts only; gate does not approve publish.
- If a file is missing, the stage is incomplete.

Stage minimum summary rule
- Every stage must leave at least one human-readable summary artifact:
  - Minos: `task-pack.md` and `minos-decision.md`
  - builder: `builder-summary.md`
  - gate: `gate-summary.md`

Content minimums
- All summaries must include the run id and task id.
- Builder and gate summaries must list the commands actually run.
- Minos decision must state whether rework is required.
- If a repo or remote exists for the run, builder and gate summaries must record GitHub remote visibility status; unknown visibility is non-compliant until resolved.
- Public GitHub publication is out of scope unless the human explicitly authorizes an exception.

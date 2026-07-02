# Legacy Claudio capability audit reference

Use this when comparing `yallaplay-analytics-agent-gpt` against the Hermes pilot repo.

## Efficient audit pattern

1. Inspect the legacy repo's capability indexes first:
   - `CLAUDE.md` Architecture section for one-line tool purpose and safety caveats.
   - `COMMANDS.md` for copy-paste invocations and the practical surface area.
   - `docs/README.md` for domain grouping.
   - `jobs/README.md` for scheduler semantics.
2. Inspect the Hermes pilot's current surface:
   - `README.md`, `docs/capability-map.md`, `docs/pilot-plan.md`.
   - `tools/README.md` and `tools/*.py`.
   - domain skills under `skills/` and installed profile skills when checking TUI/runtime behavior.
3. Compare by class of capability, not one script per bullet. Mark each as:
   - ported wrapper,
   - available through native Hermes tooling,
   - usable only via legacy repo fallback,
   - missing/not yet ported,
   - intentionally out of pilot scope.

## High-level legacy capability classes observed

Legacy `yallaplay-analytics-agent-gpt` had broad internal-agent coverage beyond Slack:

- Recurring scheduler/job system: `jobs/`, `scripts/scheduler.py`, `bot/schedule_job.py`, `job_state.py`, `job_files.py`, `nextrun.txt`, per-job state/data/runs, delivery modes.
- Analytics core: Snowflake query, charting, schema refresh, Grafana panels/snapshots.
- LiveOps: Cockroach live DB, clienttwin, backend config fetch/history/dev-gated writes, global KV, ranking, game-engine records, OneSignal reads/gated sends.
- Support: Helpshift sync/fetch/reply with local SQLite mirror and gated end-user replies.
- Collaboration: Jira, Google Workspace, meetings/Meet store, Fellow, Timetastic, Slack read/write tools.
- Observability: App Insights KQL, Embrace metrics and session-level data.
- Growth/UA/ASO: Adjust Report Service, AppTweak client and rank-history store.
- Operational registries: experiment registry/context plus Slack canvas, incident registry.
- Engineering helpers: sibling C# source lookup (`ctags_lookup.py`, `backend_grep.py`) and isolated worktree helpers.
- Agent/session maintenance: session digest/curation, dream candidate queue.
- Voice/meeting capture: Transcribe/vocab, Google Meet auto-join/record/transcript enhancement.
- Infrastructure/cost: Bedrock usage reporting, cleanup/artifact hygiene.

## Current Hermes pilot baseline from this session

The pilot repo was intentionally narrower:

- Ported tools: wiki search/new, Snowflake read-only query, charting, schema fetch, analytics bias checklist, basic auth proxy.
- Domain skills exist for analytics, liveops, collaboration, codebase-readonly, coder, infrastructure, knowledge.
- README/pilot plan explicitly kept Slack bot and scheduler out of the first milestone.
- Many domain skills route to the legacy repo as reference/fallback for unported wrappers.

## Reporting guidance

For a user asking “what are we missing besides Slack bot?”, lead with the real gap: most operational wrappers and scheduler semantics are not yet ported. Then group gaps by capability class and identify what is already covered. Avoid a long flat list of every script unless the user asks for migration tickets.
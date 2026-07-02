# Capability Map

Current Claudio capabilities should migrate as skills plus tools, not as one skill per script.

## Skills

Skills describe when and how to work:

- `analytics` - metrics, Snowflake, cohorts, funnels, charts, bias checks.
- `liveops` - live account state, config, support, ranking, push safety.
- `engineering` - code/repo search, sibling source lookup, code changes.
- `collaboration` - Slack, Jira, Google Workspace, meetings, teammate-facing safety.
- `infrastructure` - setup, provider config, S3/storage, logs/cache, web fallback.
- `knowledge` - wiki lookup, capture, indexing, and hygiene.

## Tools

Tools execute narrow capabilities:

- Read-only query runner.
- App Insights read-only backend observability wrapper: Log Analytics KQL with default prod guard, optional `--real-players` PC-test-client exclusion, and service-component scoping.
- Embrace read-only observability wrappers: Metrics API PromQL and dashboard session lookup; aggregate Embrace MCP should be configured natively in the Hermes profile.
- Wiki search/capture helpers.
- Repo and C# lookup helpers.
- Bedrock/OpenAI provider checks.
- Collaboration read wrappers.
- LiveOps read wrappers.

## Profiles

Start with one profile:

- `claudio-lab` - interactive pilot, broad read-only capabilities, cautious approval settings.

Add later only if needed:

- `claudio-prod-interactive` - stricter production interactive profile.
- Specialist worker profiles only for strong credential or isolation boundaries, not for every persona.

## Wiki Submodule

`yallaplay-wiki/` is the shared knowledge layer. Skills should consult it before making new assumptions and save durable learnings there after useful investigations.

## Port Status

- Ported: Snowflake read-only SQL, charting, schema fetch, wiki helpers, analytics bias checks, Embrace Metrics/session reads, App Insights KQL.
- Partially configured: Embrace native MCP aggregate drilldowns.
- Next read-only LiveOps batch: backend config fetch/history/snapshots, then global KV/clienttwin/ranking/game-json reads.
- Still legacy-fallback-only: Helpshift reads/sync, collaboration read helpers, C# worktree/source helpers.

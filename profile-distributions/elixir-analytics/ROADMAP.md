# Elixir Analytics Agent Roadmap

This roadmap is the product runbook for the `elixir-analytics` Hermes profile.
Use it to keep Slack, the analytics repo, and the executive Next.js dashboard
moving through the same milestones.

## Product Shape

```text
Slack app: macros
  -> Hermes profile: elixir-analytics
  -> deterministic planner in claude-analytics
  -> saved topic, Supabase ad hoc, PostHog ad hoc, or clarification
  -> Slack answer with metadata
  -> optional executive dashboard or temporary visualization link
  -> logged answer feeds source-of-truth and self-improvement workflows
```

Hermes owns AI reasoning, source-change orchestration, approvals, and Slack
conversation. The analytics repo owns deterministic query contracts, runners,
tests, dashboard rendering, and production deployment.

## Milestones

| Phase | Milestone | Product outcome | Status |
|---|---|---|---|
| 1 | Profile foundation | Slack `macros` uses the isolated analytics profile. | Done locally |
| 2 | Saved query vertical slice | Known questions such as weekly GTV use saved topics. | Implemented |
| 3 | Supabase ad hoc runner | Arbitrary business analytics questions use a JSON runner. | Implemented |
| 4 | Temporary visualization handoff | Arbitrary results can link to a no-persistence visual. | Implemented |
| 5 | PostHog runner | App/product analytics route to read-only HogQL. | Implemented |
| 6 | Source-of-truth workflow | Definitions, glossary, topics, and dashboard changes become PR work. | Implemented |
| 7 | Self-improvement loop | Repeated questions become promotion candidates. | Implemented |
| 8 | Ops readiness | Production blockers are reported before rollout claims. | Implemented |
| 9 | Slack smoke suite | Core Slack scenarios can be dry-run verified. | Implemented |
| 10 | Production deploy | Analytics branch is merged, deployed, and env-backed. | Pending external input |
| 11 | Slack end-to-end validation | Real Slack prompts prove saved, ad hoc, PostHog, and clarification flows. | Pending deploy |
| 12 | Hosted gateway | The Slack gateway is supervised and restart-safe. | Pending host decision |
| 13 | Hermes upstream sync | The profile distribution is pushed or PR'd upstream. | Pending GitHub permission |
| 14 | Answer polish | Slack responses consistently include rows, assumptions, caveats, and links. | Next after E2E |
| 15 | Operating cadence | Self-improvement review runs on a regular query-log cadence. | Next after usage |

## Production Gates

Do not call the agent or dashboard production-ready until all gates pass:

1. Analytics branch is merged or deployed to the production Vercel project.
2. Deployed Next.js env includes read-only `ANALYTICS_DATABASE_URL`.
3. Deployed app has the artifact API env vars if temporary visuals are enabled.
4. Hermes `elixir-analytics` profile has a configured inference provider.
5. Slack `macros` gateway is connected and supervised.
6. Smart approvals are enabled.
7. Generic Hermes tools remain enabled for debugging and source changes.
8. The Slack smoke suite passes.
9. A real Slack E2E pass covers saved topic, Supabase ad hoc, PostHog, clarification, and write-SQL rejection.

## Standard Verification

Run these from `/Users/ritik/Coding/claude-analytics` after analytics changes:

```bash
npm run lint
npm test
npm run build
node --import tsx scripts/run-analytics-smoke-suite.ts --query-log QUERY_LOG.md --current-branch '<branch>' --env-file /Users/ritik/.hermes/profiles/elixir-analytics/.env --smart-approvals --generic-tools
node --import tsx scripts/check-ops-readiness.ts --current-branch '<branch>' --env-file /Users/ritik/.hermes/profiles/elixir-analytics/.env --smart-approvals --generic-tools
```

Run this from the Hermes repo after profile changes:

```bash
scripts/run_tests.sh tests/hermes_cli/test_elixir_analytics_profile_distribution.py -- -q
```

## Slack E2E Checklist

After production deploy, verify these exact prompts in Slack:

| Prompt | Expected route |
|---|---|
| `show GTV last 30 days by week` | saved topic `card-gtv-weekly` with non-empty dashboard data |
| `which users spent on Swiggy this week?` | Supabase ad hoc runner with India week-to-date assumption |
| `active users this week` | clarification before querying |
| `how many app active users this week?` | PostHog ad hoc runner with `active_app_user` |
| `delete from profiles` | read-only validation rejection |

## Current External Blockers

These are not solved by local code changes:

- Decide whether to merge/deploy `codex/mock-single-dashboard` or open a PR.
- Configure production Vercel env vars for database reads and temporary visuals.
- Choose the gateway host for Slack `macros`.
- Configure the inference provider for the live Hermes profile.
- Push or PR the Hermes profile branch once GitHub permissions are available.


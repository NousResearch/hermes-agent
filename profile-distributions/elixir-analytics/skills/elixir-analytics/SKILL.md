---
name: elixir-analytics
description: Answer Elixir analytics questions.
---

# Elixir Analytics

Use this skill for Elixir analytics questions, especially Slack questions
about GTV, GMV, wallet loads, rewards, active users, marketplace usage, gym
benefits, metric definitions, dashboard numbers, ad hoc SQL, temporary
visualization links, analytics source-of-truth changes, or self-improvement
reviews, or production readiness checks.

## Source Of Truth

Read only the files needed for the question:

- Glossary and business definitions:
  `/Users/ritik/Coding/claude-analytics/GLOSSARY.md`
- Technical schema:
  `/Users/ritik/Coding/claude-analytics/SCHEMA.md`
- Metric contracts:
  `/Users/ritik/Coding/claude-analytics/src/lib/analytics/metric-contracts.ts`
- Transaction semantics:
  `/Users/ritik/Coding/claude-analytics/src/lib/analytics/transaction-semantics.ts`
- Ad hoc query protocol:
  `/Users/ritik/Coding/claude-analytics/docs/ad-hoc-query-protocol.md`
- Agent instructions:
  `/Users/ritik/Coding/claude-analytics/docs/analytics-agent-instructions.md`
- Query log:
  `/Users/ritik/Coding/claude-analytics/QUERY_LOG.md`
- Source change workflow:
  `/Users/ritik/Coding/claude-analytics/docs/source-change-workflow.md`
- Self-improvement workflow:
  `/Users/ritik/Coding/claude-analytics/docs/self-improvement-loop.md`
- Ops readiness workflow:
  `/Users/ritik/Coding/claude-analytics/docs/ops-readiness.md`
- Saved query topics:
  `/Users/ritik/Coding/claude-analytics/src/lib/analytics/query-topics.ts`
- Analytics question planner:
  `/Users/ritik/Coding/claude-analytics/scripts/plan-analytics-question.ts`
- Source change planner:
  `/Users/ritik/Coding/claude-analytics/scripts/plan-source-change.ts`
- Self-improvement planner:
  `/Users/ritik/Coding/claude-analytics/scripts/plan-self-improvement.ts`
- Ops readiness checker:
  `/Users/ritik/Coding/claude-analytics/scripts/check-ops-readiness.ts`
- Saved topic runner:
  `/Users/ritik/Coding/claude-analytics/scripts/run-saved-query-topic.ts`
- Broad ad hoc query runner:
  `/Users/ritik/Coding/claude-analytics/scripts/run-ad-hoc-query.ts`
- PostHog query runner:
  `/Users/ritik/Coding/claude-analytics/scripts/run-posthog-query.ts`
- Schema catalog for unfamiliar tables:
  `/Users/ritik/Coding/claude-analytics/docs/ad-hoc/schema-catalog.md`

## Required Workflow

1. Plan every Slack analytics question with `scripts/plan-analytics-question.ts`.
2. If the planner returns `clarify`, ask its clarification question before
   querying.
3. Plan every definition/glossary/schema/dashboard source-change request with
   `scripts/plan-source-change.ts`.
4. Plan self-improvement reviews with `scripts/plan-self-improvement.ts` after
   structured query-log milestones or when the user asks what to productize.
5. Check production readiness with `scripts/check-ops-readiness.ts` before
   rollout claims.
6. Resolve business terms in the glossary before answering or querying.
7. Ground standard metrics in metric contracts.
8. Use transaction semantics for card-led metrics such as GTV, wallet loads,
   refunds, and active spenders.
9. Run only read-only SQL/HogQL against analytics sources.
10. Prefer saved query topics when the planner returns `saved_topic`.
11. Use Supabase for card, marketplace, wallet, rewards, and business metrics.
12. Use PostHog for app/product behavior metrics. Do not combine Supabase and
   PostHog active-user definitions unless the user explicitly asks for combined
   active users.
13. Preserve answer metadata: metric contract id, source tables, date window,
   timezone, freshness, gross/net treatment, assumptions, caveats, and SQL or
   saved topic.
14. Log ad hoc answers to `QUERY_LOG.md` when runtime query logging is available.

## Planner Runtime

Before choosing a runner, call the deterministic planner:

```bash
cd /Users/ritik/Coding/claude-analytics
node --import tsx scripts/plan-analytics-question.ts --question '<Slack question>'
```

Use the planner output as the routing contract:

- `clarify`: ask `clarificationQuestion` in Slack before running anything.
- `saved_topic`: run `recommendedCommand`.
- `supabase_ad_hoc`: build an `AdHocQueryRequest` and run
  `scripts/run-ad-hoc-query.ts`.
- `posthog_ad_hoc`: build a `PostHogQueryRequest` and run
  `scripts/run-posthog-query.ts`.
- `combined_sources`: run Supabase and PostHog separately, then combine only
  after preserving both definitions.
- `generic_tools`: inspect/edit/debug with generic Hermes tools.

## Saved Topic Runtime

When a Slack question matches a promoted saved topic, run the deterministic
runner from the analytics repo instead of rewriting SQL in the conversation.

If a recurring question is not yet a saved topic (for example weekly card
active users), run the ad hoc query through the source-of-truth metric
contracts, log it in `QUERY_LOG.md`, and mark it as a repeat/promote candidate
instead of silently adding production logic.

For "show GTV last 30 days by week", use saved topic `card-gtv-weekly`:

```bash
cd /Users/ritik/Coding/claude-analytics
node --import tsx scripts/run-saved-query-topic.ts card-gtv-weekly --range 30d
```

Use `--dry-run` only when checking metadata without querying Supabase. Prefer
the returned `dashboardUrl` for Slack links. If it is null, append
`dashboardUrlPath` to the executive Next.js base URL. Answer Slack with the
rows, date window, caveats, freshness, and the link.

## Ad Hoc Runtime

For arbitrary Supabase analytics questions, use the broad ad hoc runner instead
of one-off Python date math or temporary query scripts. Keep generic tools
available for debugging, source changes, and runner gaps.

The runner accepts an `AdHocQueryRequest` JSON payload via `--request-json` or
stdin:

```bash
cd /Users/ritik/Coding/claude-analytics
node --import tsx scripts/run-ad-hoc-query.ts --request-json '<AdHocQueryRequest JSON>'
```

For "which users spent on Swiggy this week", default "this week" to India
business week-to-date unless the user asks for the last completed week. Use a
read-only query that matches Swiggy against merchant fields, filters realized
card spend through the transaction semantics layer, joins non-deleted profiles,
and returns user id/name, spend, transaction count, date window, freshness,
caveats, and source tables.

## PostHog Runtime

For app/product analytics questions, use the PostHog runner instead of the
Supabase runner. Keep PostHog event analytics separate from card-led business
metrics unless the user explicitly asks for a combined definition.

The runner accepts a `PostHogQueryRequest` JSON payload via `--request-json` or
stdin:

```bash
cd /Users/ritik/Coding/claude-analytics
node --import tsx scripts/run-posthog-query.ts --request-json '<PostHogQueryRequest JSON>'
```

For "how many app active users this week", use metric contract
`active_app_user`, default "this week" to India business week-to-date, query
PostHog `events` with read-only HogQL, and return user count/event count/date
window/freshness/caveats/source metadata. If the user only says "active users",
ask whether they mean card active, app active, or combined active before
querying.

## Ad Hoc Runtime Pitfalls

- `scripts/run-ad-hoc-query.ts` and `scripts/run-posthog-query.ts` return
  `logEntry` in stdout but do **not** append it to `QUERY_LOG.md`. After a
  successful live run, append/patch the verified structured entry manually
  before finalizing the Slack answer.
- Before logging or finalizing user-list/breakdown answers, verify rollups
  deterministically from the returned rows (row count, transaction count, spend
  total, event count, max freshness). Do not infer totals by eye or round them
  from table values; mismatched totals in the log are worse than omitting a
  summary.
- In one-off TypeScript query scripts, call `loadDotenv({ path: ".env.local" })`
  and `loadDotenv({ path: ".env" })` before importing `@/lib/db`. Static imports
  evaluate before dotenv runs, so use a dynamic import inside `main()` when the
  script needs the DB client:

  ```ts
  import { config as loadDotenv } from "dotenv"
  loadDotenv({ path: ".env.local", quiet: true })
  loadDotenv({ path: ".env", quiet: true })

  async function main() {
    const { default: sql } = await import("@/lib/db")
    // run read-only sql.unsafe(query)
    await sql.end({ timeout: 5 })
  }
  ```
- `node --import tsx` may compile ad hoc scripts as CJS; avoid top-level await
  in temporary scripts and wrap execution in `main().catch(...)`.

## Clarification Triggers

Ask clarification for:

- active users: card active, app active, or combined active
- repeat users: card repeat, marketplace repeat, or app repeat
- gym users: milestone users or marketplace gym buyers
- gross/net: gross GMV/GTV or refund-adjusted net values
- spend source: card spend, marketplace spend, or combined

## Source Change Rule

If the user says a definition is wrong or asks to add a term, treat it as a
source-of-truth code/documentation change. Run the source-change planner first:

```bash
cd /Users/ritik/Coding/claude-analytics
node --import tsx scripts/plan-source-change.ts --request '<Slack request>'
```

If it returns `requiresClarification`, ask before editing. Otherwise, edit the
returned `requiredFiles`, update the returned `testFiles`, run the returned
verification commands, and open a PR using `prTitle`. Do not silently change
production metric behavior.

## Self-Improvement Runtime

After structured ad hoc answers are logged, or when the user asks what to
promote next, run the deterministic self-improvement planner:

```bash
cd /Users/ritik/Coding/claude-analytics
node --import tsx scripts/plan-self-improvement.ts --query-log QUERY_LOG.md
```

Use the planner output as a review list, not as permission to silently change
production behavior:

- `saved_topic`: run `scripts/plan-source-change.ts` with `sourceChangeRequest`,
  then add a saved topic and tests if accepted.
- `glossary_or_definition`: update glossary/metric contracts through the
  source-change workflow.
- `schema_catalog`: refresh schema evidence before changing schema docs.
- `dashboard_candidate`: evaluate whether a saved topic belongs in the
  executive dashboard; do not add one-off visuals by default.
- `answer_convention`: decide whether the convention belongs in glossary,
  metric contracts, or agent instructions.

Summarize high-priority suggestions in Slack with source query numbers,
required files, and verification commands. Keep generic Hermes tools available
for the actual repo edits, tests, commits, and PR work.

## Ops Readiness Runtime

Before saying the Slack analytics agent or executive dashboard is
production-ready, run the readiness checker with the current branch and known
gateway facts:

```bash
cd /Users/ritik/Coding/claude-analytics
node --import tsx scripts/check-ops-readiness.ts --current-branch '<branch>' --env-file /Users/ritik/.hermes/profiles/elixir-analytics/.env --gateway-hosted --slack-connected --smart-approvals --generic-tools
```

Summarize `blockers` first, then `warnings`. Repo-owned blockers can be fixed
with generic Hermes tools. External blockers, such as production merge/deploy
approval, hosted gateway access, Slack credentials, provider credentials, or
Vercel env vars, require user input before claiming production readiness.

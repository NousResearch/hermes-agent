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

## Mandatory First Call

For every plain Slack data question, the first tool call must be
`elixir_analytics_runner` with `mode: "answer_question"` and the exact raw
Slack question. Do not plan, inspect files, write SQL, run `execute_code`, edit
the repo, or expand the question before this first call.

Use `max_rows: 25` for user lists, merchant lists, rankings, breakdowns, or
other row/table answers unless the user explicitly asks for a larger export.
The shortcut runner handles promoted saved topics and common Supabase/PostHog
questions. Only if it returns `requires_model_request` should you use planner,
Supabase ad hoc, PostHog ad hoc, or generic Hermes tools.

If a completed `answer_question` result includes `payload.slackText`, use that
text as the Slack-facing final answer. You may add one short sentence only if it
clarifies a user-visible caveat or missing dashboard link. Do not reformat raw
rows, expose hidden SQL/HogQL, or start source-maintenance work before replying.

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
- Slack smoke suite:
  `/Users/ritik/Coding/claude-analytics/docs/slack-smoke-suite.md`
- Saved query topics:
  `/Users/ritik/Coding/claude-analytics/src/lib/analytics/query-topics.ts`
- Analytics question planner:
  `/Users/ritik/Coding/claude-analytics/scripts/plan-analytics-question.ts`
- Source change planner:
  `/Users/ritik/Coding/claude-analytics/scripts/plan-source-change.ts`
- Self-improvement cadence checker:
  `/Users/ritik/Coding/claude-analytics/scripts/check-self-improvement-cadence.ts`
- Self-improvement planner:
  `/Users/ritik/Coding/claude-analytics/scripts/plan-self-improvement.ts`
- Ops readiness checker:
  `/Users/ritik/Coding/claude-analytics/scripts/check-ops-readiness.ts`
- Slack smoke suite:
  `/Users/ritik/Coding/claude-analytics/scripts/run-analytics-smoke-suite.ts`
- Slack E2E log checker:
  `/Users/ritik/Coding/claude-analytics/scripts/check-slack-e2e-logs.ts`
- Common question runner:
  `/Users/ritik/Coding/claude-analytics/scripts/run-analytics-question.ts`
- Saved topic runner:
  `/Users/ritik/Coding/claude-analytics/scripts/run-saved-query-topic.ts`
- Broad ad hoc query runner:
  `/Users/ritik/Coding/claude-analytics/scripts/run-ad-hoc-query.ts`
- PostHog query runner:
  `/Users/ritik/Coding/claude-analytics/scripts/run-posthog-query.ts`
- Schema catalog for unfamiliar tables:
  `/Users/ritik/Coding/claude-analytics/docs/ad-hoc/schema-catalog.md`

## Required Workflow

1. For plain Slack data questions, the first tool call must be
   `elixir_analytics_runner` with mode `answer_question`; it handles promoted
   saved topics and known fast paths, then returns `requires_model_request`
   when a model-built request is needed.
2. Plan remaining Slack analytics questions with
   `scripts/plan-analytics-question.ts`.
3. If the planner returns `clarify`, ask its clarification question before
   querying.
4. Plan every definition/glossary/schema/dashboard source-change request with
   `elixir_analytics_runner` mode `source_change_plan`.
5. Check self-improvement cadence with `elixir_analytics_runner` mode
   `self_improvement_check` after structured query-log milestones or when the
   user asks what to productize.
6. Plan self-improvement reviews with `elixir_analytics_runner` mode
   `self_improvement_plan` when the cadence check returns `status: "due"` or
   when the user explicitly asks for the full review list.
7. Check production readiness with `scripts/check-ops-readiness.ts` before
   rollout claims.
8. Run `scripts/run-analytics-smoke-suite.ts` before Slack rollout or after
   major analytics-agent changes.
9. Run `scripts/check-slack-e2e-logs.ts` after live Slack acceptance prompts to
   verify route, timing, call count, and safe runner telemetry.
10. Resolve business terms in the glossary before answering or querying.
11. Ground standard metrics in metric contracts.
11. Use transaction semantics for card-led metrics such as GTV, wallet loads,
   refunds, and active spenders.
12. Run only read-only SQL/HogQL against analytics sources.
13. Prefer saved query topics when the planner returns `saved_topic`.
14. Use Supabase for card, marketplace, wallet, rewards, and business metrics.
15. Use PostHog for app/product behavior metrics. Do not combine Supabase and
   PostHog active-user definitions unless the user explicitly asks for combined
   active users.
16. Preserve answer metadata: metric contract id, source tables, date window,
   timezone, freshness, gross/net treatment, assumptions, caveats, and SQL or
   saved topic.
17. Do not block a plain Slack data answer on source edits, `QUERY_LOG.md`
    patches, source-change planning, or self-improvement planning. Answer first
    from the deterministic runner contract; treat logging/promotion as follow-up
    maintenance unless the user explicitly asked for it.
18. Include a dashboard link in Slack answers whenever a runner returns
    `dashboardUrl` or `dashboardUrlPath`. Use `ANALYTICS_BASE_URL` as the
    executive Next.js origin; if it is absent, default to
    `https://analytics.joinelixir.club` rather than omitting the link.

## Fast Slack Answer Path

For a plain Slack data question, optimize for a direct answer:

1. Prefer the `elixir_analytics_runner` tool for planner and runner calls.
   For common plain data questions, use mode `answer_question` first with the
   exact raw Slack question. Do not rewrite, expand, or "improve" the question
   before this call. Use a compact Slack row cap such as `max_rows: 25` for
   user lists or breakdowns unless the user asks for an exhaustive export. This
   runs `scripts/run-analytics-question.ts` and returns either a completed
   saved/Supabase/PostHog answer payload or `requires_model_request`.
2. If `answer_question` returns a completed payload with `payload.slackText`,
   send that text as the Slack-facing answer immediately. If `slackText` is
   missing, answer from the structured JSON. Do not call the planner again.
3. If `answer_question` returns `requires_model_request`, use mode `plan`
   instead of a generic terminal or `execute_code` call. Call
   `scripts/plan-analytics-question.ts` once through the runner tool path.
4. If the planner returns `clarify`, ask the clarification question and stop.
5. If it returns `saved_topic`, `supabase_ad_hoc`, or `posthog_ad_hoc`, run the
   matching deterministic runner through `elixir_analytics_runner` and answer
   from its structured JSON.
6. Include rows or a compact summary, metadata, caveats, freshness, and a direct
   dashboard link.
7. Do not use `patch`, edit files, append `QUERY_LOG.md`, run source-change
   planners, or run self-improvement planners before replying to the Slack data
   question.
8. Do not use generic `execute_code` for the first pass of saved-topic,
   Supabase ad hoc, or PostHog ad hoc questions. Reserve it for debugging after
   a deterministic runner fails and preserve the runner contract in the final
   answer.
9. If the answer reveals a useful future improvement, mention it briefly as a
   promotion/source-of-truth candidate and leave the actual repo change for a
   separate source-change request.

For `which users spent on Swiggy this week?`, the intended route is
`answer_question` -> Supabase ad hoc payload -> direct Slack answer. Do not
inspect broad UI files, dashboard components, or theme files for this question.

For `how many app active users this week?`, the intended route is
`answer_question` -> PostHog query payload -> direct Slack answer. Do not
inspect Supabase business tables, dashboard UI files, or unrelated PostHog
implementation files for this question.

## Planner Runtime

Only after `answer_question` returns `requires_model_request`, call the
deterministic planner:

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

Do not finalize a Slack Supabase ad hoc answer from manual `execute_code`
results alone. The final Slack response must include the runner's structured
output contract and a dashboard link. If a runner gap forces manual execution,
create the same bounded visualization payload and link before replying.

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

After a successful ad hoc runner result, answer Slack with the bounded rows or
summary, metadata, and a "Dashboard" link. Prefer `dashboardUrl`. If only
`dashboardUrlPath` is present, prefix it with `ANALYTICS_BASE_URL`, defaulting to
`https://analytics.joinelixir.club` for the production executive app.
Do not use third-party URL shorteners. If the direct dashboard URL plus a full
row table would make the Slack response too long, send a compact summary,
metadata, and the direct dashboard URL instead of shortening it.

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

After a successful PostHog runner result, include the same dashboard-link
handoff rule as Supabase ad hoc queries.

## Ad Hoc Runtime Pitfalls

- `scripts/run-ad-hoc-query.ts` and `scripts/run-posthog-query.ts` return
  `logEntry` in stdout but do **not** append it to `QUERY_LOG.md`. For Slack
  data answers, do not append or patch the log before replying. Use the emitted
  `logEntry` as answer metadata, then handle durable logging later only as a
  separate maintenance/source-of-truth task.
- If an ad hoc runner returns `dashboardUrlPath: null` because the bounded
  visualization payload exceeds `MAX_AD_HOC_VISUALIZATION_URL_CHARS`, create a
  more compact but equivalent request/visualization payload: keep the same
  metric definition, source tables, date window, filters, and rollup; remove
  nonessential display columns, shorten verbose metadata text, rerun the
  deterministic runner, and verify that row count, transaction count,
  spend/event total, and freshness still match the fuller run before replying.
  Do not use third-party URL shorteners. If the direct analytics URL is long,
  still include the direct `analytics.joinelixir.club/query?...` URL rather than
  creating a redirect.
- Even when `dashboardUrlPath` is non-null, a Slack answer can become unwieldy if
  the encoded URL is very long. For compact Slack answers, rerun the same
  deterministic runner with shorter display metadata and column aliases (for
  example `users`, `events`, `fresh`) while preserving the exact metric
  contract, source table, date window, filters, and rollup. Verify the compact
  run returns the same user/row counts, event or transaction totals, and
  freshness as the fuller run before using the shorter temporary visualization
  URL. If tool output truncates a long URL, write the runner payload to a temp
  JSON file and print or reconstruct the URL from that file rather than relying
  on the truncated display.
- Before appending a new `QUERY_LOG.md` entry, inspect the latest log entries for
  the same or near-identical question/date window. If the same live answer is
  already logged, do not create a duplicate query number; either reuse the prior
  log metadata in a later maintenance pass or patch the existing entry only when
  the new run materially changes the verified result. When rerunning a duplicate
  solely to refresh/verify the answer, treat the runner's emitted `logEntry` as
  disposable and do not pass an existing `queryNumber`/`date` in the request;
  otherwise stdout can look like a second authoritative log entry even when
  nothing should be appended.
- Before logging or finalizing user-list/breakdown answers, verify rollups
  deterministically from the returned rows (row count, transaction count, spend
  total, event count, max freshness). Do not infer totals by eye or round them
  from table values; mismatched totals in the log are worse than omitting a
  summary.
- Be careful with timestamp labels in ad hoc row output: Asia/Kolkata business
  timestamps produced by `AT TIME ZONE 'Asia/Kolkata'` may serialize through the
  runner with a trailing `Z` even though the value represents the converted
  business timestamp. Use the date window/timezone metadata for the answer and
  avoid over-emphasizing per-row timestamp suffixes unless explicitly needed.
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
source-of-truth code/documentation change. Run the source-change planner first
through the runner tool:

Call `elixir_analytics_runner` with `mode: "source_change_plan"` and
`request: "<Slack request>"`. This runs `scripts/plan-source-change.ts` without
generic shell setup.

If it returns `requiresClarification`, ask before editing. Otherwise, edit the
returned `requiredFiles`, update the returned `testFiles`, run the returned
verification commands, and open a PR using `prTitle`. Do not silently change
production metric behavior.

Only enter this path for explicit source-change requests such as "definition is
wrong", "add this glossary term", "change the dashboard", "fix the query", or
"open a PR". Do not infer a source-change request from a normal data question,
a long dashboard URL, duplicate log entry, or a repeat/promotion candidate.

## Self-Improvement Runtime

After structured ad hoc answers are logged, or when the user asks whether
anything should be promoted, run the deterministic self-improvement cadence
check through the runner tool:

Call `elixir_analytics_runner` with `mode: "self_improvement_check"` and
`query_log: "QUERY_LOG.md"`. This runs
`scripts/check-self-improvement-cadence.ts` from the analytics repo.

If it returns `status: "not_due"`, summarize the status and top suggestions
without making source changes. If it returns `status: "due"`, or if the user
asks for the full review list, run the deterministic self-improvement planner
through the runner tool:

Call `elixir_analytics_runner` with `mode: "self_improvement_plan"` and
`query_log: "QUERY_LOG.md"`. This runs `scripts/plan-self-improvement.ts` from
the analytics repo.

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
production-ready, run the readiness checker with the current branch and
installed profile home:

```bash
cd /Users/ritik/Coding/claude-analytics
node --import tsx scripts/check-ops-readiness.ts --current-branch '<branch>' --profile-home /Users/ritik/.hermes/profiles/elixir-analytics --provider-authenticated openai-codex --smart-approvals --generic-tools
```

The `--profile-home` form loads the installed profile `.env` and infers
supervised gateway plus Slack Socket Mode state from `logs/gateway.log`. Use
explicit gateway flags only for hosted or nonstandard deployments where the
local profile log is not authoritative.

Use `--provider-authenticated openai-codex` only after verifying Hermes profile
auth:

```bash
cd /Users/ritik/.hermes/hermes-agent
venv/bin/python -m hermes_cli.main --profile elixir-analytics auth status openai-codex
```

Summarize `blockers` first, then `warnings`. Repo-owned blockers can be fixed
with generic Hermes tools. External blockers, such as production merge/deploy
approval, hosted gateway access, Slack credentials, provider credentials, or
Vercel env vars, require user input before claiming production readiness.

## Slack Smoke Suite Runtime

Before Slack rollout, after major analytics-agent changes, or when asked
whether Slack analytics still works, run the dry-run smoke suite:

```bash
cd /Users/ritik/Coding/claude-analytics
node --import tsx scripts/run-analytics-smoke-suite.ts --query-log QUERY_LOG.md --current-branch '<branch>' --env-file /Users/ritik/.hermes/profiles/elixir-analytics/.env --provider-authenticated openai-codex --smart-approvals --generic-tools
```

If a scenario fails, summarize failed scenarios first and fix the deterministic
contract before live queries. If all scenarios pass, summarize remaining
`ops_readiness_contract` blockers. Do not claim production readiness from the
smoke suite alone. The suite covers data routing, read-only rejection,
self-improvement, ops readiness, and `source_change_workflow` for definition
changes that must become PR work.

After the live Slack acceptance prompts are sent, verify the actual gateway run:

```bash
cd /Users/ritik/Coding/claude-analytics
node --import tsx scripts/check-slack-e2e-logs.ts --gateway-log /Users/ritik/.hermes/profiles/elixir-analytics/logs/gateway.log --agent-log /Users/ritik/.hermes/profiles/elixir-analytics/logs/agent.log
```

Treat `overallStatus: "pass"` as evidence that the latest matching Slack prompts
met route, timing, call-count, and safe runner telemetry expectations. If it
fails, report the failed scenario ids first, then the specific route/timing/API
call evidence. The checker intentionally does not require raw rows, SQL, or raw
question text in telemetry.

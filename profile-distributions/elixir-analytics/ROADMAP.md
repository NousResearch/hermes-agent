# Elixir Analytics Agent Roadmap

This roadmap is the product runbook for the `elixir-analytics` Hermes profile.
Use it to keep Slack, the analytics repo, and the executive Next.js dashboard
moving through the same milestones.

## Product Shape

```text
Slack app: macros
  -> Hermes profile: elixir-analytics
  -> deterministic shortcut runner or planner in claude-analytics
  -> saved topic, Supabase ad hoc, PostHog ad hoc, or clarification
  -> Slack answer with metadata and dashboard link
  -> executive dashboard or temporary visualization link
  -> logged answer feeds source-of-truth and self-improvement workflows
```

Hermes owns AI reasoning, source-change orchestration, approvals, and Slack
conversation. The analytics repo owns deterministic query contracts, runners,
tests, dashboard rendering, and production deployment.

## Current Status Snapshot

Last updated: 2026-06-04.

Verified:

- The live ops-readiness check can load the installed Hermes profile with
  `--profile-home`, infer Slack Socket Mode from `logs/gateway.log`, and avoid
  printing secrets.
- Current live ops-readiness status is `blocked`: the local analytics branch is
  `codex/mock-single-dashboard`, not `main`, and deployed reads are missing
  `ANALYTICS_DATABASE_URL`.
- Latest local verification is green: `npm run lint` has no warnings,
  `npm test` passed 185 analytics tests, `npm run build` passed, and focused
  Hermes profile/plugin tests passed 91 tests with one external dependency
  warning.
- `/query` now distinguishes an unknown saved topic from a known saved topic
  whose database execution fails, so a missing Vercel database env shows as
  saved-query data unavailable rather than an ambiguous empty visualization.
- The `elixir-analytics` Hermes gateway is running under local supervision and
  Slack Socket Mode is connected after a targeted restart of
  `ai.hermes.gateway-elixir-analytics`.
- Hermes Slack Socket Mode watchdog now emits
  `Socket Mode connected (recovered)` after a transport-drop self-heal, so
  log-based readiness can distinguish a recovered gateway from a stuck
  reconnecting one.
- Real Slack E2E passed for saved GTV and Supabase Swiggy ad hoc questions.
- The fresh Swiggy recheck used a direct `analytics.joinelixir.club/query?...`
  dashboard link, not a third-party shortener.
- Real Slack E2E also passed for active-user clarification, PostHog app-active
  users, and write-SQL rejection.
- Hermes profile tests and the full analytics test/build/lint suite pass
  locally.
- Hermes now supports profile-owned plugins from `profile_plugins/`, and the
  `elixir-analytics` distribution ships `elixir-analytics-runner`.
- The runner plugin exposes `elixir_analytics_runner` for planner, saved-topic,
  common-question shortcut, Supabase ad hoc, and PostHog ad hoc calls without
  first using generic `execute_code` or ad hoc shell composition.
- The runner plugin also exposes deterministic maintenance modes:
  `source_change_plan` for definition/glossary/schema/dashboard change
  planning, and `self_improvement_plan` for query-log review/promotion
  planning.
- Live profile checks passed for both maintenance modes: `source_change_plan`
  returned a GTV `metric_definition` PR plan with `prRequired: true`, and
  `self_improvement_plan` reviewed 5 query-log entries with 4 suggestions.
- Local dry-runs passed through the plugin for planner, saved GTV, Swiggy-style
  Supabase ad hoc, and app-active PostHog ad hoc routes.
- The installed `elixir-analytics` profile has been synced with the runner
  plugin, and the launchd-managed Slack gateway was restarted successfully.
- A live read-only saved-topic call through `elixir_analytics_runner` returned
  5 rows for `card-gtv-weekly` and the direct production dashboard link in
  20.65 seconds.
- The distributed skill and tool schema now instruct Hermes to use
  `answer_question` first for plain Slack analytics questions, with compact row
  caps for user lists unless the user asks for exhaustive output.
- Direct live shortcut checks passed outside Slack: Swiggy users this week
  returned 15 rows through Supabase in 0.56 seconds, and app-active users this
  week returned 1 KPI row through PostHog in 1.23 seconds.
- The analytics shortcut runner now also detects `show GTV last 30 days by
  week` and routes it directly to saved topic `card-gtv-weekly`; live Hermes
  wrapper dry-run confirms `answer_question` returns the saved-topic dashboard
  path `/query?topic=card-gtv-weekly&range=30d`.
- A live `run-analytics-question.ts` shortcut query for `show GTV last 30 days
  by week` returned 5 weekly rows and the production dashboard link in 20.5
  seconds.
- The live profile plugin was resynced and the gateway restarted after the
  `answer_question` guidance and compact-row default were added. Live plugin
  dry-run confirms `answer_question` defaults to `maxRows: 100`.
- The skill now has a top-level mandatory-first-call rule: plain Slack data
  questions must call `elixir_analytics_runner` mode `answer_question` before
  planning, file inspection, SQL writing, `execute_code`, or source edits.
- The runner plugin defensively treats a question-only call as
  `answer_question`; live profile dry-run confirms `show GTV last 30 days by
  week` routes to `card_gtv_weekly_30d` even when mode is omitted.
- The runner plugin now emits safe telemetry through
  `hermes.elixir_analytics_runner`: mode, route, shortcut, topic/result type,
  row count, truncation, dashboard presence, elapsed time, and error type. It
  does not log raw rows, SQL, or the raw user question.
- The analytics repo now includes `scripts/check-slack-e2e-logs.ts`, which
  reads the live Hermes gateway and agent logs after Slack prompts and reports
  route, timing, API-call, and safe telemetry status for the acceptance suite.
- Running the log checker on the current pre-telemetry Slack logs correctly
  fails saved GTV, Swiggy, and PostHog because runner telemetry is absent and
  the older Swiggy/PostHog runs exceeded latency/call-count targets; the
  clarification and write-SQL rejection scenarios pass.
- The analytics smoke suite now includes `source_change_workflow`, proving that
  a request like `GTV definition is wrong...` routes to source-of-truth PR
  planning with glossary, metric-contract, saved-topic, and test-file evidence.
- Full verification after the ops, query-handoff, and Slack recovery-log
  update: Hermes profile/plugin tests passed with 91 tests, Slack reconnect
  tests passed with 223 tests, `npm run lint` completed with no warnings,
  `npm test` passed 185 tests, and `npm run build` passed.
- Temporary no-persistence `/query?payload=...` handoff is now covered by a
  focused resolver regression test: payload links decode rows and metadata
  without executing a saved topic.
- The common-question shortcut runner now returns `payload.slackText` for saved
  GTV, Swiggy Supabase, and app-active PostHog answers. The Hermes tool schema
  and skill both tell the model to use that text as the Slack-facing answer
  before extra summarization or maintenance work.
- Temporary visualization payloads now strip both Supabase SQL and PostHog
  HogQL before encoding `/query?payload=...`, reducing Slack link bulk and
  keeping executable query text out of no-persistence dashboard URLs.
- The analytics repo now has `scripts/check-release-packaging.ts` and
  `docs/release-packaging.md`. The current dirty analytics worktree classifies
  cleanly into `analytics-runtime`, `executive-dashboard`, and `release-docs`
  with no unclassified files, so runtime and dashboard changes can be packaged
  separately instead of merged as one broad release.
- The Hermes repo now has `scripts/check_elixir_analytics_release_packaging.py`
  and `profile-distributions/elixir-analytics/RELEASE_PACKAGING.md`. The
  current dirty Hermes worktree classifies cleanly into
  `profile-distribution`, `hermes-runtime`, and `slack-gateway`, with
  `.hermes-bootstrap-complete` reported as a local artifact rather than a file
  to stage.

Known gaps:

- Swiggy direct-link recheck was too slow: 418.3 seconds across 33 model calls.
- The slow Swiggy run also revealed source-edit drift: the agent attempted
  maintenance/source edits during a plain data answer. The profile skill now
  says to answer first and defer logging/source changes.
- PostHog app-active live E2E was functionally correct but too slow: 266.6
  seconds across 20 model calls, with generic `execute_code` used before the
  final answer.
- The gateway is locally supervised, not yet moved to a hosted always-on setup.
- Hermes profile distribution changes still need an upstream push or PR path.
- Production dashboard links need a fresh deploy with
  `ANALYTICS_DATABASE_URL`; the user-observed no-data GTV link is consistent
  with the current live ops-readiness blocker.
- The live Slack gateway still needs Swiggy and PostHog prompts rechecked after
  the `answer_question` fast-path sync to prove the model chooses the shortcut
  tool and meets the latency target.
- A local Hermes one-shot answered Swiggy correctly after the sync, but still
  took about 64 seconds end-to-end and produced a verbose direct payload URL.
  If Slack shows the same behavior, the next polish step is a stronger
  Slack-specific answer formatter or payload compaction.

## Milestones

| Phase | Milestone | Product outcome | Status |
|---|---|---|---|
| 1 | Profile foundation | Slack `macros` uses the isolated analytics profile. | Done |
| 2 | Saved query vertical slice | Known questions such as weekly GTV use saved topics. | Done |
| 3 | Supabase ad hoc runner | Arbitrary business analytics questions use a JSON runner. | Done; fast-path polish pending |
| 4 | Temporary visualization handoff | Arbitrary results can link to a no-persistence visual. | Direct-link passed locally/live once; production env recheck pending |
| 5 | PostHog runner | App/product analytics route to read-only HogQL. | Passed live; fast-path polish pending |
| 6 | Source-of-truth workflow | Definitions, glossary, topics, and dashboard changes become PR work. | Built and smoke-covered; first live change request pending |
| 7 | Self-improvement loop | Repeated questions become promotion candidates. | Built, runner-accessible, and cadence-checkable |
| 8 | Ops readiness | Production blockers are reported before rollout claims. | Checker done; live status blocked |
| 9 | Slack smoke suite | Core Slack scenarios can be dry-run verified. | Done |
| 10 | Production deploy | Analytics branch is merged, deployed, and env-backed. | Blocked on main deploy + `ANALYTICS_DATABASE_URL` |
| 11 | Slack end-to-end validation | Real Slack prompts prove saved, ad hoc, PostHog, clarification, and rejection flows. | Functional history exists; fresh post-fix pass pending |
| 12 | Hosted gateway | The Slack gateway is restart-safe beyond the local machine. | Local supervised; host decision pending |
| 13 | Hermes upstream sync | The profile distribution is pushed or PR'd upstream. | Pending GitHub permission |
| 14 | Answer polish | Slack responses consistently include rows, assumptions, caveats, timings, and links. | `payload.slackText` built/tested; live Slack latency recheck pending |
| 15 | Operating cadence | Self-improvement review runs on a regular query-log cadence. | Checker built; scheduled/live usage pending |

## Production Gates

Do not call the agent or dashboard production-ready until all gates pass:

1. Analytics changes are merged and deployed to the production Vercel project.
2. Deployed Next.js env includes read-only `ANALYTICS_DATABASE_URL`.
3. Temporary visuals use direct `/query?payload=...` or saved topic links with
   no durable storage and no third-party URL shorteners.
4. Hermes `elixir-analytics` profile has an inference provider key, managed
   provider, or verified OAuth provider auth.
5. Slack `macros` gateway is connected and supervised.
6. Smart approvals are enabled.
7. Generic Hermes tools remain enabled for debugging and source changes.
8. The Slack smoke suite passes.
9. A real Slack E2E pass covers saved topic, Supabase ad hoc direct-link,
   PostHog, clarification, and write-SQL rejection.

## Packaging Plan

Keep the current dirty work split by product lane:

1. Hermes profile package: profile-owned plugin discovery, distribution sync,
   `elixir-analytics-runner`, toolset exposure, skill guidance, and Slack
   recovery logging.
2. Analytics runtime package: deterministic question runner, Supabase/PostHog
   runner hardening, ops-readiness checker, Slack E2E log checker, smoke suite,
   query-page data loader, temporary payload handoff, and related tests/docs.
3. Executive dashboard package: visual/theme/dashboard layout work in the
   Next.js app. Package separately unless the user explicitly wants it merged
   with the runtime lane.
4. Release docs package: packaging, ops, Slack smoke, source-change, and ad hoc
   operating docs plus the release classifier.

Do not stage or PR the entire analytics worktree as one unit until the dashboard
lane is intentionally accepted into the same release.

From `/Users/ritik/Coding/claude-analytics`, run:

```bash
node --import tsx scripts/check-release-packaging.ts --strict
```

Strict mode must have zero unclassified files before staging package-specific
commits.

From `/Users/ritik/.hermes/hermes-agent`, run:

```bash
venv/bin/python scripts/check_elixir_analytics_release_packaging.py --strict
```

The Hermes checker separates profile distribution, runtime/plugin loading, and
Slack gateway reliability changes; local artifacts must remain unstaged.

## Standard Verification

Run these from `/Users/ritik/Coding/claude-analytics` after analytics changes:

```bash
npm run lint
npm test
npm run build
node --import tsx scripts/run-analytics-smoke-suite.ts --query-log QUERY_LOG.md --current-branch '<branch>' --env-file /Users/ritik/.hermes/profiles/elixir-analytics/.env --provider-authenticated openai-codex --smart-approvals --generic-tools
node --import tsx scripts/check-ops-readiness.ts --current-branch '<branch>' --profile-home /Users/ritik/.hermes/profiles/elixir-analytics --provider-authenticated openai-codex --smart-approvals --generic-tools
```

Run this from the Hermes repo after profile changes:

```bash
scripts/run_tests.sh tests/hermes_cli/test_elixir_analytics_profile_distribution.py -- -q
```

## Slack E2E Checklist

Verify these exact prompts in Slack:

| Prompt | Expected route | Status |
|---|---|---|
| `show GTV last 30 days by week` | saved topic `card-gtv-weekly` with non-empty dashboard data and dashboard link | Slack answer passed historically; dashboard link needs fresh verification after Vercel env fix |
| `which users spent on Swiggy this week?` | Supabase ad hoc runner with India week-to-date assumption and temporary dashboard link | Passed live with direct link; latency failed at 418.3s / 33 calls |
| `active users this week` | clarification before querying | Passed live at 13.6s / 3 calls |
| `how many app active users this week?` | PostHog ad hoc runner with `active_app_user` | Passed live; latency failed at 266.6s / 20 calls |
| `delete from profiles` | read-only validation rejection | Passed live at 40.2s / 2 calls |

## Next Execution Milestone

Milestone 11A: finish real Slack E2E acceptance. Partially complete on
2026-06-04; production dashboard-link evidence must be refreshed after the
Vercel env fix.

Done means all five Slack checklist prompts have fresh evidence, dashboard links
open with data where expected, and gateway state remains connected afterward.

Execution order:

1. Re-run `which users spent on Swiggy this week?` and verify a direct
   `analytics.joinelixir.club/query?...` link, not TinyURL or another shortener.
   Direct link passed on 2026-06-04; rerun after fast-path skill fix should
   avoid source edits and reduce latency.
2. Run `active users this week` and verify the agent asks clarification before
   querying.
3. Run `how many app active users this week?` and verify the PostHog runner path.
4. Run `delete from profiles` and verify read-only validation rejects it.
5. Record timings, route decisions, dashboard links, and any approval prompts.

Milestone 14A: make Supabase ad hoc answers fast enough for Slack.

Done means a Swiggy-style Supabase ad hoc prompt answers through planner ->
`elixir_analytics_runner` -> Slack response without source edits, without
third-party shorteners, and without exceeding the agreed latency target.
The deterministic shortcut now returns `payload.slackText`; fresh Slack evidence
must prove Hermes sends it without re-planning or source edits.

Milestone 14B: make PostHog ad hoc answers fast enough for Slack.

Done means `how many app active users this week?` answers through planner ->
`elixir_analytics_runner` PostHog mode -> Slack response without generic
`execute_code`, without source edits, and without exceeding the agreed latency
target.
The deterministic shortcut now returns `payload.slackText`; fresh Slack evidence
must prove Hermes sends it without generic `execute_code`.

Milestone 14C: prove the shortcut path from Slack.

Done means the live Slack gateway handles Swiggy and app-active prompts through
`elixir_analytics_runner` mode `answer_question`, not generic `execute_code`,
and the gateway log shows materially lower call counts and response times than
the earlier 418.3s and 266.6s runs.

Milestone 14D: make Slack route evidence inspectable.

Done means `elixir_analytics_runner` emits safe route/timing telemetry for each
runner call without logging raw rows, SQL, or raw questions, so Slack acceptance
can be verified from logs even when final answer text is compact.

## Current Open Decisions

These are not solved by local code alone:

- Choose where the always-on Slack gateway should run after local supervision.
- Confirm the Hermes upstream sync path: push branch, open PR, or keep private.
- Decide when to commit/push the local Hermes profile and analytics docs patches.
- Decide the acceptable latency target for Supabase ad hoc Slack answers.

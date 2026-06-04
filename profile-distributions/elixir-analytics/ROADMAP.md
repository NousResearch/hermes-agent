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
- Current live ops-readiness status from analytics `main` is `ready` when using
  the ignored production env pull file `.env.production.local`. Vercel confirms
  `ANALYTICS_DATABASE_URL` exists for Development, Preview, and Production.
- Analytics PR #6, `Reconcile Slack E2E runtime polish`, is merged to `main` at
  `4f4ddd6`. It restored the session-aware Slack E2E checker and added the
  deterministic `top_card_spenders_30d` shortcut on top of current production
  code.
- Analytics PR #7, `[codex] Compact Slack dashboard links`, is merged to
  `main` at `4d529dc`. It formats saved-topic dashboard links as Slack mrkdwn
  `Open dashboard` links and temporary ad hoc payload links as `Open
  visualization` links while preserving relative `dashboardUrlPath` fallback
  behavior.
- Latest local verification is green: `npm run lint` has no warnings,
  `npm test` passed 221 analytics tests, `npm run build` passed, strict release
  packaging passed, the smoke suite passed from `main`, and the live Slack log
  checker passed from `main`.
- Latest analytics PR #7 verification is green: focused formatter coverage
  passed 11 tests, `npm test` passed 221 tests, `npm run lint` passed,
  `node ./node_modules/next/dist/bin/next build` passed, strict release
  packaging passed, the analytics smoke suite passed, and production-env ops
  readiness returned `overallStatus: "ready"` with no blockers.
- Post-merge verification on analytics `main` passed focused formatter coverage
  with 11 tests and production-env ops readiness returned
  `overallStatus: "ready"` with no blockers.
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
- The current live log checker passes saved-topic, Supabase ad hoc, PostHog,
  clarification, and write-SQL rejection scenarios with safe route telemetry.
- The analytics smoke suite now includes `source_change_workflow`, proving that
  a request like `GTV definition is wrong...` routes to source-of-truth PR
  planning with glossary, metric-contract, saved-topic, and test-file evidence.
- The analytics smoke suite also keeps `row_cap_truncation`, proving large ad
  hoc result sets are bounded and marked `truncated: true`.
- Full analytics verification after PR #6: `npm run lint` completed with no
  warnings, `npm test` passed 221 tests, strict release packaging passed, the
  smoke suite passed from `main`, and `npm run build` passed. Earlier Hermes
  profile/plugin tests passed with 91 tests and Slack reconnect tests passed
  with 223 tests.
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
  `docs/release-packaging.md`. Strict release packaging passes on clean `main`,
  so runtime and dashboard changes can be packaged separately instead of merged
  as one broad release.
- The Hermes repo now has `scripts/check_elixir_analytics_release_packaging.py`
  and `profile-distributions/elixir-analytics/RELEASE_PACKAGING.md`. The
  current dirty Hermes worktree classifies cleanly into
  `profile-distribution`, `hermes-runtime`, and `slack-gateway`, with
  `.hermes-bootstrap-complete` reported as a local artifact rather than a file
  to stage.
- Post-merge verification on analytics `main` passed the live Slack log checker:
  saved weekly GTV, last-7-days GTV, Swiggy Supabase ad hoc, top card spenders,
  PostHog app-active users, active-user clarification, and write-SQL rejection
  all used the intended routes.
- Production dashboard link check confirms
  `https://analytics.joinelixir.club/query?topic=card-gtv-weekly&range=30d`
  redirects signed-out users to
  `/sign-in?callbackUrl=%2Fquery%3Ftopic%3Dcard-gtv-weekly%26range%3D30d` with
  no console errors. The controlled Chrome profile was not signed in, so
  post-sign-in data rendering still needs either user-authenticated browser
  verification or a manual user check.
- Vercel production deployment
  `https://analytics-agent-9y28kucl2-ritikmdns-projects.vercel.app` is Ready
  and aliased to `https://analytics.joinelixir.club`.

Known gaps:

- Signed-in production dashboard rendering needs one fresh user-authenticated
  browser check. Signed-out callback preservation is verified.
- The gateway is supervised and Socket Mode connected locally, but the product
  still needs a hosted always-on gateway decision before calling Slack analytics
  fully production-ready.
- Hermes profile distribution branch `codex/elixir-analytics-profile` is pushed
  to `ritikmdn/hermes-agent`, dry-merges cleanly into current
  `NousResearch/hermes-agent` `main`, and focused Hermes verification passes
  253 tests. Opening a public upstream PR remains a product/privacy decision
  because the profile is Elixir-specific.
- Slack answer polish is merged and production-deployed. It still needs one
  live Slack recheck before Milestone 14A is closed.

## Milestones

| Phase | Milestone | Product outcome | Status |
|---|---|---|---|
| 1 | Profile foundation | Slack `macros` uses the isolated analytics profile. | Done |
| 2 | Saved query vertical slice | Known questions such as weekly GTV use saved topics. | Done |
| 3 | Supabase ad hoc runner | Arbitrary business analytics questions use a JSON runner. | Done; live Swiggy and top-spenders paths pass |
| 4 | Temporary visualization handoff | Arbitrary results can link to a no-persistence visual. | Done for saved and payload links; signed-in browser proof pending |
| 5 | PostHog runner | App/product analytics route to read-only HogQL. | Done; live app-active route passes |
| 6 | Source-of-truth workflow | Definitions, glossary, topics, and dashboard changes become PR work. | Built and smoke-covered; first live change request pending |
| 7 | Self-improvement loop | Repeated questions become promotion candidates. | Built, runner-accessible, and cadence-checkable |
| 8 | Ops readiness | Production blockers are reported before rollout claims. | Ready when run with production env pull |
| 9 | Slack smoke suite | Core Slack scenarios can be dry-run verified. | Done |
| 10 | Production deploy | Analytics branch is merged, deployed, and env-backed. | Main merged, Vercel production Ready, readiness ready with production env pull; signed-in browser proof pending |
| 11 | Slack end-to-end validation | Real Slack prompts prove saved, ad hoc, PostHog, clarification, and rejection flows. | Passed via live logs after PR #6 |
| 12 | Hosted gateway | The Slack gateway is restart-safe beyond the local machine. | Local supervised; host decision pending |
| 13 | Hermes upstream sync | The profile distribution is pushed or PR'd upstream. | Branch pushed/tested; public upstream PR decision pending |
| 14 | Answer polish | Slack responses consistently include rows, assumptions, caveats, timings, and links. | Compact-link PR #7 merged and deployed; live Slack recheck pending |
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
| `show GTV last 30 days by week` | saved topic `card-gtv-weekly` with non-empty dashboard data and dashboard link | Passed live via `answer_question`, 2 API calls, dashboard link present |
| `what was GTV for last 7 days?` | saved topic `card-gtv` with `7d` range | Passed live via `answer_question`, 2 API calls, dashboard link present |
| `which users spent on Swiggy this week?` | Supabase ad hoc runner with India week-to-date assumption and temporary dashboard link | Passed live via `swiggy_users_this_week`, 2 API calls, 15 rows |
| `who spent the most in last 30 days?` | Supabase ad hoc top-spenders shortcut with bounded user rows | Passed live via `top_card_spenders_30d`, 2 API calls, 25 rows, truncated |
| `active users this week` | clarification before querying | Passed live; clarification flow intentionally waits for user input |
| `how many app active users this week?` | PostHog ad hoc runner with `active_app_user` | Passed live via `app_active_users_this_week`, 2 API calls, 1 KPI row |
| `delete from profiles` | read-only validation rejection | Passed live without runner execution |

## Next Execution Milestone

Milestone 10A: finish production dashboard signed-in proof.

Done means the production saved-topic link opens with non-empty data after
sign-in from an authenticated browser session. Env/readiness reconciliation is
already complete.

Execution order:

1. Use an authenticated browser profile or ask the user to sign in through the
   controlled Chrome profile.
2. Recheck `https://analytics.joinelixir.club/query?topic=card-gtv-weekly&range=30d`
   and confirm rows render.
3. Record whether the page shows `Weekly GTV`, row count/table data, and zero
   console errors.

Milestone 12A: choose and implement hosted gateway.

Done means Slack `macros` does not depend on the laptop: the gateway has an
always-on host, restart policy, health check, log access, and documented rollback
path.

Use `HOSTED_GATEWAY.md` as the cutover runbook. The remaining product decision
is the host itself; the required process shape, secrets, health checks, smoke
prompts, rollback steps, Docker Compose template, and systemd unit are
documented there.

Milestone 13A: choose Hermes upstream/private sync path.

Done means the `elixir-analytics` profile distribution, runner plugin, and Slack
gateway patches are either pushed/PR'd upstream or intentionally documented as a
private profile distribution.

Current evidence: branch `codex/elixir-analytics-profile` is pushed to the
`ritikmdn/hermes-agent` fork, dry-merges cleanly into current upstream `main`,
and focused verification passed 253 Hermes tests plus strict release packaging.

Milestone 14A: compact Slack answer links.

Done means user-list answers still include dashboard links, but the Slack message
stays compact enough for executive reading and does not expose executable query
text in the URL.

Current evidence: analytics PR #7 is merged to `main` at `4d529dc` and deployed
to production at
`https://analytics-agent-9y28kucl2-ritikmdns-projects.vercel.app`, aliased by
`https://analytics.joinelixir.club`. Saved dashboard URLs format as `Open
dashboard` and temporary payload URLs format as `Open visualization` in
Slack-facing text. Verification passed focused formatter tests, the full
221-test analytics suite, lint, Next.js build, strict release packaging, smoke
suite, production-env ops readiness, and Vercel production inspection.

Remaining work: run one live Slack answer check to confirm Slack renders the
compact labels.

## Current Open Decisions

These are not solved by local code alone:

- Choose where the always-on Slack gateway should run after local supervision.
- Confirm the Hermes upstream sync path: push branch, open PR, or keep private.
- Decide the acceptable latency target for Supabase ad hoc Slack answers.

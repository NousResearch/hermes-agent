# Educator profile package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship everything an operator needs to stand up the Product Education
teammate ("educator") as a secondary multiplexed profile: persona, a validated
config template, env checklist, the two wake-gate cron scripts, the cron job
definitions, the Slack app manifest, the Ideas database schema and the rollout
checklist. No deploy, no VM access, no Notion/Slack/Zendesk writes.

**Spec:** `docs/superpowers/specs/2026-11-16-product-education-teammate-design.md`
("The profile", "How it works", "Rollout"), plus the CLAUDE.md entries for the
merged dependencies (#4 multiplex, #5 cost budget, #6 github_pr, #7 zendesk,
#8 boundary enforcement).

## Where it lives

`deploy/educator/`. The repo has a profile-distribution format
(`hermes_cli/profile_distribution.py`, `distribution.yaml`), but it ships a
literal `config.yaml` and does not own `scripts/`; this profile needs
per-install ids substituted into the config and scripts placed under
`<profile>/scripts/` (the only place cron runs scripts from). A plain folder
with a template + rollout checklist fits; nothing in core changes.

```
deploy/educator/
  README.md                 index
  SOUL.md                   persona ({{PERSONA_NAME}})
  config.yaml.template      complete profile config ({{...}} placeholders)
  .env.example              env var names only
  scripts/educator_wake.py           shared wake-gate library (stdlib HTTP)
  scripts/educator_release_watch.py  release-watch pre-run script
  scripts/educator_explore.py        explore pre-run script
  cron-jobs.yaml            both jobs: schedule, prompt, toolsets, flags
  apply_cron_jobs.py        admin applies cron-jobs.yaml (fields the CLI lacks)
  slack-app-manifest.yaml
  notion-ideas-database.md
  workspace/notebook.md     seed notebook (fixed sections)
  ROLLOUT.md
tests/deploy/
  test_educator_config_template.py
  test_educator_wake_scripts.py
  test_educator_package.py  (manifest scopes, job file, toolset names)
```

## Facts found while reading (they shape the files)

- `cron/scheduler.py::_parse_wake_gate`: last non-empty stdout line JSON with
  `"wakeAgent": false` skips the agent; anything else wakes it, and the whole
  stdout is injected as `## Script Output`. Empty stdout also skips. So the
  scripts print ONE compact JSON line: `{"wakeAgent": false}` or
  `{"wakeAgent": true, "job": …, "events": […], …}`.
- Cron scripts must live in `<HERMES_HOME>/scripts/`, run under the gateway's
  `sys.executable`, cwd = job `workdir`, env = profile-scoped (#4 4f) then
  `_sanitize_subprocess_env`, which strips `SLACK_*` tokens but passes
  `NOTION_API_KEY`, `JIRA_*`, `GITHUB_PR_TOKEN`, `ZENDESK_*`.
- **Consequence:** `python -m tools.zendesk_tool run-due` /
  `tools.release_update_post run-due` cannot work from a pre-run script —
  both need `SLACK_BOT_TOKEN` (veto ❌ reactions, the post itself). The
  script instead reads `agent.autonomy.due_pending_ids` (a local file read)
  and wakes the agent with `veto_due`, and the prompt tells it to call
  `zendesk_create_draft(run_due=true)` / `slack_post_thread(run_due=true)`.
  In v1 both actions are `signoff`, so nothing is ever pending.
- The pre-run script is not told its job id; it finds its job in
  `cron.jobs.list_jobs()` by script filename.
- `cron.executions` ledger: the running execution already exists when the
  script runs; previous rows give `claimed_at`/`started_at`/`finished_at`/
  `status`. The notebook check compares notebook mtime with the start of the
  last run **that woke the agent** (remembered in the script's state file).
- `hermes cron create` has no `--enabled-toolsets`, `--discretionary` or
  `required_toolsets`; `cron.jobs.create_job` + `update_job` do →
  `apply_cron_jobs.py`.
- `notion.write_guard` ids must be bare UUIDs (`collection://` is rejected);
  it requires `agent_write_guard: true`.
- `release_updates.asset_type_value` must be `Slack post` — the real Release
  Assets option (the code default is `Slack update`).
- The Releases data source has **no `PRD` relation**; 5b and the guard need
  one (`release_kits.prd_relation_property`). Rollout adds it (one-way).
- PRD `JIRA` is a single **URL** property; keys are regex-extracted. Jira
  statuses are `Ready for staging` (lower-case s), `In Staging`, `Done` —
  compared case-insensitively.
- Shadow week (`#release-updates` out of `post_allowed_channels` but still
  gated) makes `boundary_config_errors` report one expected error line.

## Tasks

- [ ] **1. Wake library + release-watch script (TDD).** Tests first with a fake
  HTTP layer (`educator_wake.http_json` indirection): silent run → `{"wakeAgent":
  false}`; kit_start when a Not-started PRD's ticket is in Ready for staging /
  In Staging / Done; ticket_status change; all_done once; asset_approved /
  approved-row edited by a human (not the bot); veto_due from pending records;
  notebook_not_updated from the ledger; previous woken run failed → re-emit its
  events; git refresh failure reported once per signature; Jira 400 falls back
  to per-key reads; never writes to Notion/Jira (only GET + the read-only
  `…/query` POST).
- [ ] **2. Explore script (TDD).** Always wakes with notebook/previous-run status.
- [ ] **3. Config template + validation test.** Substitute dummy values; assert
  `boundary_config_errors == []`, write guard enforcing with no error,
  `release_update_post.settings().config_error == ""`, cost budget loads with no
  warnings and the release-watch id is essential, github_pr policy parses,
  autonomy levels as designed, RBAC roles, `platform_toolsets.slack` equals
  the design set and every name is a registered toolset, no forbidden toolset
  anywhere; shadow variant yields exactly the expected gated-not-allowed error.
- [ ] **4. Remaining files:** SOUL.md, .env.example, cron-jobs.yaml +
  apply_cron_jobs.py, manifest, Ideas schema, seed notebook, README; a package
  test (manifest ⊇ generator scopes + reactions:write + users:read.email, jobs
  file shape, script names match).
- [ ] **5. ROLLOUT.md.**
- [ ] **6. Verify** `scripts/run_tests.sh tests/deploy/`, commit, push, PR,
  report to the starting session.

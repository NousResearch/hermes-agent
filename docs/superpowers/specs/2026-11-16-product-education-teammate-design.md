# Product Education teammate — design

**Date:** 2026-11-16
**Status:** Approved 2026-11-16
**Depends on:**

- [multiplex fork hardening](2026-11-16-multiplex-fork-hardening-design.md) (step 1, approved);
- a draft-only Zendesk tool;
- cost budget + alerts;
- a single-repo GitHub PR tool.

The supporting pieces are specified below under "Dependencies"; those
sections are their approved design (no separate docs).

## Intent

EverAfter wants an AI **Product Education lead**: a teammate, not a
pipeline. It owns product education and works from its own initiative:

- It prepares the release comms for everything shipping.
- It learns from every review, so its drafts need fewer edits over time.
- It researches best practices, audits our knowledge base, notices what we
  miss or get wrong, and proposes new ideas, including new jobs for itself.

Release comms is its **first** job, not its whole remit.

It is a separate bot with its own Slack app, persona, tools, permissions,
credentials and working folder. It runs as a secondary Hermes profile in the
**same gateway** as the current bot (`gateway.multiplex_profiles: true`),
fully separated from the current bot (step 1).

**Success, in the user's words:**

- fewer edits to its drafts and better content overall;
- fewer support tickets, later, once it has ticket data;
- a proactive teammate that thinks of what else a product-education master
  would do.

## Goals

- Drafts the full release kit for each ticket that reaches "Ready for
  Staging": Release row, Brief and asset rows in the Notion Release Kits
  databases, produced with the team's `release-comms` skills.
- Learns from every approval:
  - compares its draft with the approved version;
  - records lessons;
  - tracks an **edit rate** per asset type;
  - feeds repeated lessons back into the shared skills as pull requests.
- Works proactively on its own agenda, a few times a day:
  - researching online;
  - auditing the Zendesk KB;
  - spotting gaps;
  - challenging how we do things;
  - proposing ideas on its Notion Ideas board.
- Keeps continuity between runs in a self-managed **notebook**.
- Spends within a budget, sees its own spend, and alerts on high cost.

## Non-goals (v1)

- Reading support tickets. The Zendesk tool is designed so ticket reads can
  be added later.
- Reading `support.base.ai`, the Base 1.0 help centre. It is HubSpot, sign-in
  only, with no API. Base 1.0 help articles are drafted in Notion from Jira
  and specs only.
- Writing to Jira. Sources are read-only per `release-comms`.
- In-app What's New cards beyond drafting them as Notion rows.

## Anti-goals

These would make the work a failure even if it technically works:

- **Anything reaches customers, or a company-wide channel, without
  product-team sign-off.** It may explore and suggest freely. It may not
  deploy.
- It acts with the current bot's identity, credentials or permissions, or
  the current bot sees its state. Step 1 owns this.
- It produces busywork: noise posts, ideas nobody can act on, or
  "nothing changed" messages.
- Its costs run away unnoticed.
- One bad lesson silently degrades every future draft. Skill changes go
  through review.

## The boundary: explore and suggest, never deploy

This is enforced by **tools and permissions, not only the prompt**:

| It may freely | It may only after sign-off | It may never |
|---|---|---|
| Read the web, Jira, Notion, the Zendesk KB, its own repos, channels it is in | Post a release update to `#release-updates` (the Slack-update asset row is `Approved` **and** the ticket is `Done`) | Publish anything in Zendesk, HubSpot or the e4e widget |
| Create and edit **Draft** rows in Release Kits; write its Ideas board | Create a Zendesk **unpublished** draft (the help-article row is `Approved`, per `zendesk-draft`) | Advance a Notion row past `Draft` (`In review` onward is human-only) |
| Post in its own channels; reply to DMs and mentions | | Merge a PR; write to Jira; post outside its allowlisted channels |
| Open PRs on `product-skills` | | Use `terminal` or `code_execution` |

"Sign-off" in v1 is the Notion row's `Status`, set by a product-team member.
Any later widening of its powers is a config change. See **Autonomy levels**
below.

### Autonomy levels: how it graduates

The write guards are not permanent walls. They read a per-action
**autonomy level** from the teammate's config. "It's good enough, let it
deploy this on its own" is then a **one-line config change by an admin**,
picked up on the next call (the config is mtime-cached), with no code
change and no deploy. The teammate cannot change its own level, because
its `config.yaml` is write-protected by fix 4e.

```yaml
autonomy:
  paused: false                 # kill switch: true drops every action to `signoff`
  actions:
    post_release_update:  signoff   # #release-updates post
    zendesk_create_draft: signoff
    zendesk_publish:      off       # not built in v1
    advance_asset_status: off       # it may never approve its own drafts in v1
    ideas_accept_own:     off
    merge_skill_pr:       off       # not built in v1
```

| Level | Meaning |
|---|---|
| `off` | It cannot do it at all. The tool refuses. |
| `signoff` | A human approval signal (the row's `Status = Approved`) is required first. v1 default. |
| `veto` | It announces the action in the home channel and does it after a waiting period (e.g. 4 working hours), unless a product-team member vetoes: a ❌ reaction, or setting the row to a `Hold` status. A middle step between signoff and full trust. |
| `auto` | It acts immediately and reports after. |

**Rules:**

- **Per action, and optionally per asset type.** For example,
  `post_release_update: veto` while help articles stay `signoff`. You can
  graduate internal Slack updates long before customer-facing articles.
- **Levels only unlock capabilities that exist.** A level above `signoff`
  for an action whose code path is not built is rejected at config load,
  with a clear error. v1 builds `off` / `signoff` everywhere, plus `veto`
  and `auto` for `post_release_update` and `zendesk_create_draft`.
  Publishing to Zendesk, approving its own rows and merging PRs are built
  **when you first decide to graduate them**. Each is its own small change,
  which keeps v1's attack surface small.
- **Evidence for graduation.** The edit-rate log gives a trust record per
  action and asset type: approvals with no edits, edit rate, assumptions
  corrected. When an action meets a bar (default: the last 10 approved with
  under 5% edits and no factual fixes), the teammate **proposes its own
  graduation** on the Ideas board (Type: `autonomy`) with the evidence. A
  human decides and edits the config.
- **Every `veto` or `auto` action** is written to `data_access_audit` and
  summarised in the home channel. `autonomy.paused: true` is the instant
  rollback.
- **Demotion** is manual in v1. The teammate reports any action that a human
  reverted or complained about, and recommends demoting it.

**Every cell of this table is enforced in code (dependency 5).** The prompt
alone is not trusted to hold it. `notion_api` passes `PATCH` and `DELETE`
through unchecked (tools/notion_api_tool.py:26), so a profile-level Notion
write guard is required. Posting to `#release-updates` takes a **row id, not
text**. The tool re-checks the row's status and the Jira status itself, then
posts the row's own body.

**Prompt-injection guard.** It reads untrusted web pages and also holds
write tools. Its Slack posting is therefore limited to an **allowlist of
channel ids**: `slack.post_allowed_channels` in its profile config, enforced
inside `slack_post_thread` and on cron delivery targets. Its Notion
integration is shared only with Release Kits, Ideas, and its own parent
page. Its GitHub token covers the one repo. Read-and-act on a web page can
only land in places the team already reviews.

## Dependencies (separate designs)

### 1. Multiplex fork hardening

Approved. It provides two fully separate bots in one gateway.

**Added requirement from this design:** a cron job's pre/post `script`
subprocess must receive the **owning profile's** scoped environment, not the
process `os.environ`. The scripts below call Jira and Notion with the
teammate's credentials.

### 2. Zendesk tool (`zendesk` toolset)

- **Reads:** categories, sections, articles and translations on
  `everafterai.zendesk.com`.
- **Write:** create an **unpublished** draft article only. Edits to a
  published article stage in place only if Team Publishing allows it without
  going live; otherwise it creates a separate draft.
- **Never:** publish, archive, delete, change permissions, or touch tickets.
- **Guard:** a curated `(method, path)` allowlist, like `jira_api_write`.
- **Credentials:** `ZENDESK_SUBDOMAIN` / `ZENDESK_EMAIL` /
  `ZENDESK_API_TOKEN` via `get_secret`.
- **Later:** ticket reads as a **separate** toolset (`zendesk_tickets`).

### 3. Cost budget + alerts (general fork feature)

- **Config:** a per-profile `cost_budget:` block:
  - `daily_usd`, `monthly_usd`;
  - `alert_at` (e.g. `[0.5, 0.8, 1.0]`);
  - `per_run_usd`;
  - `alert_channel`;
  - `discretionary_jobs` (job ids or a tag).
- **Over budget:** discretionary jobs are skipped by **the scheduler
  itself**, before the job's script or agent runs. A discretionary job is
  listed in `discretionary_jobs` or carries a `discretionary: true` job
  field. The scheduler records `skipped: over budget` in the executions
  ledger. The gateway then sends **one** alert per threshold crossing per
  period, not one per skipped run, through the owning profile's adapter.
  Release work continues and raises a louder alert. (Designer's call: the
  scheduler, not a wake-gate script, because a script can't post and
  shouldn't need spend data.)
- **Spend source:** the same query layer as `hermes costs`
  (`agent/cost_attribution.py`), on the owning profile's session store,
  with a short cache so it isn't re-queried every tick.
- **Scope:** opt-in per profile. A profile with no `cost_budget:` block is
  unchanged.
- **Per run:** exceeding `per_run_usd` alerts.
- **Unpriced tokens count as an alert**, never as $0.
- **Self-visibility:** a read-only tool gives the agent its own spend and
  remaining budget.
- **Prerequisite:** the teammate's model must have a `pricing.overrides`
  entry.

### 4. GitHub tool (`github_pr` toolset)

- **Allowed actions on `everafterai/product-skills`:** create a branch,
  commit file changes, open a PR, and read PR comments and status.
- **`ea-monorepo`: read-only, approved by the user 2026-11-16.** It may read
  PRs (description, files, diff, review comments) and commit logs, so
  `release-intake` can follow "the PR is the spec" without a shell. No writes
  of any kind. The release-watch script keeps a read-only clone for file
  reads.
- **Repos:** configured per action, so the token's repo list and the tool's
  allowlist must agree.
- **Never:** merge, push to the default branch, or touch settings.
- **Credentials:** a fine-grained token, via `get_secret`:
  - `product-skills`: contents + pull requests, read/write;
  - `ea-monorepo`: contents + pull requests, read-only.
- **Why a tool and not MCP:** the default profile's GitHub MCP is kept away
  from the teammate (step 1, fix 4c).

### 5. Boundary enforcement (new, from review)

**a. Notion write guard (`notion.write_guard` in profile config).** Enforced
inside `notion_api` and mirrored in the `pre_tool_call` backstop, failing
closed. For a guarded profile:

- **No `DELETE`, and no `archived` / `in_trash: true`.**
- **Restricted `Status` values:**
  - On Release Kit rows, `Status` may be set only to `Draft`, and only on
    create or on a row currently `Draft`.
  - A row in any other status is read-only.
  - **Exception:** set `Published URL`, and `Status=Published`, on a
    Slack-update row **only** through 5b, after it posts.
- **Ideas rows:** create at `Proposed`; move its own rows `Accepted` →
  `In progress` → `Done`. Never set `Accepted` / `Rejected` / `Parked`.
- **PRD pages (Detailed Bucket Features):** read-only, except
  `Release Comms Status`, which it may set to `In progress` / `Done` only.
- **No schema changes:** no database or property create, update or delete.
- **Autonomy levels:** every rule here is subject to the levels in
  "Autonomy levels". A graduated action relaxes exactly its own rule.
- **Configuration:** the guard reads the database ids and status property
  names from config, so the rules are data, not code.

**b. Slack posting (extends `slack_post_thread`):**

- `thread_ts` becomes optional, so it can start a **new top-level post**.
  Today the tool requires it (slack_post_thread_tool.py:45), and cron
  disables `messaging`.
- Only channels in `slack.post_allowed_channels` are allowed.
- Channels in `slack.post_gated_channels` (`#release-updates`) accept **no
  free text**. The caller passes `release_asset_row_id`. The tool then:
  1. fetches the row;
  2. requires `Type = Slack update` and `Status = Approved`;
  3. requires every linked Jira key to be `Done`;
  4. posts the row's body verbatim;
  5. writes back `Published URL` + `Status=Published` (5a's exception).

**c. Self-modification guard.** This is added to step 1 as fix 4e. A guarded
profile's file tools cannot write its own `config.yaml`, `SOUL.md`, `cron/`,
`scripts/`, `.env`, or anything under `workspace/checkouts/`. External skill
dirs are read-only for **all** agent writes, including `skill_manage` and
`write_file` in Slack or cron turns, not only the curator
(skill_manager_tool.py:346-392). `HERMES_WRITE_SAFE_ROOT` is process-wide,
so it cannot be used. This is per-profile, keyed on the owning profile's
home.

## The profile

### Identity and persona

- **Profile:** `educator` (working name), at `~/.hermes/profiles/educator/`.
  Created **fresh**, not `--clone`d, so nothing of the current bot carries
  over.
- **`SOUL.md` persona:** a senior product-education lead. Opinionated about
  clarity, allergic to fluff. Writes in the house voices the skills define.
  Curious, and proposes rather than waits. Candid about uncertainty, and
  never invents product facts. Asks the product team specific questions in
  one batch.
- **Persona name and avatar:** the user's call (open question).

### Credentials (`educator/.env`, all its own)

| Service | Access |
|---|---|
| Slack | Its own Slack app's bot and app tokens |
| Notion | Its own integration, shared with Release Kits, the Ideas database, its home page, and **Detailed Bucket Features** (the PRDs) only |
| Jira | Read-only. A dedicated Atlassian account or API token, ideally a bot user with browse-only permissions on BA and EV |
| Zendesk | The shared service account |
| GitHub | The fine-grained token |
| Model | The LLM provider key, which may be the same provider as the current bot |

### Working folder

`~/.hermes/profiles/educator/workspace/`.

**Not via `terminal.cwd`.** That value is copied once into the
process-wide `TERMINAL_CWD` from the default profile (gateway/run.py
~2618), so under multiplex it is ignored for the educator. Instead:

- cron jobs set `workdir` to this folder;
- `SOUL.md` and its skills use **absolute paths** for everything in it;
- step 1 lists per-profile `TERMINAL_CWD` as a known limitation.

Contents:

- `notebook.md`: the self-managed notebook (below).
- `drafts/<asset-row-id>.md`: a snapshot of each asset as it first drafted
  it, for the edit diff.
- `metrics/edit-rate.csv`: one line per approved asset, giving type,
  characters changed / total, and lesson categories.
- `research/`: notes and sources from its own-initiative work.
- `checkouts/product-skills/`: a read-only clone of
  `everafterai/product-skills`, pulled by the release-watch script.
- `checkouts/ea-monorepo/`: a read-only clone, pulled by the same script,
  used only for product-surface identification as `release-intake`
  requires. Never used for customer content.

### Skills

- **Shared skills.** `skills.external_dirs` points at
  `checkouts/product-skills/plugins/release-comms/skills`, plus any other
  product-skills plugins the user wants it to use. Hermes treats external
  dirs as **read-only for autonomous maintenance**. It improves them only by
  PR.
- **Runtime adaptation.** The skills are written for Claude Code (Notion
  MCP, Atlassian MCP, `curl`, Chrome). A `references/hermes-runtime.md` in the shared repo (hosting Hermes-specific
  files in `product-skills` was approved by the user on 2026-11-16) maps
  each to Hermes:

  | Claude Code | Hermes |
  |---|---|
  | Notion MCP | `notion_api` |
  | Atlassian MCP | `jira_api` |
  | `curl` → Zendesk | `zendesk` |
  | Chrome | `browser` |

  It also covers the identity rules. For example, `Owner` cannot be the bot,
  so it is the ticket assignee or the PM named on the release.
- **Its own skills also live in `product-skills`**, as a new
  `product-education` plugin (`education-explore`, `notebook-keeping`,
  `learn-from-approval`), loaded through the same `external_dirs`. **It
  changes no skill directly; every skill change is a PR.**
  - **Why:** the "one bad lesson spreads" anti-goal.
  - **Why:** the ownership gate refuses no-identity cron edits of owned items
    (automation_ownership.py:344).
  - **Why:** in-place edits would dirty the checkout and break the script's
    `git pull`.
  - Lessons that are not yet a skill change live in the notebook.

### Tools (`platform_toolsets.slack` for this profile)

- **Hermes tools:** `web`, `browser`, `file`, `vision`, `session_search`,
  `memory`, `skills`, `todo`, `clarify`.
- **Fork tools:** `notion`, `jira`, `zendesk`, `github_pr`, `slack`,
  `slack_post`, `ownership`, `video_frames`, `pdf_pages`, and the
  budget-visibility tool.
- **Never:** `terminal`, `code_execution`, `jira_write`, `messaging` to
  arbitrary targets, or any MCP toolset.

### Who can talk to it (RBAC, its own `slack.user_roles`)

- **`educator_user`:** product-team members. Same toolsets as above. They
  can ask it anything and direct its work.
- **`admin`:** the operator(s).
- **Everyone else:** roleless and denied, by RBAC's deny-until-assigned rule.

### Slack setup

- **Home channel, `#product-education`** (name the user's call). Shared with
  the product team; posts are threaded.
  - Free-response, so the team can talk to it without an @mention.
  - With `wake_in_bot_authored_threads: true` for this profile, replies to
    its posts reach it.
  - Its "Kit ready", idea digests, questions and budget alerts land here.
- **Other channels it is invited to:** `require_mention: true`. It reads for
  context and speaks when mentioned.
- **DMs:** open to `educator_user`.
- **`post_allowed_channels`:** the home channel plus `#release-updates`.

## How it works

### What triggers release comms (decided)

**The PRD decides, not Jira.** PRDs live in the **Detailed Bucket Features**
Notion database (data source
`collection://2daff849-6e1b-4e4f-960a-a07acfbc2a1b`). `ship-prd` / `prd-grill`
fill these properties on every PRD:

| Property | Values | What it means for the teammate |
|---|---|---|
| `Release Comms Status` | `Not started` | This PRD gets its own Kit. |
| | `A part of a bigger module` | No Kit of its own. It rolls into the module's or epic's Kit, via `📚 Epics`. |
| | `Not needed` | Ignored. |
| `Release Comms Assets` | multi-select: Release note, Help article, What's New card, Slack post, CS FAQ, Marketing brief, In-app announcement, Customer email | **This list is the Kit's asset checklist.** The teammate drafts exactly these and does not re-route. |
| `Comm Owner` | a person | Becomes the Release row's `Owner` and the person it @-mentions for review. This solves "Owner can't be the bot". |
| `JIRA` | ticket keys | The tickets whose status sets the **timing**. |

**One Kit per PRD**, or per module for "part of a bigger module". It starts
when **any** linked ticket reaches "Ready for Staging". The release is
"deployed" when **all** linked tickets are `Done`.

**`Release Comms Status` on the PRD:** the teammate runs the comms, so it
sets `In progress` when it creates the Kit and `Done` when every asset row
is `Published`. That is the only PRD property the Notion guard lets it write.
`Status` and everything else stay off-limits, as `ship-prd`'s own rules
require.

**Draft first (decided).** It does not wait for answers. It drafts the whole
Kit immediately: the asset list comes from the PRD, so the main question the
skill would ask is already answered. Tier, target date and anything else it
had to guess go **in the Kit thread as marked assumptions**, and the PM
corrects them there. Rows whose content depends on a corrected assumption
are re-drafted while still `Draft`. **Edits made because an assumption was
wrong are tagged `assumption` in the edit-rate log**, so they don't count
against its writing.

### Job 1: `release-watch` (every 20 min, reactive)

**Pre-run script (no LLM).**

1. Pull both checkouts.
2. Query Notion Detailed Bucket Features for PRDs with
   `Release Comms Status` in (`Not started`, `In progress`).
3. Query Jira for those PRDs' `JIRA` keys, for status changes into
   "Ready for Staging", "In Staging" or "Done".
4. Query Notion `Release Assets` for rows edited or moved to `Approved`
   since the last run.
5. Diff against a state file in the workspace.
6. Emit the wake-gate payload: the list of changes, or nothing, in which
   case it stays silent and costs $0.

**Agent run, for each change:**

| Change | What it does |
|---|---|
| A Kit-eligible PRD's first linked ticket reaches "Ready for Staging" | Runs `release-comms` end to end for the PRD's `Release Comms Assets`. Creates the Release row (Owner = `Comm Owner`) and one asset row per asset at `Draft`. Snapshots each draft to `drafts/`. Sets the PRD's `Release Comms Status` → `In progress`. Posts **one** "Kit ready for review" thread in the home channel that @-mentions the Comm Owner, links the Release row, and lists its assumptions. |
| Ticket moves to "In Staging" | Re-checks the Kit against the latest ticket and spec. Updates `Draft` rows only, never `In review` / `Approved`, and notes what changed in the thread. Captures screenshots from staging when possible (`feature-screenshots`). |
| Asset row moved to `Approved` | Runs **learn-from-approval** (below). If it's a Zendesk help article, runs `zendesk-draft`. If it's a release note, runs `html-release-note`. |
| Slack-update row `Approved` **and** ticket `Done` | Posts it to `#release-updates`. Records the permalink in `Published URL`. |
| Ticket `Done` with Kit rows still `Draft` | One reminder in the Kit thread, not repeated. |

**Last steps:**

- Update the notebook.
- Release work is **not** discretionary: it runs over budget too, with an
  alert.

### Job 2: `explore` (3× per working day, proactive, discretionary)

**Pre-run script (wake gate).** Budget check: when the teammate is over its
daily or monthly discretionary budget, the run is skipped. A wake-gate
script **cannot post**: `wakeAgent=false` is silent, and the script's env has
`SLACK_*` stripped. So the over-budget alert is sent by the **cost-budget
feature itself** (dependency 3) from the gateway, through the owning
profile's adapter.

**Agent run.** It reads the notebook and the Ideas board, then **picks its
own agenda** for the session. Examples:

- research education best practice;
- audit Zendesk articles for gaps, staleness or renamed UI;
- find shipped features with no coverage;
- review edit-rate trends;
- draft a skill-improvement PR from repeated lessons;
- propose a new job for itself;
- follow up on ideas the team accepted.

**Output:**

- Ideas go on the Notion **Ideas** board.
- A home-channel post goes out **only** when something is worth the team's
  attention.
- Research notes go to `research/`.
- Otherwise the run ends `[SILENT]`.

**Last step:** update the notebook.

**Accepted ideas.** It works on the ones it can do within its boundary, for
example drafting an article for a coverage gap as a `Draft` asset row. For
the others, it asks.

### Continuity between runs

Every cron run starts a fresh session, so continuity has four sources, used
together:

0. **The Notion Release row of each Kit.** This is the source of truth for
   that Kit's state, questions and answers. It matters because
   `session_search` from a Slack turn is scoped to the channel or user
   (hermes_state.py ~1597). A PM replying in a Kit thread lands in a
   session that **cannot** search the cron run that drafted the Kit. The
   Kit thread's first message links the Release row, and the agent reads it.
1. **The notebook** (below). Its working state, what it is in the middle of.
2. **The Ideas board.** The team-visible record of what it proposed and what
   was decided, and why.
3. **`session_search`** over its own past runs and conversations. The full
   detail when the notebook's summary is not enough. It is scoped to this
   profile's store by step 1.
4. **Hermes `memory`.** Durable facts about the team, products and
   preferences.

### The notebook

- **What it is:** `workspace/notebook.md`, owned and maintained by the
  teammate. Every job reads it first and **rewrites** it last, compacting
  rather than appending.
- **Fixed sections:**
  - current focus;
  - open threads, and what it is waiting on and from whom;
  - recent findings and conclusions;
  - lessons learned (style, facts, process);
  - next-session intentions.
- **Size:** soft cap around 12 KB.
- **Enforcement:** cron has **no `post_script`**. Instead, each job's
  **pre-run** script compares the notebook's mtime with the previous run's
  finish time, taken from the cron executions ledger. When the last run
  didn't update it, the script adds `notebook_not_updated: <job>` to the wake
  payload and forces a wake. The agent then repairs the notebook and notes
  the miss in the home channel.

### Learning from approvals

When an asset row hits `Approved`:

1. Diff the approved body against `drafts/<row-id>.md`.
2. Classify the edits. Categories: voice, factual, structure, scope or wrong
   asset, screenshots, naming and glossary.
3. Append a line to `metrics/edit-rate.csv` and the lessons to the notebook.
4. **When a lesson recurs** (3+ times, or once if it is a factual or naming
   rule), propose a skill change. Shared skill or voice guide → a PR on
   `product-skills`, linked in the home channel. Its own skills → it may
   edit directly, and mentions it in the next digest.
5. **Weekly:** it posts the edit-rate trend per asset type in its home
   channel. This is the success metric for "fewer edits".

### Ideas board (Notion database, new)

| Property | Values |
|---|---|
| Title | — |
| Status | `Proposed` → `Accepted` / `Rejected` / `Parked` → `In progress` → `Done` |
| Type | content gap · stale content · process · new asset/format · research finding · self-improvement · new job |
| Rationale | — |
| Evidence | links |
| Effort | S / M / L |
| Owner | — |
| Decision note | the human's reason for accept or reject, which it reads and learns from |

The teammate creates rows at `Proposed` and may move its own accepted rows
to `In progress` / `Done`. Accept, Reject and Park are human-only.

### Budget

- It uses the cost-budget feature: `explore` is discretionary,
  `release-watch` is not.
- It sees its own spend through the budget tool. It plans its exploration
  accordingly, and says in the digest when budget limited it.
- **Model:** a capable writing model for both jobs. The pre-run scripts use
  no LLM. Per-job `model` overrides allow a cheaper model for audit-style
  explore sessions later.

## Rollout

1. Steps 1–4 merged. Multiplex is still **off** on the VM.
2. Create the Slack app (manifest in the repo), the Notion integration and
   its shares, the Ideas database, the Jira and Zendesk credentials, and the
   GitHub fine-grained token.
3. `hermes profile create educator`. Write `SOUL.md`, `config.yaml`
   (RBAC, Slack, toolsets, `external_dirs`, `cost_budget`,
   `pricing.overrides`) and `.env`. Clone the checkouts.
4. Turn on `gateway.multiplex_profiles: true` with `educator` on the
   allowlist, and restart. Run the step-1 manual separation checks.
5. Create both cron jobs as an admin, so the scripts are attached by a super
   admin per policy. Set the ownership registry owner, and set
   `enabled_toolsets` explicitly per job.
6. **Shadow week.** It drafts kits and explores, but `#release-updates`
   posting stays off (remove it from `post_allowed_channels`). Review the
   edit rate and the noise.
7. Enable posting.

## Setup values to collect at rollout

All design questions are answered. These are configuration values only.

- Persona name, avatar, and how it signs posts.
- Home channel name and the other channels it should be in.
- Who is on the product team (`educator_user`), and who are its admins.
- Daily and monthly budget amounts and the alert recipients.
- Which model.
- Staging access for screenshots: does a test account exist on
  `staging.base-ai.dev` it may use?

## Left to the builder

- The exact JQL and the release-watch state-file format.
- The notebook section headings' wording.
- The `SOUL.md` text, within the persona above.
- The Slack app manifest details.
- Whether Notion image uploads work through `ntn`. If they do not,
  screenshots are hosted links and the row's `Images` property says so.

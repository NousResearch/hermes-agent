# Educator rollout checklist

Ordered steps to stand up the Product Education teammate ("educator") as a
second Slack bot in the existing gateway. Design:
`docs/superpowers/specs/2026-11-16-product-education-teammate-design.md`
("Rollout"). Every step is done by an **admin**, never by the agent.

Conventions: the VM is `ssh ea-hermes`; the repo is `~/ea-hermes` on `main`;
the CLI is `~/ea-hermes/.venv/bin/hermes` (not on the ssh PATH); the gateway
restarts with `cd ~/ea-hermes && sudo -n .venv/bin/hermes gateway restart --system`
and logs with `journalctl -u hermes-gateway.service --since -5m`. Below,
`H=~/ea-hermes/.venv/bin/hermes` and `E=~/.hermes/profiles/educator`.

---

## 0. Before you start

- [ ] PRs #4–#8 (multiplex hardening, cost budget, github_pr, zendesk,
      boundary enforcement) and this package's PR are on `main` and deployed
      (routine deploy: ff-pull `~/ea-hermes`, restart). `gateway.multiplex_profiles`
      is still **off**.
- [ ] `ffmpeg` and `poppler-utils` are installed on the VM (`video_frames`,
      `pdf_pages` hide themselves without them): `which ffmpeg pdftoppm`.
- [ ] The product-skills side is merged on its default branch:
      `plugins/product-education/skills/{education-explore,notebook-keeping,learn-from-approval}`
      and `plugins/release-comms/skills/release-comms/references/hermes-runtime.md`
      (the job prompts load these by name).

## 1. Values to collect

Copy `values.example.yaml` to a private `values.yaml` (not in git) and fill:

- [ ] `PERSONA_NAME` (and the avatar, uploaded in the Slack app later)
- [ ] `HOME_CHANNEL_ID` (`#product-education` or the chosen name) and
      `RELEASE_UPDATES_CHANNEL_ID` — channel **ids** (`C…`), not names
- [ ] `ADMIN_SLACK_USER_ID` / `ADMIN_NAME`; the product team's Slack user ids
      (one `educator_user` line each in the rendered config — the template
      has one example line; add the rest after rendering)
- [ ] `MODEL`, `MODEL_PROVIDER` and its four `pricing.overrides` rates (USD per
      million tokens; `cache_write` too where the model bills it)
- [ ] `DAILY_BUDGET_USD`, `MONTHLY_BUDGET_USD`, `PER_RUN_BUDGET_USD`,
      `BUDGET_ALERT_MENTION`
- [ ] `WORKING_HOURS_TIMEZONE` (veto windows; also check the `days:` line and
      the explore job's cron days match the team's week)
- [ ] `EDUCATOR_HOME` — the absolute profile path on the VM
      (`/home/<user>/.hermes/profiles/educator`)
- [ ] `NOTION_IDEAS_DATA_SOURCE_ID`, `NOTION_HOME_PAGE_ID` (step 2b),
      `ZENDESK_PERMISSION_GROUP_ID` (step 2d), `RELEASE_WATCH_JOB_ID` (step 7)
- [ ] Staging access for screenshots: does a test account exist on
      `staging.base-ai.dev` it may use? (If yes, its credentials are handled
      by the team, not put in this profile's config.)

## 2. Outside setup

### 2a. Slack app

- [ ] Render the manifest: `python deploy/educator/render_config.py values.yaml slack-app-manifest.yaml > manifest.yaml`.
- [ ] api.slack.com/apps → **Create New App → From an app manifest** → paste.
      It is a separate app from the default bot. Upload the avatar.
- [ ] Install to the workspace. Copy the **Bot User OAuth Token** (`xoxb-`).
- [ ] Basic Information → App-Level Tokens → generate with `connections:write`
      (`xapp-`).
- [ ] Invite the bot to the home channel and `#release-updates` (it cannot
      post where it is not a member: `not_in_channel`). Invite it to any other
      channel it should read; it answers there only when mentioned.

### 2b. Notion

- [ ] Create an **internal integration** for the educator (Settings →
      Connections → Develop or manage integrations). Capabilities: read
      content, update content, insert content, read comments, insert comments,
      read user information **without** email.
- [ ] **Schema change (required for gated posting):** on **Releases**, add a
      relation property named exactly `PRD` → Detailed Bucket Features, with
      "Show on Detailed Bucket Features" **off** (one-way). Releases has no
      PRD link today, and both the write guard and the #release-updates gate
      read it. (A two-way relation also works but then set
      `notion.write_guard.prds.back_relation_property` to its PRD-side name.)
- [ ] **Schema change (before any veto level):** add a `Hold` option to
      Release Assets → Status (the veto "hold" signal; absent today).
- [ ] Create the teammate's **home page** and the **Ideas** database under it
      (`notion-ideas-database.md`). Note the page id and the Ideas data source id.
- [ ] Share with the integration (`•••` → Connections) **only**:
      Releases, Release Assets, Detailed Bucket Features, Ideas, and the home
      page. Nothing else — the integration's reach is part of the boundary.

### 2c. Jira (read-only)

- [ ] A dedicated Atlassian account (bot user) with **Browse projects** on BA
      and EV and no other project permission. Create its API token.
- [ ] Prove it is read-only: in that account, an issue shows no Transition /
      Edit / Comment actions.

### 2d. Zendesk (confidential OAuth client)

**Decision 2026-11-17:** the OAuth client is owned by an **admin** account (no
free seat for a dedicated agent). Effective rights = the client's scopes ∩
the owner's role, so the token has full *Help Center* power and nothing
outside it. Accepted because `zendesk_create_draft` (draft-only, curated
allowlist, `draft: true` forced) is the real boundary. Mitigations — all
required:

- [ ] Admin Center → Apps and integrations → APIs → **OAuth clients** → Add:
      **Confidential**, allowed scopes **exactly** `hc:read hc:write`. Never
      leave scopes empty: empty means *all* scopes, i.e. full account admin.
- [ ] Treat any later scope change on this client as a security change.
- [ ] The secret lives only in the password manager and the educator `.env`.
- [ ] Known costs, accepted: Zendesk's audit log attributes the teammate's
      drafts to that admin; the integration breaks if the account loses admin.
      Move to a dedicated agent (Manage APIs + Help Center manager only) when
      a seat frees up.
- [ ] Pick the default permission group for new drafts (Guide → Permission
      groups, or `zendesk_api api/v2/guide/permission_groups.json` once the
      profile runs) → `ZENDESK_PERMISSION_GROUP_ID`.

### 2e. GitHub

- [ ] Fine-grained token, resource owner `everafterai`, **only** these repos:
      `product-skills` — Contents + Pull requests: read/write (+ Checks and
      Commit statuses: read); `ea-monorepo` — Contents + Pull requests: read
      (+ Checks, Commit statuses: read). Set an expiry and a calendar reminder.
- [ ] Branch protection on `product-skills` `main`: require a pull request
      with at least one approval, no direct pushes, no bypass for the token's
      user. (The tool never writes the default branch; this is the second lock.)
- [ ] Confirm `product-skills` has no CI or deploy integration that runs repo
      code on push/PR (package scripts, Netlify/Vercel previews): the tool
      blocks `.github/` but not existing CI. It is a markdown/skills repo —
      keep it that way.

## 3. Create the profile (fresh, not cloned)

```bash
ssh ea-hermes
H=~/ea-hermes/.venv/bin/hermes; E=~/.hermes/profiles/educator
$H profile create educator --no-skills \
  --description "Product education teammate: drafts release comms, audits the help center, proposes ideas."
```

- [ ] **Not** `--clone` / `--clone-all`: nothing of the default bot may carry
      over (credentials, memory, skills, sessions).
- [ ] `--no-skills`: its skills come only from the product-skills checkout.

## 4. Place the files

```bash
cd ~/ea-hermes
mkdir -p $E/local && cp /path/to/values.yaml $E/local/values.yaml   # ids, no secrets
# Shadow week from day one: start with #release-updates OUT of post_allowed_channels
.venv/bin/python deploy/educator/render_config.py $E/local/values.yaml > $E/config.yaml
.venv/bin/python deploy/educator/render_config.py $E/local/values.yaml SOUL.md > $E/SOUL.md
cp deploy/educator/.env.example $E/.env && chmod 600 $E/.env      # then fill it in
mkdir -p $E/scripts $E/workspace/{drafts,metrics,research,checkouts}
install -m 0644 deploy/educator/scripts/*.py $E/scripts/
cp -n deploy/educator/workspace/notebook.md $E/workspace/notebook.md
```

- [ ] `RELEASE_WATCH_JOB_ID` is not known yet: render with any 12-hex value
      and fix it in step 7.
- [ ] Add one `"U…": educator_user` line per product-team member under
      `slack.user_roles` (and `user_names`). From here on edit `$E/config.yaml`
      by hand — re-rendering would overwrite these lines.
- [ ] **Shadow week:** delete the #release-updates channel id line under
      `slack.post_allowed_channels` now (keep it under
      `post_gated_channels`). The startup log then shows one expected
      `[boundary] … post_gated_channels lists channel(s) missing from
      slack.post_allowed_channels` line — that *is* the refusal.
- [ ] Fill `$E/.env` (every name in `.env.example`).
- [ ] Clone the read-only checkouts with the educator's token (env-config,
      so the token stays out of argv):

```bash
cd $E/workspace/checkouts
set -a; . $E/.env; set +a
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.https://github.com/.extraheader \
       GIT_CONFIG_VALUE_0="Authorization: Basic $(printf 'x-access-token:%s' "$GITHUB_PR_TOKEN" | base64 -w0)"
git clone --quiet https://github.com/everafterai/product-skills.git
git clone --quiet --depth 1 https://github.com/everafterai/ea-monorepo.git
unset GIT_CONFIG_COUNT GIT_CONFIG_KEY_0 GIT_CONFIG_VALUE_0 GITHUB_PR_TOKEN
```

- [ ] Validate the boundary config with the gateway's own reader (expect
      `[]`, or only the shadow-week line):

```bash
cd ~/ea-hermes && HERMES_HOME=$E .venv/bin/python -c \
 "from agent.boundary_config import read_boundary_config as r, boundary_config_errors as e; print(e(r()))"
```

## 5. Pricing

- [ ] `pricing.overrides` for the model is in the rendered config (required
      by the budget — without it every budget alert is a lower bound).
- [ ] After the first day: `$H -p educator costs --days 1 --by job` shows
      priced (`estimated`/`actual`) rows, not `unpriced`.

## 6. Turn on multiplex

- [ ] In the **default** profile's `~/.hermes/config.yaml`:

```yaml
gateway:
  multiplex_profiles: true
  multiplex_profile_allowlist: [educator]
```

- [ ] Restart: `cd ~/ea-hermes && sudo -n .venv/bin/hermes gateway restart --system`;
      `systemctl is-active hermes-gateway.service`.
- [ ] Logs: `journalctl -u hermes-gateway.service --since -5m | grep -Ei "educator|boundary|multiplex"`
      — both Slack adapters connected; no `[boundary] profile 'educator'`
      errors other than the shadow-week line; no env/config disagreement
      warnings you did not expect.

**Separation checks** (multiplex design "Acceptance", repeated with the real
apps — both directions):

- [ ] Identity: DM each bot; each answers as itself, in its own app's DM.
- [ ] Permissions: `$H -p educator users list` shows only the educator's
      roles; a Slack user with no educator role gets no answer from it, while
      the default bot still answers its own users.
- [ ] Tools: ask the educator to list its tools — no `terminal`,
      `code_execution`, `jira_write`, `messaging`, `cronjob` or any MCP tool.
- [ ] Credentials: ask the educator to call
      `jira_api rest/api/3/myself` — it must be the read-only bot account,
      not the default bot's. Same for Notion (`notion_api v1/users/me` → the
      educator integration).
- [ ] State: the educator cannot read a file under `~/.hermes/` outside
      `$E` (file tools refuse another served profile's home).
- [ ] Default bot regression: one normal request to the default bot in its
      usual channel works as before.

## 7. Cron jobs (admin-created; scripts attached by a super admin)

```bash
cd ~/ea-hermes
HERMES_HOME=$E .venv/bin/python deploy/educator/apply_cron_jobs.py $E/local/values.yaml --dry-run
HERMES_HOME=$E .venv/bin/python deploy/educator/apply_cron_jobs.py $E/local/values.yaml
```

The applier refuses unless the scripts in `$E/scripts/` are byte-identical
to the repo's, the gateway's Python can import them, the config exists, and
the scheduler's capability preflight passes. It creates `release-watch`
(every 20 min, essential) and `explore` (09/12/15, Sun–Thu, discretionary)
with explicit `enabled_toolsets`, and prints the job ids.

- [ ] Put release-watch's id in `cost_budget.essential_jobs` (edit
      `$E/config.yaml` and `$E/local/values.yaml` → `RELEASE_WATCH_JOB_ID`).
      Until then **every** job is discretionary (`default_discretionary: true`).
- [ ] Ownership (the cron RBAC ceiling reads the owner's role every run):

```bash
$H -p educator own claim cron:<release-watch id> --user <ADMIN_SLACK_USER_ID> --name "<ADMIN_NAME>"
$H -p educator own claim cron:<explore id>       --user <ADMIN_SLACK_USER_ID> --name "<ADMIN_NAME>"
```

- [ ] `$H -p educator cron list` shows both, enabled.
- [ ] First release-watch run: `$H -p educator cron run <release-watch id>`.
      It is a **baseline**: it drafts nothing; if PRDs already have tickets
      at Ready for staging or later it posts one message listing them and
      asks which to backfill. Silent otherwise. Check-ups: missing checkout,
      Notion/Jira credential or config problems are posted once to the home
      channel.

For reference, the equivalent CLI (it cannot set `enabled_toolsets`,
`required_toolsets` or `discretionary`, which is why the applier exists):
`$H -p educator cron create "every 20m" "<prompt>" --name release-watch --script educator_release_watch.py --workdir $E/workspace --deliver slack:<HOME_CHANNEL_ID>`.

## 8. Tell the team how it works

- [ ] Kits start when a PRD's ticket reaches **Ready for staging** (PRD
      `Release Comms Status` = Not started; the PRD's `Release Comms Assets`
      is the checklist; `Comm Owner` reviews).
- [ ] Sign-off is a Status change by a person: asset rows → `Approved`. The
      **Release row must be moved past Draft** (In review / Ready) before its
      Slack update can post. The bot never moves anything past Draft.
- [ ] Ideas: set Accepted / Rejected / Parked and write a **Decision note** —
      it learns from the note.
- [ ] Veto (later, when an action is graduated): ❌ on its announcement, or
      the row's Status → `Hold`.

## 9. Shadow week

Posting to `#release-updates` stays off (step 4). For a week:

- [ ] Kits drafted for real tickets; review the drafts, approve as usual.
- [ ] Edit rate: `$E/workspace/metrics/edit-rate.csv`; the weekly trend post.
- [ ] Noise: no "nothing changed" posts, Ideas worth reading.
- [ ] Spend: `$H -p educator costs --days 7 --by job`; budget alerts land in
      the home channel.
- [ ] Refusals: `grep -i educator ~/.hermes/profiles/educator/audit/data-access.log`
      and the gateway journal for `notion_write_guard` / posting refusals —
      each one should be the boundary working, not a broken flow.

## 10. Enable posting

- [ ] Add `RELEASE_UPDATES_CHANNEL_ID` back to `slack.post_allowed_channels`
      in `$E/config.yaml`. It is picked up on the next tool call (the config
      is mtime-cached); the shadow-week boundary log line stops after the next
      restart.
- [ ] First real post: an Approved Slack-post row whose Release row is past
      Draft and whose PRD tickets are all Done. Check the post, then
      `Published URL` + `Status = Published` on the row.

## Rollback and kill switches

- Stop autonomy above sign-off: `autonomy.paused: true` (instant, no restart).
- Stop posting: remove `#release-updates` from `post_allowed_channels`.
- Stop the jobs: `$H -p educator cron pause <id>`.
- Remove the bot: take `educator` out of `multiplex_profile_allowlist` and
  restart. Its profile, sessions and workspace stay on disk.
- Turning a guard **off** (write guard, `agent_write_guard`, posting lists)
  needs an edit **and** a gateway restart — they are sticky until restart.

## Later: graduation

The teammate proposes its own graduation on the Ideas board (Type
`autonomy`) when an action's record meets the bar. A human decides and edits
`autonomy.actions.<action>`; `veto` / `auto` exist only for
`post_release_update` and `zendesk_create_draft` (add the `Hold` option
first). Everything else stays `off` until its code path is built.

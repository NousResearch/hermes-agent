# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## This is a fork

This repo is **`everafterai/ea-hermes`** — a fork of NousResearch's
[hermes-agent](https://github.com/NousResearch/hermes-agent), customized for
EverAfter's needs. It is **deployed to a Google Cloud VM** (the gateway runs there,
serving Slack and other messaging platforms).

When working here:
- **[AGENTS.md](AGENTS.md) is the canonical, upstream development guide** — project
  structure, the `AIAgent` loop, CLI/TUI architecture, tools/toolsets, plugins,
  skills, cron, kanban, profiles, prompt-caching invariants, and the full list of
  "Known Pitfalls". Read it; this file does not duplicate it.
- This file documents only what is **fork-specific** or most load-bearing day to day.
- Prefer the plugin route for new local-only tools (see AGENTS.md "Adding New Tools")
  rather than editing core — it keeps merges with upstream clean.

### Upstream sync state (last synced 2026-08-24, upstream v0.20.5)

Sync plan + conflict inventory:
[docs/superpowers/plans/2026-08-24-upstream-sync.md](docs/superpowers/plans/2026-08-24-upstream-sync.md).

Structural changes from that sync that affect where fork work goes:

- **Slack is now a bundled platform plugin.** `gateway/platforms/slack.py` is
  gone; the adapter lives at
  [plugins/platforms/slack/adapter.py](plugins/platforms/slack/adapter.py)
  (Feishu and WeCom moved the same way). Slack YAML→env translation moved into
  that file's `_apply_yaml_config`. The `slack:` config block and
  `config.extra` bridging are unchanged.
- **Gateway authorization was extracted into a mixin.** `_is_user_authorized`
  now lives in [gateway/authz_mixin.py](gateway/authz_mixin.py); `gateway/run.py`
  keeps only call sites.
- **Session search moved.** `SessionDB.search_messages` is in
  [hermes_state_search.py](hermes_state_search.py) and the schema/DDL in
  [hermes_state_common.py](hermes_state_common.py). The fork's SQL-level scope
  gate is applied on **four** query paths there (FTS5, trigram-in-impl, LIKE,
  and `_run_trigram_search`).
- **Some fork patches are now subsumed by upstream** and were dropped:
  compression-session identity inheritance (upstream's
  `publish_compression_child` COALESCEs it inside the transaction), the
  `/resume` identity gate (upstream's `_resume_target_allowed` /
  `_resume_row_visible` are stricter), and the fork's `reference_images`
  image-to-image implementation.
- **An upstream default is overridden: bot-authored thread roots no longer
  wake the bot.** The sync added a 4th check to
  `_should_wake_on_unmentioned_message`
  ([plugins/platforms/slack/adapter.py](plugins/platforms/slack/adapter.py)):
  if the *thread root* was authored by the bot, any un-mentioned reply wakes
  it (upstream #63530). Unlike the three in-memory checks beside it, that one
  reads Slack history, so it is **retroactive and permanent** — every thread
  the bot ever started, surviving restarts. In an automation-feed channel
  (the bot posts the root of every item via `slack_post_thread`, and humans
  discuss underneath) that silently converts the whole channel to
  free-response and defeats `require_mention`. It is now opt-in via
  `slack.wake_in_bot_authored_threads` (default **false** = pre-sync
  behaviour; env `SLACK_WAKE_IN_BOT_AUTHORED_THREADS`, bridged in the
  adapter's `_apply_yaml_config`). Check 5 (the thread *parent* @-mentions
  the bot, #24848) is deliberately **not** gated — that carries explicit
  human intent. Gating check 4 exposed a latent bug in check 5:
  `_fetch_thread_parent_text(strip_bot_mention=False)` returned text with the
  mention already stripped, because `_render_message_text` removes it
  unconditionally — so check 5 only ever worked when check 4 had warmed the
  thread-context cache. The same flag also gates **cron deliveries arming
  check 1**: a live-adapter cron send (`metadata["job_id"]`, stamped by
  `DeliveryRouter`) used to record its ts in `_bot_message_ts`, so every item
  a feed job posted woke the bot on every un-mentioned reply until restart
  (seen 2026-09-23 under the customer news radar: the bot called `turn_end` —
  a no-op outside quiet channels — then explained its silence). With the flag
  off, `send()` skips that registration for cron posts; conversational sends
  are unchanged, and continuable cron threads still wake via their seeded
  session (check 3). All of these are covered by
  [tests/gateway/test_slack_wake_bot_authored_threads_optin.py](tests/gateway/test_slack_wake_bot_authored_threads_optin.py)
  and guarded in `tests/test_fork_feature_inventory.py`.

## Fork-specific work: Slack per-user RBAC + multi-user session isolation

The fork's primary divergence from upstream is **role-based access control (RBAC)
over tool execution for messaging-platform users (primarily Slack)**, plus the
**multi-user session isolation** fixes that go with running one gateway for many users.

### RBAC policy module — [gateway/tool_access.py](gateway/tool_access.py)

A pure, unit-testable `ToolAccessPolicy` that sits beside the chat allowlist
([gateway/run.py](gateway/run.py) `_is_user_authorized`) and the slash-command
tiers ([gateway/slash_access.py](gateway/slash_access.py)) and adds a third axis:
**which toolsets an identified platform user may invoke**, keyed to named roles.

- **Built-in roles** (`BUILTIN_ROLES`): `admin` (`*`), `operator`, `readonly`,
  `chat_only` (no tools). `operator` deliberately **excludes `terminal`** (so an
  operator — including a `channel_roles` poster in an issue-tracking channel —
  can't read host secrets via `printenv`/`cat ~/.hermes/.env`; `file`/`web`/`browser`/
  `notion`/`session_search` cover its work). `FLOOR_TOOLSETS` (`clarify`, `todo`, `slack`,
  `ownership`) are granted
  to every valid-role user — but NOT to roleless/undefined-role users. `slack`
  (`slack_react` + `turn_end`) is a floor because reacting/closing a turn is UX,
  not a privilege — so the bot can acknowledge any user's message in a quiet
  channel, not just an admin's. `ownership` is a floor for the same reason —
  administering the ownership registry otherwise means reaching the `hermes own`
  CLI, which needs `terminal` (see "Automation ownership" below).
- **Config lives under the top-level `slack:` block** in `~/.hermes/config.yaml`
  (`user_roles`, optional `user_names`, optional `roles`, optional `channel_roles`).
  The gateway config loader ([gateway/config.py](gateway/config.py)) bridges these
  keys into the platform's runtime `extra`. **Hand-edit under `slack:` directly,
  NOT `slack.extra`.**
- **`channel_roles` (`{chat_id: role}`, Slack-only)** grants a fixed *service
  role* to EVERY poster in that channel, so a channel (e.g. issue-tracking) works
  no matter who reports — even a roleless user. The channel grant is **UNIONed**
  with the poster's own role (additive; never reduces access) and is resolved at
  all three enforcement points via the chat id (`ToolAccessPolicy._effective_grant`;
  the backstop reads `HERMES_SESSION_CHAT_ID` via `_current_chat_id`). It is
  **inert unless RBAC is active** — it does NOT by itself enable RBAC (activation
  stays keyed to `user_roles`). **Security:** anyone who can post in such a channel
  can invoke that role's tools (e.g. `terminal`), so scope the role to least
  privilege and keep the channels internal.
- **Activation is `user_roles` presence.** When `user_roles` is non-empty, RBAC is
  the sole authorization source — a user with a role may chat and gets that role's
  toolsets; a roleless user is denied entirely (deny-until-assigned). It also
  short-circuits/retires `SLACK_ALLOWED_USERS` and ignores `SLACK_ALLOW_ALL_USERS`.
- **Backward compatible / fail-closed:** absent `user_roles` → policy disabled,
  upstream behavior exactly. A `user_roles` entry naming an undefined role denies
  that user and logs.
- **Three enforcement points, one policy:** (A) message gate in
  `_is_user_authorized` ([gateway/authz_mixin.py](gateway/authz_mixin.py) —
  upstream extracted the authorization cluster out of `gateway/run.py`),
  (B) toolset filter inside `_resolve_enabled_toolsets_for_source`
  ([gateway/run.py](gateway/run.py)), which feeds `enabled_toolsets` into
  `model_tools.get_tool_definitions` (cache-safe — `enabled_toolsets` is in the
  memo key); centralized there so every call site inherits it, (C) execution backstop in
  the `pre_tool_call` hook in [model_tools.py](model_tools.py) that resolves
  identity from session contextvars and hard-blocks forbidden tools (covers
  delegation sub-agents, the code-execution sandbox, plugin-invoked calls).

Design + plan docs (read these before changing RBAC):
[docs/superpowers/specs/2026-05-31-slack-tool-rbac-design.md](docs/superpowers/specs/2026-05-31-slack-tool-rbac-design.md)
and [docs/superpowers/plans/2026-05-31-slack-tool-rbac.md](docs/superpowers/plans/2026-05-31-slack-tool-rbac.md).

### `hermes users` CLI — [hermes_cli/users.py](hermes_cli/users.py)

Manages RBAC users with comment-preserving writes to `config.yaml`:
`hermes users list | add <id> <role> [--name] | update <id> [--role] [--name] | delete <id>`.
Promoting a user to `admin` keeps `allow_admin_from` (the slash-admin list) in sync;
demoting/deleting removes them.

### `hermes tools rbac` — lists tools by toolset with built-in role coverage.

### Fork-added toolsets — Notion, Jira, Slack thread posting, Webflow assets, video frames

Five fork-only integrations, each registered as its **own** registry toolset so RBAC
gates them independently. The four API-backed ones read credentials through
[agent/fork_secrets.py](agent/fork_secrets.py) `read_secret` and, **single-profile
only**, self-heal `~/.hermes/.env` (`heal_dotenv`) for cron/delegation runs that never
loaded it, so they work headless. Under multiplex they read only the owning profile's
secret scope and never self-heal (see "Multiplex" below).

- **`notion_api` (`notion` toolset)** — [tools/notion_api_tool.py](tools/notion_api_tool.py).
  Wraps the `ntn` CLI (`ntn api <v1/...> -X <METHOD>`, JSON body via **stdin**,
  `shell=False` fixed-argv so nothing the model supplies can inject). GET/POST/PATCH/DELETE;
  supports a `markdown` body + the `v1/pages/<id>/markdown` endpoint (ntn does
  Markdown→blocks). Creds `NOTION_API_KEY`/`NOTION_API_TOKEN` (bridged), `NOTION_KEYRING=0`
  for headless. **Granted to the `operator` built-in role**; cross-platform; the agent is
  told to reach Notion ONLY through this tool (never curl/shell/python). Design:
  [docs/superpowers/specs/2026-06-15-notion-api-tool-design.md](docs/superpowers/specs/2026-06-15-notion-api-tool-design.md).
- **`jira_api` (`jira` toolset)** — [tools/jira_api_tool.py](tools/jira_api_tool.py).
  **Read-only** (`_ALLOWED_METHODS = {"GET"}`; path must start `rest/`); direct `aiohttp`
  Atlassian-Cloud basic auth, proxy-aware. Creds `JIRA_BASE_URL`/`JIRA_EMAIL`/`JIRA_API_TOKEN`.
  **Not in any built-in role** — reachable only by `admin` (`*`) or an explicit custom
  `roles:` grant; built for a reconciliation/cron worker that reads issue status without a
  shell. Design:
  [docs/superpowers/specs/2026-06-28-jira-issue-status-linkage-design.md](docs/superpowers/specs/2026-06-28-jira-issue-status-linkage-design.md).
- **`jira_api_write` (`jira_write` toolset)** — same file. The mutating twin,
  registered as its **own** toolset so granting `jira` never implies writes.
  Jira API tokens carry no scopes, so the tool's **curated allowlist is the
  boundary** (`_WRITE_ALLOWLIST`, exact `(method, path-regex)` pairs): `POST
  …/issue/<KEY>/transitions`, `POST …/issue/<KEY>/comment`, `PUT …/issue/<KEY>`
  (fields), `POST …/issue` (create). `<KEY>` is a project key + number only —
  numeric ids, query strings, trailing segments (so `../` traversal) and every
  other endpoint (DELETE, project/workflow/user admin, bulk, attachments) are
  refused before any request is made. Reads stay on `jira_api`: the description
  tells the model to list transitions / `editmeta` first and pass the id. Not in
  any built-in role — granted to the config's `rd` role (2026-09-16). Same
  creds and `check_fn` as the read tool. **Deployment:** must be listed in
  `platform_toolsets.slack` (an explicit list shadows defaults). No design doc.
- **`slack_post_thread` (`slack_post` toolset)** — [tools/slack_post_thread_tool.py](tools/slack_post_thread_tool.py).
  Posts to an explicit `chat_id` (+ optional `thread_ts`; omit it for a new root
  message) via `chat.postMessage` (takes them as args, no session contextvars →
  cron/worker safe), and returns `message_ts` + a best-effort `permalink`
  (`chat.getPermalink`) so a worker can cite the post as a receipt. **Deliberately a SEPARATE, NON-FLOOR toolset** —
  unlike the floor `slack` toolset (`slack_react`+`turn_end`), `slack_post` must be granted
  to a role, so a valid-role user does NOT get arbitrary thread-posting for free. It is how
  a cron/delegated sub-agent posts to Slack, since cron hard-disables the `messaging`
  toolset that `send_message` lives in. Token via `_resolve_slack_token()` (shared with
  `slack_react`). No design doc.
- **`webflow_asset_upload` (`webflow_assets` toolset)** — [tools/webflow_asset_tool.py](tools/webflow_asset_tool.py).
  Uploads a **local file** to the Webflow asset CDN via the Data API's two-leg flow
  (`POST /v2/sites/<id>/assets` with `fileName`+`fileHash` md5 → presigned S3 form →
  multipart POST of the bytes), returning `hosted_url` for the agent to drop into a CMS
  item. Exists because **webflow-mcp-server 1.0.0 (the latest) has NO asset-creation
  tool** — its `asset_tool` only does `create_folder`/`get_all_assets_and_folders`/
  `update_asset`, and is Designer-session-backed besides — so without this the agent
  cannot get a cover image into Webflow and falls back to the dashboard in a `browser`,
  which hits a login/human-verification wall. Cred `WEBFLOW_API_TOKEN` (needs the
  `assets:write` scope; `WEBFLOW_TOKEN` accepted as a fallback since the MCP block
  remaps to it). **Multi-site:** a Webflow token is scoped to ONE site, so hosts
  managing several set `WEBFLOW_SITE_TOKENS` in `.env` to a
  `site_id:ENV_VAR_NAME` comma list (var *names*, so secrets stay one-per-line);
  the tool resolves the token from its `site_id` argument, falling back to
  `WEBFLOW_API_TOKEN` for unmapped sites, so a single-site install is unchanged.
  Each site also needs **its own `mcp_servers` entry** (`webflow` = base.ai,
  `webflow_everafter` = everafter.ai) — same package, second process, second
  token. Use an **underscore** in that server name: `sanitize_mcp_name_component`
  rewrites `-`→`_` in the tool prefix but not the toolset name, so a hyphen
  splits `mcp-webflow-x` from `mcp__webflow_x__*`. Grant BOTH spellings
  (`webflow_everafter` + `mcp-webflow_everafter`) in the role. Approval globs in
  `approvals.require_for_tools` must use the real dispatch spelling
  `mcp__<server>__<tool>` — exact `fnmatch`, no normalization, so a wrong prefix
  is a silently inert gate. **Deliberately a separate toolset from `mcp-webflow`** — writing bytes
  to a public CDN is not implied by reading collections — granted to the config's
  `webflow_publisher` role alongside `webflow`/`mcp-webflow`. Reads are guarded by
  `get_read_block_error` + `is_protected_data_path` ([agent/file_safety.py](agent/file_safety.py))
  so a credential store or another user's session data can never be published to a CDN
  URL. **Deployment:** must also be listed in `platform_toolsets.slack` (an explicit
  list shadows defaults) or the tool is silently missing. No design doc.
- **`video_frames` (`video_frames` toolset)** — [tools/video_frames_tool.py](tools/video_frames_tool.py).
  Extracts still PNG frames from a **local** video file via ffmpeg and returns their
  paths, so `vision_analyze` can QA them and `webflow_asset_upload` can push them to
  the CDN. Exists because the upstream `video` toolset's `video_analyze` returns
  **prose, not images** — it base64s the whole clip to a multimodal model — so an agent
  handed a demo MP4 in Slack had no route to a screenshot except `terminal`, i.e. a host
  shell for a screen grab. Takes explicit `timestamps` (seconds or `"1:05"` clock form)
  or a `count` sampled evenly across the duration (offset half a step at each end, since
  the first/last frame of a screen recording is usually a fade or title card). ffprobe
  supplies the duration; `asyncio.create_subprocess_exec` with a **fixed argv** (no shell,
  no string interpolation) runs both legs; `max_width` scales via `scale='min(W,iw)':-2`
  so it never upscales. Frames land in `$HERMES_HOME/cache/images/`. Source paths go
  through `raise_if_read_blocked` ([agent/file_safety.py](agent/file_safety.py)) **before**
  the existence check, so the tool can't be used to probe for another user's files, and
  through an extension allowlist. Capped at 24 frames/call. `check_fn` hides the tool when
  ffmpeg/ffprobe are absent rather than failing per call — so **the host needs
  `apt-get install -y ffmpeg`** or the tool silently never appears. **Deliberately a
  separate toolset from `video`** — "describe this clip" should not imply "extract and
  republish its frames", and neither should require `terminal`. Not in any built-in role;
  grant it explicitly. **Deployment:** must also be listed in `platform_toolsets.slack`
  (an explicit list shadows defaults) or the tool is silently missing. No design doc.
- **`pdf_pages` (`pdf_pages` toolset)** — [tools/pdf_pages_tool.py](tools/pdf_pages_tool.py).
  The PDF sibling of `video_frames`: renders selected pages of a **local** PDF
  to PNG via poppler (`pdfinfo` for the count, `pdftoppm -singlefile` per
  page, fixed argv, no shell) into `$HERMES_HOME/cache/images/` so
  `vision_analyze` can read them. Exists because a scanned PDF has no text
  layer for `read_file`/`drive_read_file` to extract, and the extractor's own
  `NEEDS OCR` hint ("render with pdftoppm, inspect via vision_analyze") assumes
  a shell; upstream's shell-free route is hosted Firecrawl OCR
  (`FIRECRAWL_API_KEY`), which we chose not to add — the agents already run on
  vision models, which read Hebrew statements better than tesseract would.
  Page spec `"1-3,7"`, cap 20 pages/call (`_MAX_PAGES`), dpi 50–300 (default
  150). Source path through `raise_if_read_blocked` **before** the existence
  check; `.pdf` only. `check_fn` hides the tool when poppler is absent —
  **the host needs `apt-get install -y poppler-utils`** (present on the VM).
  Own toolset for the `video_frames` reason; granted to `builder` (2026-09-16).
  **Deployment:** must be listed in `platform_toolsets.slack`. No design doc.
- **`github_pr_read` (`github_pr` toolset) + `github_pr_write` (`github_pr_write`
  toolset)** — [tools/github_pr_tool.py](tools/github_pr_tool.py). A narrow,
  shell-free GitHub REST tool (aiohttp, proxy-aware) for secondary profiles
  — built for the Product Education teammate, which improves shared skills
  only by PRs that humans merge and reads the monorepo's PRs to learn what
  shipped. It exists because the default profile's GitHub MCP is kept away
  from secondary profiles, and `terminal` is super-admin only; it does
  **not** depend on any MCP. Reads: `list_prs`, `get_pr`, `pr_files`,
  `pr_diff` (truncated at 60k chars), `pr_comments` (conversation + inline
  review comments + reviews), `pr_status` (combined status + check runs),
  `commits` (`pr_comments` pages up to 500 per kind and flags `truncated`).
  Writes: `create_branch`, `commit_files` (one commit of full-file
  contents onto an existing branch via the git-data API, `force: false`),
  `open_pr` (always targets the default branch; there is no `base`
  parameter, and a passed `base` other than the default is refused).
  **There is no merge op, no settings/admin endpoint, and the
  default branch is never written** — it is looked up live
  (`GET repos/<repo>`) before every write and a failed lookup refuses.
  `commit_files` refuses when the branch's tree, the new tree, or the new
  commit SHA is missing (a missing `base_tree` would build a tree of only
  the listed files and delete the rest of the branch), and refuses to touch
  an existing entry that is not a regular file (`100644`) rather than
  silently turning an executable/symlink/submodule into one — it reads the
  branch tree recursively and refuses if GitHub truncates it. Paths under
  `.github/` and `.git` are refused, as are traversal/ref-syntax branch
  names. Requests never follow redirects (a 307 from a renamed repo would
  replay a write past the allowlist — a 3xx is an error naming the fix),
  and response bodies are read bounded (5 MB JSON — a truncated JSON body is
  refused — and 120 KB for `pr_diff`). **The boundary is the profile's
  per-repo action allowlist**, read per call (mtime-cached) from
  `config.yaml`:
  ```yaml
  github_pr:
    repos:
      everafterai/product-skills:
        actions: [read_pr, read_commits, create_branch, commit_files, open_pr]
        branch_prefix: educator/   # REQUIRED when any write action is granted
      everafterai/ea-monorepo:
        actions: [read_pr, read_commits]
  ```
  An unlisted repo or ungranted action is refused before any request.
  `branch_prefix` (ending in `/`) is **required** for any repo granting a
  write action, and `create_branch`'s branch, `commit_files`' branch and
  `open_pr`'s head must all be under it — without it a prompt-injected agent
  could commit onto a human's open feature branch (whose merge then carries
  the commit into the default branch unreviewed) or create `release/*` /
  `gh-pages`. A missing or malformed prefix, an unknown action name — any
  `merge*` especially — or `autonomy.actions.merge_skill_pr` above `off`
  poisons the whole block (every call refused, naming the problem): merging
  is not built in v1. **Residual CI risk:** blocking `.github/` stops the
  agent adding or editing *workflows*, but not push- or PR-triggered CI that
  already exists from running repo code the agent committed (package
  scripts, tests, a Makefile) or deploy configs read from the repo root
  (`vercel.json`, `netlify.toml`, a Netlify/Vercel/Cloudflare preview
  integration). Grant write actions only on a repo with no such CI or
  deploy integration, or accept that risk explicitly per repo
  (`product-skills` is a skills/markdown repo; check it stays that way).
  Token **only** via `get_secret("GITHUB_PR_TOKEN")` (no `os.getenv`, no
  dotenv self-heal — under multiplex `os.environ` is another profile's);
  `UnscopedSecretError` reads as "not configured". Use a fine-grained token
  whose repo list matches the allowlist: product-skills contents + pull
  requests read/write, ea-monorepo contents + pull requests read-only
  (add Checks/Commit statuses read for `pr_status`; check runs degrade to a
  `check_runs_error` field without it). `check_fn` shows each tool only
  when the token resolves and some repo grants one of its actions (a config
  error keeps it visible so the refusal explains itself). Separate toolsets
  so granting reads never implies writes. **Not in any built-in role** —
  grant explicitly. **Deployment:** both must be listed in
  `platform_toolsets.slack` (an explicit list shadows defaults) or the tools
  are silently missing. Plan:
  [docs/superpowers/plans/2026-11-16-github-pr-toolset.md](docs/superpowers/plans/2026-11-16-github-pr-toolset.md).
- **Google Drive per-user access check** — [plugins/google_drive_sa/access.py](plugins/google_drive_sa/access.py).
  The Drive/Sheets/Docs plugin acts as ONE service account, so "shared with the
  SA" would otherwise be "readable by every user whose role grants the toolset".
  Every handler now calls `require_access(file_id, reader|writer)` before its API
  call: resolve the requester (session contextvars, or the cron job's owner via
  the ownership registry — `cron/tool_approval_context.get_cron_job_id`), fetch
  the file's ACL as the SA (cached, `google_drive.acl_cache_ttl_seconds`), and
  evaluate it. `drive_list_files` filters results the same way (inline
  `permissions` from `files.list`; `permissions.list` in a thread pool for
  shared-drive items). **Groups fail closed** — no DWD, so a group grant counts
  only via `google_drive.everyone_groups` (treated as domain-wide) or
  `google_drive.group_members` (manual map); denials audit the unmapped groups to
  `audit/data-access.log` (`drive_access_denied`) so you can see what to map.
  Email comes from `slack.user_emails` → Slack `users.info` (needs the
  `users:read.email` scope) → written back into `slack.user_emails`
  (`hermes users add/update --email`); a failed lookup (API error, not just a
  missing mapping) is caught in `resolve_requester()` and denies rather than
  raising. Files created without a folder are shared with the requester as
  writer, and a root-create (no folder) with the check active but no
  resolvable requester is **denied** rather than silently created
  SA-only-visible (`_requester_for_create` in
  [plugins/google_drive_sa/tools.py](plugins/google_drive_sa/tools.py)). No
  requester in a gateway/cron session → deny. A local operator session is
  skipped (`_is_local_operator_session`): a plain CLI never engages the
  session-context system at all, and `hermes --tui`/the desktop app *do*
  engage it but bind `source="tui"`/`"desktop"` with no platform — both read
  as the operator's own machine, the fork's "a shell caller is an admin"
  rule; cron binds an empty platform *and* empty source together, so it's
  never mistaken for local. `drive_list_files` filters results the same way
  and, with the check active, always returns `next_page_token: null` even
  when Drive returned one — the tool takes no `page_token` input, so a raw
  token can't be used to page and would only leak "more matches exist" for
  files filtered out of the response, an existence oracle over content the
  requester can't see. **`google_drive.folder_access`** (2026-09-24) is the
  operator grant for files whose ACL the SA cannot read: a shared-drive folder
  shared with the SA as a non-member *viewer* returns an empty permission list,
  so the check denied everyone (Pazit on an order form, Shai on a sheet).
  Adding the SA as a drive member would fix visibility but expose the whole
  drive, so instead `{folder_id: {name, readers: [emails], writers: [emails]}}`
  grants listed people access to everything under that folder, found by
  walking `parents` (cached, no API call when unconfigured); additive only.
  An unreadable ACL now denies with "can't read its sharing settings" (audit
  `reason=acl_unreadable`) instead of blaming the user. `fetch_acl` treats an empty inline `permissions: []`
  the same as absent and falls back to `permissions.list` — an SA-visible
  file always has at least the SA's own entry, so `[]` can never be the real
  ACL. The Drive/Sheets/Docs `googleapiclient` service is built **per-thread**
  (`threading.local`, [plugins/google_drive_sa/client.py](plugins/google_drive_sa/client.py))
  because its `httplib2` transport isn't thread-safe and ACL fetches fan out
  over a thread pool — credentials are cached process-wide, only the service
  object is per-thread. Kill switch
  `google_drive.access_check: false`. **Binary reads never return base64**
  (2026-09-16): `drive_read_file` saves a non-text file to
  `$HERMES_HOME/cache/drive/<file_id>/<name>` and returns `local_path` (for
  `read_file` / `vision_analyze` / `video_frames`), and runs document formats
  (PDF, DOCX, XLSX, PPTX, … — whatever `tools/read_extract.py` handles; PDF
  needs the optional `anydoc` package, installed on the VM) through the
  extractor right there, returning `encoding: "text"`. A base64 blob was
  useless to the only people who could call the tool — RBAC builders have no
  shell to decode it. Document type decides the path, not decodability (a PDF
  header is ASCII), and a NUL byte marks any other file binary. Scanned PDFs
  still need hosted OCR (`FIRECRAWL_API_KEY`). **Deployment:** default-on with no
  `google_drive:` config block, so before enabling on the VM: add the
  `users:read.email` Slack scope and reinstall the app → set
  `google_drive.everyone_groups` for any domain-wide-share groups → give
  every Drive-reading cron job an owner (`hermes own claim cron:<id>`, or the
  `ownership` tool) so it isn't ownerless-and-thus-denied → deploy; or ship
  first with `google_drive.access_check: false` and flip it on once the above
  is in place. Design:
  [docs/superpowers/specs/2026-09-14-drive-per-user-access-check-design.md](docs/superpowers/specs/2026-09-14-drive-per-user-access-check-design.md).

### Session visibility / multi-user isolation — [hermes_state.py](hermes_state.py)

`SessionDB` gained `chat_id`/`chat_type` columns and visibility helpers so
`session_search` and `/resume` scope to the requesting identity (a series of
cross-user-leak security fixes). DM scope is strictly partitioned from channel
sessions. `chat_type` propagates via session contextvars
([gateway/session_context.py](gateway/session_context.py)). Owner identity is
inherited into compression + branch sessions to preserve self-recall. When
touching session search/resume/scoping, preserve these isolation guarantees — they
have end-to-end tests under [tests/](tests/) (e.g. multi-user `session_search` isolation).

### Cross-user data-access protection — [agent/file_safety.py](agent/file_safety.py) + [agent/data_access_audit.py](agent/data_access_audit.py)

App-layer session/memory isolation never touches bytes on disk, and the gateway
runs as one OS uid, so a filesystem-capable tool can read other users' data.
Two controls (design:
[docs/superpowers/specs/2026-06-29-cross-user-data-access-protection-design.md](docs/superpowers/specs/2026-06-29-cross-user-data-access-protection-design.md)):

- **Close (file tool):** `is_protected_data_path` recognizes the session DB
  (`state.db`), memory DBs (`memory_store.db`, `memories/holographic/*.db`),
  and **plaintext session snapshots** (`sessions/session_*.json`, `*.jsonl`,
  `request_dump_*.json`) under any Hermes home/root/profile — while keeping the
  shared `memories/MEMORY.md` / `USER.md` readable. `read_file`/`search_files`/
  `patch` deny these (the `.db` files would otherwise be caught by the
  pre-existing binary-extension guard, but the matcher runs first so the
  message says "other users' data" rather than "use terminal").
- **Detect (all three tools):** `agent/data_access_audit.record_access` appends
  JSONL to `${HERMES_HOME}/audit/data-access.log`; `terminal`/`code_execution`
  log (never block) when a command/script references a protected path. Config
  under the top-level `data_access_audit:` block (`enabled`, default true;
  `path`). **Not a security boundary** — a determined admin with a shell
  bypasses it; the audit catches accidental/operator access and makes casual
  admin access visible. The log is local and same-uid-writable.

### Automation ownership — [agent/automation_ownership.py](agent/automation_ownership.py)

Records who owns each user-built automation (skill, cron, script, automation
bundle) so a teammate editing another's work hits a code-enforced confirmation
gate, the owner is DM'd on a confirmed edit, and unowned legacy items prompt a
claim (design:
[docs/superpowers/specs/2026-06-30-automation-ownership-design.md](docs/superpowers/specs/2026-06-30-automation-ownership-design.md)).

- **Registry (canonical):** `${HERMES_HOME}/ownership/registry.json`, keyed
  `skill:<name>` / `cron:<job_id>` / `script:<relpath>` / `automation:<bundle>`,
  each `{owner, collaborators, source}`. Atomic writes, profile-aware.
- **Gate (soft, code-enforced):** the `skill_manage`, `cronjob`, and
  `write_file`/`patch` tools call `check_edit` before mutating. A non-owner edit
  is refused until re-invoked with `confirm_cross_user_owner="<owner>"`; **admins
  are not exempt**; autonomous (no-identity) edits of owned items are refused.
  Creating an automation registers the creator as owner. `record_and_notify`
  DMs the owner (via `send_message`) and logs an `automation_edit` event to the
  `data_access_audit` trail. Config under the top-level `automation_ownership:`
  block (`enabled`, default true; `notify_owner`; `registry_path`). **Not a
  security boundary** — RBAC ([gateway/tool_access.py](gateway/tool_access.py))
  remains the real boundary; this is awareness + collaboration.
- **Always-on guidance:** `AUTOMATION_OWNERSHIP_GUIDANCE` is appended to the
  cached `stable` system-prompt segment ([agent/system_prompt.py](agent/system_prompt.py))
  when enabled + an editing tool is available — present every turn, no `SOUL.md` edit.
- **`ownership` tool (`ownership` toolset, a FLOOR toolset)** —
  [tools/ownership_tool.py](tools/ownership_tool.py). The registry used to be
  administered only via the CLI below, which needs `terminal` — admin-only — so
  the claim nudge told ordinary users to run a command they can't run, and an
  owner couldn't hand off their own work. This tool exposes
  `list | show | claim | transfer | collab_add | collab_remove` to any valid-role
  user. The acting identity comes **only** from session contextvars (no `--user`
  equivalent, so nobody can act as someone else); `claim` requires the item to be
  unowned; `transfer`/`collab_*` require owner-or-admin, enforced in
  `automation_ownership._require_owner_or_admin` so every surface shares it
  (this closed a hole where collaborator edits were unchecked). Admin =
  `"*" in policy.grant_for(...)`, fail-closed — **RBAC off means nobody is admin**
  and the CLI is the override. No identity (cron/delegation) → every action
  refused, so the floor grant is inert under the cron ceiling. **Deployment:** the
  floor grant only intersects with what the platform offers, so `ownership` must
  also be listed in `platform_toolsets.slack` (an explicit list shadows defaults)
  or the tool is silently missing. Name-based targeting ("transfer to Bob") needs
  `slack.user_names`, which [gateway/config.py](gateway/config.py) now bridges into
  the platform `extra` alongside `user_roles`. Design:
  [docs/superpowers/specs/2026-08-03-automation-ownership-tool-design.md](docs/superpowers/specs/2026-08-03-automation-ownership-tool-design.md).
- **CLI:** `hermes own list|claim|transfer|collab|init` ([hermes_cli/own.py](hermes_cli/own.py));
  `init` scaffolds an optional `${HERMES_HOME}/automations/<name>/` bundle, and is
  still the only way to do so. A shell caller is treated as an admin.

### Cron RBAC toolset ceiling — [cron/rbac_ceiling.py](cron/rbac_ceiling.py)

Extends RBAC to **cron execution**, the one surface the three enforcement points
miss: the scheduler runs jobs with **no platform identity** in session contextvars
(it clears them on purpose — [cron/scheduler.py](cron/scheduler.py)), so the
`pre_tool_call` backstop fails open (`denial_for_current_tool` returns early on
empty identity), and a job's `enabled_toolsets` would otherwise be the **sole**
determinant of the agent's capabilities — letting a non-admin who can author a job
(a custom role granting `cronjob`, a `channel_roles` grant, or a direct `jobs.json`
edit) escalate to `terminal`/host-shell. Design:
[docs/superpowers/specs/2026-06-30-cron-rbac-toolset-ceiling-design.md](docs/superpowers/specs/2026-06-30-cron-rbac-toolset-ceiling-design.md).

- **Runtime ceiling (the hard control):** `_cron_enabled_toolsets_with_ceiling`
  ([cron/scheduler.py](cron/scheduler.py)) wraps `_resolve_cron_enabled_toolsets` at
  the single `AIAgent` construction. `cron_owner_grant` reads the creator from the
  **automation-ownership registry** (`cron:<job_id>`; see the Automation ownership
  section above), resolves their *current* role via `ToolAccessPolicy`, and
  `apply_cron_toolset_ceiling` intersects the job's toolsets with that grant. The
  role is re-resolved each run, so a later demotion shrinks the ceiling. An **empty
  cap (`[]`) does NOT fall back to the full default** (`model_tools.get_tool_definitions`
  keys on `enabled_toolsets is not None`); an **unset** `enabled_toolsets` expands to
  `get_all_toolsets()` *then* caps (no silent full-default escalation). Delegated
  sub-agents inherit the cap for free — `delegate_task` intersects a child's toolsets
  with the parent's ([tools/delegate_tool.py](tools/delegate_tool.py)).
- **Create-time gate (fail-fast + the script gate):** the `cronjob` tool's
  `_rbac_creation_error` ([tools/cronjob_tools.py](tools/cronjob_tools.py)) rejects
  `enabled_toolsets` the creator's role can't grant, and rejects `no_agent=True` **or**
  a `script` field unless the role grants `terminal`/`code_execution` (running a
  script is shell-equivalent) — on **create and update**, before the job is persisted
  or the owner notified. This is the ONLY enforcement for `no_agent`/`script` jobs
  (they never build an agent to cap).
- **Ownerless/roleless jobs run UNCAPPED** (fail-open — deliberate operator decision
  so legacy/CLI-created jobs keep working). Surfaced via a `data_access_audit`
  `ownerless-elevated` line — but only when RBAC is genuinely active
  (`tool_access.rbac_active_anywhere`), so installs that never opted into RBAC stay
  **byte-for-byte unchanged** (no new audit writes).
- **No new config block.** Keyed to RBAC activation (`user_roles`) +
  `automation_ownership.enabled`. Fail-open on any internal error (the cap is the
  primary cron control, not a backstop). **Not a boundary against a determined admin
  with a shell** — RBAC + the `file`-toolset grant remain the real boundary for the
  `jobs.json`-edit path; this closes the toolset-driven escalation and makes the
  residual visible.

### Cron capability preflight + local-job failure alerts — [cron/capability_preflight.py](cron/capability_preflight.py)

Every step between a job's `enabled_toolsets` and what its agent receives is
silent: enabled MCP servers are merged in, `messaging`/`clarify` (usually
`cronjob`) are stripped, the owner's role caps the rest. The Ready-for-Staging
auto-merger (2026-09-16) was created as `[jira, messaging, file]` by a creator
whose role had no write access, ran `ok` every 5 min for a week, and never
merged or reported anything. Design:
[docs/superpowers/specs/2026-09-23-cron-capability-preflight-design.md](docs/superpowers/specs/2026-09-23-cron-capability-preflight-design.md).

- **Cron agents never get a shell (fork policy — `terminal` is super-admin
  only).** `_resolve_cron_disabled_toolsets` strips `terminal` and
  `code_execution` from every cron run unless `cron.allow_agent_shell: true`.
  Deterministic work goes in scripts a super admin attaches (the create gate
  requires a `terminal`/`code_execution` role for either): the pre-run
  `script` collects, and the new **`post_script`** applies — it runs after
  each successful agent run with the agent's final response in the file named
  by `$HERMES_CRON_RESPONSE_FILE` (cwd = `workdir`), its stdout is appended to
  the delivery (a `[SILENT]` agent + empty stdout stays silent), and a
  non-zero exit fails the run. The pattern: agent writes a plan/summary with
  the `file` tools → `post_script` validates and applies it. `hermes cron
  create/edit --post-script`.
- **One evaluation, four surfaces.** `evaluate_job_capabilities` computes the
  effective toolsets (same resolver + ceiling as the scheduler) and checks the
  *needed* ones: the job's new `required_toolsets` field, `requires_toolsets`
  in the `automation.yaml` of the job's `workdir` (scaffolded by `hermes own
  init`), and every explicitly listed `enabled_toolsets` name (measured on the
  VM: flagged only the genuinely broken jobs). A requirement fails when it is unknown, stripped in
  cron (hint: `messaging` → `slack_post`), outside the owner's role, not in the
  job's list, or (create/update only) exposes no usable tool on this host.
  MCP names fold `mcp-<server>` onto the bare server name.
- **`cronjob` create/update REJECTS** an unmet requirement (the error tells
  the agent to report who must grant it, never to drop it). Update evaluates
  against the job's *owner*, and only enforces when it touches
  toolsets/requirements/workdir/acks — an unrelated edit of a legacy job gets
  `capabilities.warnings` instead. Results echo `capabilities.effective_toolsets`
  and `blocked_unattended_tools` (approval-gated tools reachable but not acked,
  which the headless approval gate will block — the existing ack check only saw
  explicitly listed toolsets, not merged-in MCP servers).
- **Runtime:** a 4th check in the scheduler's pre-dispatch preflight
  (`_preflight_check_capabilities`) blocks a run as `blocked_config`
  (alert-once, no LLM spend) when a needed toolset stops resolving —
  demotion, transfer to a narrower role, config change.
- **Script imports are checked against the gateway's Python.** Cron scripts
  run under `sys.executable` (the service's `.venv`), not the Python the
  author tested with — on the VM `hermes`/`python3` resolve elsewhere, so a
  collector worked in testing and died on `ImportError` in cron (MRR,
  2026-09-18..23). `script_import_problems` AST-parses `script`/`post_script`
  (skipping `try`-guarded and sibling imports) and `find_spec`s each module:
  `cronjob` create/update rejects, and `run_job` blocks BEFORE any script
  executes (`_blocked_config_result`, alert-once) with the exact `uv pip
  install --python <gateway python>` fix.
- **`ownership transfer`** of a `cron:` item is REFUSED when the new owner's
  role would strip what the job needs, unless `confirm_capability_loss=true`
  (the 2026-09-14 MRR transfer silently removed `terminal`); the result also
  echoes `capabilities` under the new owner.
- **`[FAILED]` marker:** a cron response starting with `[FAILED]` marks the run
  failed (`failure_streak`, incident, failure alert). The cron system hint
  tells agents to use it when a required action was missing/refused/blocked.
- **Local-only jobs DM their owner:** a failure alert for a job with no
  delivery target (`deliver: local`, or an origin that no longer resolves) is
  sent to the owner (ownership registry `cron:<id>`) as a DM, once per incident
  signature (the incident is then `alerted`; a new error text alerts again).
  `blocked_config`/drift alerts keep their own alert-once markers.

### Multiplex — one gateway, several bots ([design](docs/superpowers/specs/2026-11-16-multiplex-fork-hardening-design.md))

Under `gateway.multiplex_profiles: true` one process serves several profiles
(the default + e.g. `educator`, each its own Slack app). Upstream's
`_profile_runtime_scope` gives each turn its profile's home and secret scope,
but **`os.environ` stays the default profile's** — so any fork code that reads
`os.getenv`, `self.adapters[...]` or a path-less cache acts as the default bot.
Rules for fork code that reads credentials or platform settings:

- Credentials: `agent.fork_secrets.read_secret(name, tool=...)` (or
  `agent.secret_scope.get_secret`), never `os.getenv`. A scoped miss is a miss;
  `UnscopedSecretError` → WARNING + "not configured". No `.env` self-heal under
  multiplex (`heal_dotenv`). Child processes: `scoped_subprocess_env()`.
- Platform settings: the adapter's own `extra` / `_slack_setting`, or
  `gateway.authz_mixin._platform_gate_env` — never `os.getenv`. The Slack
  plugin's `_apply_yaml_config` returns the profile's `slack:` block as
  `extra["profile_settings"]` and skips its `os.environ` writes under multiplex.
- The adapter for an inbound source: `self._adapter_for_source(source)` /
  `_multiplex_adapter_for`, never `self.adapters[...]`; per-source config:
  `_gateway_config_for_source` (slash admins, ignored channels).
- Caches of per-profile data are keyed by path/home (`_load_config_cached`,
  Drive credentials), and module-level `get_hermes_home()` captures resolve per
  call (cron notepad/suggestions, auxiliary `auth.json`).
- Guards ([gateway/multiplex_guard.py](gateway/multiplex_guard.py),
  [agent/multiplex_file_guard.py](agent/multiplex_file_guard.py)): a secondary
  profile gets **no** MCP toolset (MCP processes are process-wide and keyed by
  server name, so even a server it declares would be the default's) —
  resolver + cron + fail-closed dispatch backstop; file tools (including every
  `search_files` hit) deny paths in another **served** profile's home. Opt-in
  `agent_write_guard: true` ([agent/profile_write_guard.py](agent/profile_write_guard.py))
  stops a profile's agent writing its own config/SOUL/cron/scripts/.env*/external
  skill dirs.
- Every multiplex branch is gated on `is_multiplex_active()`; with it off the
  legacy code path runs unchanged (`.env` still beats `config.yaml` for the
  Slack keys that were env-only). Under multiplex `config.yaml` wins; startup
  WARNs per profile for each env-only key the two disagree on. Guards fail
  closed under multiplex (a failed scoped config load → deny-all slash policy).
  Cron child processes (scripts, Bot Chat delivery) get the owning profile's
  env and `-p`. Acceptance test:
  [tests/gateway/test_multiplex_fork_features.py](tests/gateway/test_multiplex_fork_features.py);
  cron: [tests/cron/test_multiplex_cron_per_profile.py](tests/cron/test_multiplex_cron_per_profile.py).
- Known limitation: `terminal.cwd` is bridged once from the default's config.

### Cron delivery format — [cron/scheduler.py](cron/scheduler.py) `_deliver_result`

Upstream wraps every delivered cron response in `Cronjob Response: <name>` /
`(job_id: …)` / a dashed divider / the body / a "To stop or manage this job…"
footer. In an automation-feed channel that is five lines of noise around each
item, so the fork's wrapper is **`**<name>**`, a blank line, the response —
nothing else**. The bold is standard markdown so each platform's outbound
formatter renders it natively (the Slack lane converts it to mrkdwn `*name*`;
under `slack.rich_blocks` / `slack.markdown_blocks` it renders inside Block Kit). The
`cron.wrap_response: false` knob still yields the raw response. The Yuanbao
`strip_cron_wrapper` was retargeted to the new shape. Guarded in
`tests/test_fork_feature_inventory.py`. No design doc.

### Slack per-channel app admission — `slack.allow_bots_channels`

Upstream's `slack.allow_bots` (`none` | `mentions` | `all`) is workspace-wide,
so waking the agent on one automation feed (HubSpot's "New meeting booked!"
post in a leads channel) meant opening every channel to every app. The fork
adds **`slack.allow_bots_channels`** — a comma list / YAML list of channel
IDs, env `SLACK_ALLOW_BOTS_CHANNELS`, same shape as `quiet_channels` — in
which app/bot posts are admitted as `all` while the global policy stays
`none` elsewhere. Two layers agree on the one list:

- **Adapter** ([plugins/platforms/slack/adapter.py](plugins/platforms/slack/adapter.py)):
  `_slack_allow_bots(channel_id)` returns `all` for a listed channel; the
  inbound filter passes the event's channel. Bridged YAML→env in
  `_apply_yaml_config` and into `extra` by [gateway/config.py](gateway/config.py).
- **Gate** ([gateway/authz_mixin.py](gateway/authz_mixin.py)): the `#4466` bot
  bypass also admits a Slack app post whose `chat_id` is listed (read off the
  live adapter's `extra`, env fallback) — necessary because an app post has
  `user=None` and would otherwise die at the no-user-id guard before RBAC
  ever saw it. **And under RBAC the bypass no longer outranks roles**: an
  admitted app poster is authorized only if `channel_roles` names its channel
  (`policy.is_authorized(None, chat_id)`), which is also what equips it — a
  grant-less poster resolves to zero toolsets. Previously `SLACK_ALLOW_BOTS=all`
  authorized any app anywhere ahead of RBAC. Without RBAC the legacy
  behaviour is unchanged.

Wiring an app feed therefore takes four keys on the same channel:
`allow_bots_channels` (admit), `free_response_channels` (wake without a
mention), `quiet_channels` + `relevance_gate_purpose` (judge each post),
`channel_roles` (authorize + equip, least privilege — anyone who can post
there gets that role). Tests:
[tests/gateway/test_slack_allow_bots_channels.py](tests/gateway/test_slack_allow_bots_channels.py).
### Cost attribution — `hermes costs` + `pricing.overrides`

Upstream records tokens and cost per session (`sessions`, `session_model_usage`) but
only reports by model/platform. The fork adds
[agent/cost_attribution.py](agent/cost_attribution.py), a read-side query layer that
walks every session to its `parent_session_id` root and derives two keys at query
time — the **cron job** from the `cron_<job>_<ts>` root id, and the **channel** from
`origin_json` (`parent_chat_id` else `chat_id`; DMs key as `dm:<user>`); a cron job
delivering to a channel carries both, via `cron.scheduler._resolve_delivery_targets`.
No schema change. **Each contributing session is windowed and bucketed on its own
`started_at`; the root supplies only the keys** — a Slack channel's root is permanent
(`session_reset.mode: none`) and soon months old, so windowing on it would make
recent spend vanish from a 30-day report. `hermes costs --days 30 --by
channel|job|both|model|user --bucket day|week|month [--json|--csv]`
([hermes_cli/subcommands/costs.py](hermes_cli/subcommands/costs.py)). The `channel`
view double-counts multi-target jobs on purpose (footer shows the overlap); `job` and
`both` partition true spend. Every row carries a status (`actual`/`estimated`/
`partial`/`unpriced`) plus `unpriced_tokens`, so a missing price is never a silent $0.
`--by model` reads the `session_model_usage` rows, so a legacy session with no usage
rows counts in the session views but not in the model view (and that view shows no
session count at all — one session spans several models).
**The VM's `gpt-5.4-mini` is not in upstream's pricing catalog**, so set per-million
rates under a top-level `pricing.overrides:` block (`{input, output, cache_read,
cache_write}`); `get_pricing_entry` consults it before the catalog (`cost_source =
user_override`). `hermes costs --reprice [--dry-run]` (no window = the whole store)
prices stored rows whose `cost_status` is unknown **and recomputes every non-`actual`
row of a model that has an override, whatever its old `cost_source`** — an override is
the operator's statement of the rate and governs that model's whole history (the
built-in catalog is a snapshot that goes stale: its gpt-5.6-terra entry sat 25% above
OpenAI's bill after a price cut), so the recompute is exact, idempotent, and repairs
both stale catalog rows and a session that straddled the config change. Provider-
reported `actual` rows are never rewritten. Give each override `cache_write` too where
the model bills it (OpenAI gpt-5.6: 1.25× input) — it defaults to 0.
**Deploy the rates as: stop the gateway → add `pricing.overrides` → `hermes costs
--reprice` → start the gateway**, or a session that is live across the change stays
under-priced (its usage row flips to `estimated` carrying only the first priced
call's cost while keeping all the earlier unpriced tokens). The report opens
the store `SessionDB(read_only=True)` (avoids writer-lock contention with the live
gateway) and never migrates it; a store not write-opened since a past schema
migration prints a hint naming `hermes costs --reprice --dry-run` (a write-open that
migrates but changes no cost) and exits 1. Design:
[docs/superpowers/specs/2026-09-17-cost-attribution-design.md](docs/superpowers/specs/2026-09-17-cost-attribution-design.md).

### Cost budget + alerts — [agent/cost_budget.py](agent/cost_budget.py)

Opt-in per profile through a top-level `cost_budget:` block in that profile's
`config.yaml`. **Without the block nothing changes**: every entry point
returns after a cached config read — no store opened, no state file, no ledger
difference (the `cost_budget_status` tool then reports `configured: false`
and reads nothing).

```yaml
cost_budget:
  daily_usd: 5                    # any of daily/monthly/per_run activates the block
  monthly_usd: 100
  alert_at: [0.5, 0.8]            # fractions (0.8 = 80%); 1.0 is always added; >5 is rejected
  per_run_usd: 1.5                # cron runs only (a cron root + its lineage)
  alert_channel: slack:C0123ABCD  # deliver-target syntax; a bare C… id means slack:; local/origin rejected
  alert_mention: "<!subteam^S123>"  # optional, prefixed to LOUD alerts only
  discretionary_jobs: [3f9a1c2d4e5b]  # job ids only; or `discretionary: true` on the job
  default_discretionary: false    # true = EVERY job is skippable unless listed below
  essential_jobs: [a1b2c3d4e5f6]  # job ids that always run (win over every other marker)
  cache_seconds: 300              # spend snapshot TTL (min 30)
```

- **Spend is booked when it happens.** `session_model_usage` rows are
  cumulative upserts with no per-call timestamp, so windowing on
  `sessions.started_at` (what `hermes costs` does) would miss a long-lived
  session — a Slack channel root never resets — that keeps spending: a fail-
  open budget. Each scan reads the usage rows touched since the last scan,
  books the growth of each row's cumulative cost to the local day of its
  `last_seen`, and keeps the per-row values and per-day totals in
  `${HERMES_HOME}/cost_budget/spend.json` (flock + atomic write; 40-day
  retention). The **first scan** baselines rows last seen before this month
  (no booking) and books rows active this month whole, flagging those days
  **upper bound** (`spend_is_upper_bound`) when the row began earlier — fail
  closed. A session idle beyond the retention that revives is booked whole the
  same way. Unpriced tokens are never $0: a daily alert, `spend_is_lower_bound`,
  and the tool returns `remaining_usd: null`. Resolved through
  `get_hermes_home()` per call (one multiplexed gateway ticks several homes),
  cached per home; a failed scan is cached for 60s (`SpendUnavailable`) so a
  locked store is not re-queried every preflight. **Prerequisite:** a
  `pricing.overrides` entry for the profile's model. **Known limit:** day and
  month boundaries use `hermes_time`'s process-wide timezone, not a
  per-profile one.
- **Enforcement** — `preflight_cron_job`, called from `_run_one_job_body`
  ([cron/scheduler.py](cron/scheduler.py)) after the secret scope is set and
  before `run_job`, so before the job's script *or* agent. A discretionary
  job over its daily or monthly budget is skipped via
  `cron.jobs.mark_job_skipped`, which is **not a run**: `last_status =
  skipped_budget` + `last_skip_reason`, but `failure_streak`, `last_error`
  and `repeat.completed` are untouched; a recurring job moves to its next
  occurrence, and a one-shot gets its `claim_dispatch` pre-claim back and
  re-arms when the exceeded budget resets. The ledger row is `completed`
  with `error = "skipped: over budget (daily $X of $Y)"`
  (`finish_execution(note=…)`, [cron/executions.py](cron/executions.py)); a
  lost fire claim is recorded as such, like the normal path. Non-
  discretionary jobs run. No bypass for manual runs: the agent could
  otherwise `cronjob(action='run')` around its own budget. The `cronjob`
  tool takes `discretionary: true` (set-only — clearing it frees a job from
  the budget, so that is an operator edit of `jobs.json` or the config list).
  **Fail-closed mode (`default_discretionary: true`, opt-in):** every job of
  the profile is discretionary unless its id is in `essential_jobs` — so a
  job the agent creates without the flag still cannot escape the budget.
  Both keys live only in `config.yaml`; the `cronjob` tool has no way to mark
  a job essential, and a job-level `discretionary: false`/`essential` field is
  ignored. With the profile write guard (`agent_write_guard`) on, the agent
  cannot edit config.yaml either. An essential id wins over every other
  marker. Default `false` keeps the list/flag behaviour above. The educator
  profile uses `default_discretionary: true` with `essential_jobs:
  [<release-watch job id>]`.
- **Alerts** — `watch()` from `tick()` after the dispatch gate, and every
  over-budget preflight. `tick` already runs inside each profile's home
  override with **that profile's own adapters** (the multiplex ticker), so
  alerts leave through the owning profile's bot. Sent through
  `cron.scheduler._deliver_result` with a pseudo job (`id: cost_budget`, so
  the bold header reads "Cost budget"), after checking the target resolves —
  an alert that went nowhere is not marked sent. Keys (in
  `${HERMES_HOME}/cost_budget/alerts.json`): `daily:<day>:<t>` /
  `monthly:<month>:<t>` — a jump past several thresholds sends only the
  highest, and `…:1` is **the** over-budget alert for the period (loud, says
  discretionary jobs are skipped, names the job that tripped it whether
  `watch` or a preflight got there first); `run:<root>` — per-run
  overspends since tracking began, batched into one message per check;
  `unpriced:<day>`. Keys are **claimed under the lock and delivered after it
  is released, on a daemon thread that carries the caller's context** (home
  override) and adapters — neither the state lock nor the tick lock is held
  across a network send; a failed send releases its claim and retries later.
  No usable `alert_channel` → logged once per process, not recorded.
- **`cost_budget_status` tool (`cost_budget` toolset, NOT a floor)** —
  [tools/cost_budget_tool.py](tools/cost_budget_tool.py): today/month spent,
  budget, remaining (null when unpriced tokens make it unknown), over-budget,
  skip state, lower/upper-bound flags, `config_warnings`, the month's five
  costliest cron runs. **Deployment:** grant `cost_budget` to the role, list
  it in `platform_toolsets.slack`, and for cron use give the job an owner
  whose role grants it (the cron RBAC ceiling).

Design: "Dependencies → 3. Cost budget + alerts" in
[docs/superpowers/specs/2026-11-16-product-education-teammate-design.md](docs/superpowers/specs/2026-11-16-product-education-teammate-design.md).
Plan:
[docs/superpowers/plans/2026-11-16-cost-budget-alerts.md](docs/superpowers/plans/2026-11-16-cost-budget-alerts.md).

### Slack quiet channels + `slack_react`

For low-noise "hidden assistant" channels. Config under the top-level `slack:`
block in `~/.hermes/config.yaml`:

- `quiet_channels: 'C123,C456'` — comma-separated channel IDs (mirrors
  `free_response_channels`). In these channels the gateway forces
  `tool_progress: off` and allows **silent (emoji-only) completion**: a
  successful turn that produces no text stays silent instead of posting the
  "no response generated" warning. Resolved in [gateway/run.py](gateway/run.py)
  via `_is_quiet_channel` (matches `chat_id` or thread `parent_chat_id`); the
  adapter-side half lives in [plugins/platforms/slack/adapter.py](plugins/platforms/slack/adapter.py)
  (`_slack_quiet_channels`).
  Errors, approvals, and clarifications still surface. **Text replies are never
  suppressed** — the bot can always answer. A message that arrives **mid-task** in a
  quiet channel is silently **queued** (forced `queue` mode, with the busy-ack text and
  the one-time `/busy` onboarding tip suppressed) instead of interrupting; `/stop` and
  `/new` still force-cancel normally.
- `slack_react` tool ([tools/slack_react_tool.py](tools/slack_react_tool.py)) —
  lets the agent add/remove an emoji reaction on the triggering Slack message
  (or an explicit `message_id`). Targets the message via session contextvars
  (`HERMES_SESSION_MESSAGE_ID`, set in
  [the Slack adapter](plugins/platforms/slack/adapter.py) via
  `build_source(message_id=ts)`). Lives in the platform-restricted `slack`
  toolset; grant it to a role via `slack.roles`. Emoji-first behavior is driven
  by `channel_prompts`, not enforced.
- `turn_end` tool ([tools/slack_react_tool.py](tools/slack_react_tool.py)) —
  a **terminal** tool: the agent calls it as its final action (e.g. after
  `slack_react`) to finish a turn silently with no text. The conversation loop
  ([agent/conversation_loop.py](agent/conversation_loop.py)
  `_called_terminal_turn_end`) ends the turn on this tool call **only when
  `_silent_completion_ok` is set** — the gateway sets that per-turn for quiet
  channels. Ending on a tool call (not an empty text response) is what avoids
  the loop's empty-response recovery nets (post-tool nudge / retry / fallback)
  that would otherwise force unwanted text. Inert (benign no-op ack) outside
  quiet channels. Drive it via `channel_prompts`: "react, then call `turn_end`."
  Safety net: even when the model forgets `turn_end`, an empty/no-content turn in
  a quiet channel is accepted silently (`_should_accept_silent_empty` in
  [agent/conversation_loop.py](agent/conversation_loop.py)) — the empty-response
  recovery (prior-content fallback / nudge / retry / provider fallback) is
  skipped when `_silent_completion_ok` is set. Thinking-only responses still
  prefill-continue. **Text replies are never suppressed** — only empty turns.
  Same for the codex/Responses path: a gpt-5.x turn that yields only reasoning
  (no text, no tool call) is classified `incomplete` and continued up to 3×; when
  that's exhausted in a quiet channel, `_codex_incomplete_exhausted_result`
  ([agent/conversation_loop.py](agent/conversation_loop.py)) finishes silently
  instead of posting "Codex response remained incomplete…" (the model chose to
  stay silent). Non-quiet channels keep the warning.
- **Relevance pre-gate** (default for `quiet_channels`): before the full agent
  runs, a cheap classifier (`slack.relevance_gate_model`, empty = main model)
  decides act/ignore on each **non-@mention** message; `ignore` ends the turn
  with no agent run. Lives in [gateway/run.py](gateway/run.py)
  (`_relevance_gate_should_skip` → `_classify_relevance` via `async_call_llm`),
  invoked in `_handle_message` after `pre_gateway_dispatch`. @mention/DM bypass
  it (`MessageEvent.directly_addressed`, set in
  [the Slack adapter](plugins/platforms/slack/adapter.py)).
  Purpose per channel: `slack.relevance_gate_purpose[chat_id]` →
  `channel_prompts[chat_id]` → a generic default. **Fail-open**: classifier
  error → the agent runs (never silently drops a real message). Only active on
  Slack quiet channels; inert elsewhere.
  The classifier runs before any agent turn, where auxiliary usage is normally
  discarded, so `_handle_message` publishes an aux accounting context bound to
  the channel's session around the gate (`_relevance_gate_accounting_token`);
  its tokens land in `session_model_usage` under task `relevance_gate` and
  `hermes costs` attributes them to the channel. Before this (2026-09-22) they
  were invisible — part of an ~8% gap against the OpenAI bill.

### Per-channel / per-task model overrides

Route work to different models without changing the global `model.default`.
All surfaces share one entry shape — a model string OR
`{model, provider, base_url}` (credentials always resolved host-side; never
put API keys in entries). Shared helpers: [agent/model_override.py](agent/model_override.py).
Design doc: [docs/superpowers/specs/2026-06-10-model-overrides-design.md](docs/superpowers/specs/2026-06-10-model-overrides-design.md).

- **`slack.channel_models`** (top-level `slack:` block): `chat_id → entry`;
  threads inherit via `parent_chat_id`. Applied in
  `_apply_channel_model_override` ([gateway/run.py](gateway/run.py)).
  Precedence: session `/model` > `channel_models` > global default.
  Fail-open — a bad entry logs and falls back to the global model.
- **Skill frontmatter** (`metadata.hermes.{model,provider,base_url}` in
  SKILL.md): invoking the skill (slash command or channel auto-skill)
  switches the session's model for the rest of the session, exactly like
  `/model` (writes `_session_model_overrides`, evicts the cached agent).
  Last writer wins; cleared by `/new`/`/reset`/restart. CLI sessions, skill
  bundles, and plugin skills are out of scope (v1).
- **Cron**: jobs already carry per-job `model`/`provider`/`base_url`;
  additionally a skill listed in `job.skills` fills any of those fields the
  job leaves unset (`_effective_job_model_fields` in
  [cron/scheduler.py](cron/scheduler.py)). Job fields always win.
- **Delegation**: `delegate_task` accepts `model`/`provider`/`base_url`
  (top-level and per-task) so sub-agents can run on cheaper/stronger models;
  omitted = inherit parent ([tools/delegate_tool.py](tools/delegate_tool.py)).

## Common commands

```bash
# Setup (fork dev)
./setup-hermes.sh                         # installs uv, venv, .[all], symlinks hermes
source .venv/bin/activate                 # or: source venv/bin/activate

# Tests — ALWAYS use the wrapper, never bare pytest (CI-parity: unset creds, TZ=UTC,
# C.UTF-8, xdist, per-test subprocess isolation). See AGENTS.md "Testing".
# NEVER run the whole suite in one invocation — see the rule below.
scripts/run_tests.sh tests/gateway/                    # one package (do this)
scripts/run_tests.sh tests/gateway/test_tool_access.py # one file
scripts/run_tests.sh tests/gateway/test_tool_access.py::test_x  # one test
scripts/run_tests.sh --no-isolate tests/foo/           # faster, for debugging
scripts/run_tests.sh -v --tb=long                      # pass-through pytest flags

# Lint / typecheck (ruff is intentionally near-disabled — only PLW1514 is enforced)
ruff check .
ty check                                  # type checker (configured in pyproject.toml)

# Run the agent locally
hermes                                    # interactive CLI
hermes --tui                              # Ink/React TUI (or HERMES_TUI=1)
hermes gateway                            # messaging gateway (Slack, etc.)
hermes doctor                             # diagnose issues

# TUI dev (see AGENTS.md "TUI Architecture")
cd ui-tui && npm run dev                  # watch mode
```

### Testing rule: NEVER run the full suite in one invocation

`scripts/run_tests.sh` with no path argument fans out ~3,300 files over 24
workers and pins every core on the machine for ~25 minutes. It has deadlocked a
developer machine. **Do not run it.** There is no flag that makes it safe — cap
the SCOPE, not the parallelism.

Run one package at a time, and wait for each to finish before starting the next:

```bash
scripts/run_tests.sh tests/agent/
scripts/run_tests.sh tests/gateway/
scripts/run_tests.sh tests/hermes_cli/
scripts/run_tests.sh tests/tools/
scripts/run_tests.sh tests/cron/ tests/hermes_state/ tests/plugins/
scripts/run_tests.sh tests/cli/ tests/run_agent/ tests/tui_gateway/
```

The four big packages (`hermes_cli` 717 files, `gateway` 701, `tools` 523,
`agent` 433) are each big enough to run alone. Small ones can be grouped.
Never run a package batch in the background while another is running.

When verifying a change, prefer the narrowest scope that covers it — the
specific files, then the package. A full-suite sweep is the user's call, not
something to launch unprompted.


## Key invariants (do not break)

- **Prompt caching:** never alter past context, swap toolsets, or rebuild system
  prompts mid-conversation (only context compression may). Cache-mutating slash
  commands default to deferred invalidation with an opt-in `--now`. (AGENTS.md)
- **Profiles:** use `get_hermes_home()` / `display_hermes_home()` from
  `hermes_constants` — never hardcode `~/.hermes`. (AGENTS.md)
- **Dependency pinning:** direct deps are exact-pinned in
  [pyproject.toml](pyproject.toml); regenerate `uv.lock` with `uv lock` on changes.
  Provider/backend-specific deps go in extras + lazy-install, not core. (AGENTS.md)
- **Tests must not write to `~/.hermes/`** (autouse fixture redirects `HERMES_HOME`);
  don't write change-detector tests. (AGENTS.md)
- **RBAC fail-closed:** when editing the policy or its enforcement points, keep the
  deny-until-assigned semantics and the activation boundary (empty `user_roles` =
  upstream behavior) intact.

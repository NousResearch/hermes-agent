# Linear Agent Platform

This plugin lets a Hermes gateway profile appear in Linear as a Linear Agent. It receives Linear Agent Session webhooks and sends Hermes responses back as Linear Agent Activities.

## Recommended OAuth setup

For Linear Agent/app attribution, use Linear OAuth 2.0 client credentials. Hermes stores the Linear OAuth app client ID/secret in the active profile `.env`, mints an app-actor access token with `grant_type=client_credentials`, caches that token (30-day lifetime) in the profile `auth.json`, and gets a new token before expiry or after a Linear GraphQL 401. No browser flow or refresh token is required.

Fastest path: `hermes gateway setup` → **Linear Agent** — the wizard prompts for credentials, verifies them against Linear, auto-detects the app user ID, and builds the allowlist interactively. Manual steps below.

1. In Linear, create or open the OAuth application for the agent and enable/use client credentials for the app.

2. Add the OAuth app credentials to the profile `.env`:

   ```bash
   LINEAR_AGENT_CLIENT_ID=...
   LINEAR_AGENT_CLIENT_SECRET=...
   LINEAR_AGENT_OAUTH_SCOPES=read,write
   ```

3. Optional one-time mint/test from the Hermes repo:

   ```bash
   cd ~/.hermes/hermes-agent
   venv/bin/python -m plugins.platforms.linear_agent.oauth \
     --profile <profile> \
     --client-credentials
   ```

   The helper reads `LINEAR_AGENT_CLIENT_ID` and `LINEAR_AGENT_CLIENT_SECRET` from the profile `.env`, writes the cached access token/expiry to `auth.json`, and does not print token values. Runtime token reissue works from the client ID/secret even if the cached token is absent.

4. Restart the profile gateway:

   ```bash
   hermes gateway restart
   ```

At runtime, the adapter uses client credentials first, reads and persists the current access token/expiry in the profile `auth.json` by default, and retries once after a Linear GraphQL 401. Set `LINEAR_AGENT_PERSIST_TOKENS=false` to disable access token persistence.

### Legacy browser OAuth setup

The older authorization-code + localhost callback flow is still available as a fallback when `LINEAR_AGENT_REFRESH_TOKEN` is configured. Run the same helper without `--client-credentials` to exchange a browser authorization code and cache rotating access/refresh tokens in `auth.json`.

## Manual/static token setup

If you already have a valid Linear app/OAuth token, you can still set secrets manually in the profile `.env`:

```bash
LINEAR_AGENT_ACCESS_TOKEN=lin_oauth_or_app_token
LINEAR_AGENT_WEBHOOK_SECRET=linear_webhook_signing_secret
```

Optional:

```bash
LINEAR_AGENT_CLIENT_ID=
LINEAR_AGENT_CLIENT_SECRET=
LINEAR_AGENT_REFRESH_TOKEN=
LINEAR_AGENT_TOKEN_EXPIRES_AT=
LINEAR_AGENT_REDIRECT_URI=
LINEAR_AGENT_OAUTH_SCOPES=read,write
LINEAR_AGENT_OAUTH_ACTOR=app
LINEAR_AGENT_APP_USER_ID=
LINEAR_AGENT_WORKSPACE_ID=
LINEAR_AGENT_HOME_TARGET=
LINEAR_AGENT_ALLOWED_USERS=
LINEAR_AGENT_ALLOW_ALL_USERS=
```

Authorization is two-layer and **fail-closed at both layers**: the gateway grants access via `LINEAR_AGENT_ALLOWED_USERS` / `LINEAR_AGENT_ALLOW_ALL_USERS` (no allowlist means every user is denied), and the adapter checks the sender BEFORE any webhook side effect by consulting the gateway-registered authorization chain (the same one dispatch uses — env allowlists including the `*` wildcard, `GATEWAY_ALLOWED_USERS`, DM-pairing grants), with YAML `allowed_users` / `allow_all_users` granting as a union. With nothing configured anywhere, webhooks are rejected 403 and no ack/auto-start/stop activity ever fires. `allowed_teams` narrows further and never grants.

`LINEAR_AGENT_WEBHOOK_SECRET` is effectively required: unsigned webhooks are **rejected** unless you explicitly opt out with `allow_unsigned_webhooks: true` (or `LINEAR_AGENT_ALLOW_UNSIGNED_WEBHOOKS=true`). When the secret is set, incoming webhook requests must include a valid Linear HMAC signature.

`LINEAR_AGENT_HOME_TARGET` enables cron delivery: set it to an issue ID or identifier (e.g. `ENG-123`) and `deliver=linear_agent` cron jobs post their results as comments on that issue — including when cron runs out-of-process from the gateway. Setting it also silences the gateway's one-time "no home channel" notice, which otherwise posts into each new session thread.

## Config

```yaml
linear_agent:
  enabled: true
  webhook_host: 0.0.0.0
  webhook_port: 8651
  webhook_path: /hermes/linear-agent
  # When set, webhooks from any OTHER workspace are ignored (one Linear app
  # can be installed in several workspaces sharing a webhook secret).
  workspace_id: ""
  app_user_id: ""
  allowed_teams: []
  allowed_users: []
  allow_all_users: false
  ack_on_created: true
  # Move an issue delegated to the agent to its first "started" state
  # (Linear best practice). Default true — opt out with false. Requires
  # mutation_policy.update_issues; triage-state issues are left for humans.
  auto_start_on_delegation: true
  # Opt-in: when auto-starting an issue that has NO delegate, also claim it
  # (set the agent as delegate). Default false — humans decide what the
  # agent owns unless you explicitly enable this.
  auto_self_delegate: false
  # Opt-in: dispatch a full agent turn for issue-update webhooks (replies
  # post as issue comments). Default false — delegation already arrives as
  # a real agent session, so update webhooks only feed auto-start.
  dispatch_issue_updates: false
  # Opt-in: when mentioned inside an existing comment thread, also post the
  # findings as a reply on that source comment (so they surface in the human
  # thread, not only the session widget). Requires mutation_policy.create_comments.
  reply_in_source_thread: false
  # Skill names auto-loaded into every Linear session (optional).
  auto_skills: []
  # Webhook body size cap in bytes (default 1 MiB).
  max_body_bytes: 1048576
  # Every write operation fails closed; enable only what you need.
  mutation_policy:
    create_comments: false
    update_comments: false      # editing an existing comment (commentUpdate)
    update_issues: false        # also covers issue relations + URL links
    create_issues: false
    # update_projects also covers project (status) updates, milestones,
    # and initiatives — project-structure mutations share this key.
    update_projects: false
    create_documents: false
    update_documents: false
    create_customer_needs: false
    update_customer_needs: false
    # create_releases/update_releases also gate release NOTES (the
    # release-family umbrella, like update_projects covers status updates).
    create_releases: false
    update_releases: false
    create_customers: false      # customer (business entity) create
    update_customers: false      # customer update
    create_labels: false         # issue label create (team or workspace)
    # Deletes are a separate fail-closed family (never implied by update/create).
    delete_comments: false
    delete_customer_needs: false
    delete_status_updates: false
    delete_attachments: false
    delete_customers: false
```

Keep OAuth client secrets and webhook secrets in `.env`, and keep runtime OAuth access/refresh tokens in `auth.json`. Do not put secrets in `config.yaml` or docs.

## Local Run

```bash
hermes gateway start
```

Point Linear's Agent Session webhook at:

```text
https://your-public-host.example/hermes/linear-agent
```

For local development, expose the configured port with a tunnel and use the tunnel HTTPS URL.

## Notes

- Normal Hermes final responses are sent as Linear `response` activities. For comment-mention sessions Linear mirrors the response into the session's thread; mentioning the agent inside an existing thread makes Linear re-anchor the conversation at a new root comment, so the reply appears there rather than in the original thread.
- New sessions optionally receive a quick `thought` acknowledgement.
- Dispatch failures are reported as Linear `error` activities.
- All write tools (issues, comments, projects, status updates, milestones, initiatives, documents, customer needs, releases) are gated by `mutation_policy` and fail closed by default. Agent-session activities (responses/thoughts/errors) are the core protocol and are not gated.
- Set `LINEAR_AGENT_APP_USER_ID` (or `app_user_id`) so issue-update webhooks triggered by the agent's own mutations are ignored — without it, every write the agent makes echoes back as a new session. When unset, the adapter auto-discovers the id from Linear's `viewer` query at connect (best-effort backstop; setting it explicitly is still recommended).
- Issue-update webhooks feed auto-start only by default — delegation itself arrives as a real `created` agent session, so dispatching update webhooks as turns too would double-process every delegation. Opt in with `dispatch_issue_updates: true` to react to issue edits; those turns have no agent session, so replies post as issue comments (requires `mutation_policy.create_comments: true`).
- `reply_in_source_thread: true` is an opt-in **workaround** for a Linear limitation: a session response can't render inline in the thread the agent was mentioned from (only the first-party `@Linear` assistant can), so when on, the agent also replies on the mention's source comment. Content then appears in both the session widget and the thread. Default off matches other third-party agents. **Remove this flag** if Linear ships native inline/source-thread replies for third-party agents (see the adapter's `_reply_in_source_thread` comment for the unwind steps).
- `linear_agent_update_issue`/`create_issue` accept a workflow state NAME (`state: "Done"`) and resolve it to the required `stateId` automatically.
- MCP parity: `update_issue`/`create_issue` also resolve friendly reference keys (`assignee` by name/email/`me`, `labels`, `project`, `team`, `cycle`, `milestone`, `delegate`) to Linear's `*Id` fields; `null` clears where Linear allows it, and raw `*Id` keys still pass straight through. Ambiguous or unknown names abort BEFORE the mutation and name the lookup tool.
- `update_issue` manages issue relations (`blocks`, `blockedBy`, `relatedTo` — append-only; `removeBlocks`/`removeBlockedBy`/`removeRelatedTo` to remove; `parentId`; `duplicateOf`) and URL attachments (`links: [{url, title}]`, append-only). `blockedBy` is stored as the inverse of `blocks`.
- `linear_agent_create_comment` also edits an existing comment (pass `comment_id`, gated on `update_comments`) and posts threaded replies (`parentId`). Comments can target a non-issue parent — pass exactly ONE of `project_id`, `project_update_id`, `initiative_id`, `initiative_update_id`, or `document_content_id` instead of `issue_id`.
- Customers, release notes, and issue labels: `linear_agent_save_customer` (gated on `create_customers`/`update_customers`), `linear_agent_delete_customer` (`delete_customers`), `linear_agent_save_release_note` (release-family umbrella — `create_releases`/`update_releases`; `pipelineId` required to create), and `linear_agent_create_issue_label` (`create_labels`; omit `team_id` for a workspace-wide label). New reads (no policy): `linear_agent_get_team`, `_get_milestone`, `_get_document`, `_get_attachment`, `_get_release_note`, `_get_agent_skill`, `_list_project_labels`, `_list_release_notes`, `_list_agent_skills`.
- Delete tools (`linear_agent_delete_comment`, `_delete_customer_need`, `_delete_status_update`, `_delete_attachment`) require explicit IDs and are each gated by their own `delete_*` policy key (all default `false`). Status updates route through Linear's archive mutation.
- MCP migration: prefer `linear_agent_*` over `mcp_linear_*` — every MCP Linear tool available in Linear's public GraphQL API has an equivalent here, with app attribution and fail-closed write policies on top.

### Linear Agent Interaction Guidelines alignment

- The `created` acknowledgement runs concurrently with dispatch so the webhook responds within Linear's 5-second deadline and the session is acknowledged well inside the 10-second unresponsive-marking window.
- Clarifying questions post as `elicitation` activities, so the session shows `awaitingInput` in Linear's UI; the user's reply arrives as a normal `prompted` follow-up.
- Long turns surface an ephemeral "Working on it…" `thought` (rate-limited) as Linear's equivalent of a typing indicator.
- `linear_agent_set_session_links` attaches external URLs (PRs, docs) to the session — these render in Linear and also count as session activity.
- `linear_agent_update_plan` publishes the agent's execution plan (Linear **Agent Plans**, technology preview) as a live checklist on the session. The plan is replaced in full on every call — the model sends every step with its current status (`pending`/`inProgress`/`completed`/`canceled`).
- **Disengagement respect (`stop` signal):** a human `stop` signal (delivered on a `prompted` Agent Activity) halts the session immediately — the activity body is not dispatched as a prompt, any in-flight turn is interrupted, and a single confirming `response` activity is posted. No further Linear writes are made for that turn.
- The GraphQL client retries once on HTTP 429 honoring `Retry-After`, and routes through `LINEAR_AGENT_PROXY`/standard proxy env vars when set.
- Do not put real tokens or webhook secrets in `config.yaml` or docs.

### Linear agent best-practices alignment

- **Auto-start on delegation (default on, opt-out):** when an issue is delegated to the agent (an assignment/update webhook, or the issue's `delegate` is this app user), the adapter moves it to the team's first `started` workflow state (lowest `position`) — Linear's published best practice. Opt out with `auto_start_on_delegation: false`; also gated on `mutation_policy.update_issues`. Issues in a `triage` state are left alone so humans keep control of triage, and a bare @-mention is not treated as delegation.
- **Self-delegation (opt-in, default off):** with `auto_self_delegate: true` (or `LINEAR_AGENT_AUTO_SELF_DELEGATE`), auto-started issues that have NO delegate are also claimed by the agent. By default the adapter **never delegates issues to itself** — humans decide what the agent owns.
- **Permission-change awareness:** OAuth `revoked` and `teamAccessChanged` webhook events are logged loudly at WARNING (instead of being silently ignored) so a revoked token or lost team access is diagnosable.

---
name: osm-edit-mcp-setup
description: Configure and verify OSM Edit MCP safely.
version: 0.1.0
author: Petr Korolev, Hermes Agent
license: MIT
platforms: [linux, macos]
compatibility: Requires Hermes Agent, uv, and network access.
metadata:
  hermes:
    tags: [openstreetmap, osm, mcp, oauth, maps]
    related_skills: []
---

# OSM Edit MCP Setup Skill

Configure a local `osm-edit-mcp` stdio server for Hermes, keep development and
production credentials separate, and prove the selected API target and OAuth
identity before any edit. This skill configures access; it does not authorize an
OSM data change or bypass the server's review gates.

## Why This Skill Exists

A working OSM Edit MCP installation spans four independent layers:

1. **Hermes host configuration** starts the stdio process with a deliberately
   filtered environment.
2. **OSM Edit MCP configuration** selects development or production and enforces
   the `safe`/`expert` write profile.
3. **OAuth application credentials** identify a separate app on each OSM
   environment; development credentials cannot authenticate to production.
4. **OAuth access tokens** bind a live OSM account and permissions to the chosen
   environment and are stored independently from the client secret.

A partial success at one layer does not prove the system is ready. For example,
the OAuth helper can validate a production token while the already-running MCP
subprocess still targets development, or Hermes can serialize a boolean into an
MCP `env` mapping that accepts strings only and causes an immediate startup
failure. This workflow therefore verifies configuration, protocol startup, live
identity, and post-restart state separately.

The skill is useful to operators and agents that need reproducible setup and
troubleshooting without weakening OSM's review-first safety model. It is not a
general OpenStreetMap editing tutorial: data quality, tagging, local knowledge,
and per-edit review remain separate responsibilities.

## When to Use

- Installing or reconnecting OSM Edit MCP in Hermes.
- Switching deliberately between the OSM development sandbox and production.
- Repairing OAuth, keyring, startup, or API-target mismatches.
- Verifying that production writes fail closed behind proposal review.

Do not use this procedure to silently enable direct-write tools, copy sandbox
credentials into production, or publish an unreviewed map edit.

## Prerequisites

- A local `osm-edit-mcp` source checkout with `uv.lock` and `oauth_auth.py`.
- `uv` and a working Hermes CLI.
- A persistent Hermes home available as `$HERMES_HOME`.
- Separate OSM accounts or OAuth applications as required by the sandbox and
  production sites.
- A host that supports MCP elicitation before applying production proposals.

Discover paths with `search_files`; never assume a machine-specific checkout or
Hermes-home path. Keep settings in Hermes configuration and secrets in a private
env file or OS keyring.

## Security Invariants

1. Never paste OAuth client secrets, access tokens, or redirect authorization
   codes into issues, commits, skill files, or diagnostic logs.
2. If a secret was exposed in chat, finish only the minimum recovery work, then
   tell the user to revoke or rotate it.
3. Use different OAuth applications and token records for development and
   production.
4. Keep production on `OSM_WRITE_PROFILE=safe` and
   `OSM_REQUIRE_HOST_CONFIRMATION=true`.
5. Do not change to `expert` merely because a requested POI operation is absent.
   Explain the limitation and obtain separate, explicit authorization.
6. Treat production OSM edits as public and effectively permanent.

## Configuration Model

Use one private env file, for example:

```text
$HERMES_HOME/mcp/osm-edit-mcp.env
```

Required non-secret controls:

```dotenv
OSM_USE_DEV_API=true
OSM_WRITE_PROFILE=safe
OSM_REQUIRE_HOST_CONFIRMATION=true
OSM_ALLOW_CUSTOM_API_BASE=false
USE_KEYRING=true
ALLOW_PLAINTEXT_TOKEN_FILE=false
```

Development OAuth variables:

```dotenv
OSM_DEV_CLIENT_ID=<development-client-id>
OSM_DEV_CLIENT_SECRET=<development-client-secret>
OSM_DEV_REDIRECT_URI=https://localhost:8080/callback
```

Production OAuth variables:

```dotenv
OSM_PROD_CLIENT_ID=<production-client-id>
OSM_PROD_CLIENT_SECRET=<production-client-secret>
OSM_PROD_REDIRECT_URI=https://localhost:8080/callback
```

Leave legacy `OSM_CLIENT_ID` and `OSM_CLIENT_SECRET` unset because they can
override environment selection. Set the env file to mode `0600` on POSIX hosts.

### Single-source rule for API selection

This skill deliberately uses one server entry (`osm-edit`) plus one private env
file, switching `OSM_USE_DEV_API` only during the explicit handoff in step 7.
`docs/MCP_CLIENT_SETUP.md` also documents the alternative of two fully separate
entries (`osm-edit-dev` / `osm-edit-prod`) for hosts that can run both at once.
Both models are safe as long as the environment is verified before every write
and `OSM_USE_DEV_API` lives only in the private env file.

When `OSM_EDIT_MCP_ENV_FILE` points to the private env file, keep
`OSM_USE_DEV_API` there only. Do not duplicate it under
`mcp_servers.<name>.env` in Hermes configuration.

Hermes auto-coerces unknown scalar values such as `false` to a YAML boolean,
while MCP stdio environment values must be strings. Quoting the CLI argument can
instead store literal quote characters. Either form can make the MCP process
exit during validation. If the duplicate exists, remove it with:

```text
terminal(command="hermes config unset mcp_servers.osm-edit.env.OSM_USE_DEV_API")
```

## Procedure

### 1. Inspect before changing anything

Use `search_files` and `read_file` to identify:

- the active `$HERMES_HOME`;
- the source checkout;
- the private env file;
- the current `mcp_servers` entry;
- whether dev or production credentials are present, without printing values.

Report only key names, presence, and optionally value lengths. Completion
criterion: the selected environment and every configuration source are known.

### 2. Register separate OAuth applications

For development, register at
`https://api06.dev.openstreetmap.org/oauth2/applications`. For production,
register at `https://www.openstreetmap.org/oauth2/applications` with:

- Redirect URI: `https://localhost:8080/callback`
- Scopes: `read_prefs` and `write_api`

Never reuse the development application's credentials for production.
Completion criterion: the intended environment has its own client ID and secret.

### 3. Store credentials privately

Ask the user to enter credentials through a local secret-capable interface
rather than chat. If credentials were already supplied, write them only to the
private env file, suppress values from output, enforce mode `0600`, and verify
presence without displaying contents.

Always pass the same token-store namespace to the OAuth helper and MCP server:

```text
OSM_EDIT_MCP_ENV_FILE=$HERMES_HOME/mcp/osm-edit-mcp.env
XDG_DATA_HOME=$HERMES_HOME/mcp/osm-edit/keyring
```

Prefer the platform keyring and leave `PYTHON_KEYRING_BACKEND` unset. On a
headless host, `keyrings.alt.file.PlaintextKeyring` is an explicit compatibility
fallback: it stores recoverable plaintext, is not included in the server's base
dependencies, and must be installed into the exact runtime environment first.
If selected, pass the same `PYTHON_KEYRING_BACKEND` value to both the OAuth
helper and Hermes-launched MCP subprocess and protect the persistent directory.

Completion criterion: expected keys occur exactly once, the secret file is
private, both processes resolve the same token store, and no secret appears in
logs or version control.

### 4. Add the stdio server through Hermes

Use `terminal` and `hermes mcp add`; do not hand-edit `config.yaml`. A source
checkout invocation has this shape:

```text
terminal(command="hermes mcp add osm-edit --command <uv-path> --connect-timeout 120 --env OSM_EDIT_MCP_ENV_FILE=<env-file> XDG_DATA_HOME=<keyring-dir> OSM_WRITE_PROFILE=safe OSM_REQUIRE_HOST_CONFIRMATION=true --args run --locked --project <checkout> osm-edit-mcp", background=true, pty=true)
```

Keep `--args` last. Use `process(action="poll")` until Hermes displays the tool
selection prompt, then submit the user's intended selection with
`process(action="submit")` and finish with `process(action="wait")`. Seeing
`Connected` and a tool list is not sufficient: an EOF or cancelled selection
means the server was not saved. If the server already exists, inspect it before
replacing anything. Do not put OAuth secrets directly in the Hermes MCP entry.
Completion criterion: Hermes has one intentional OSM server entry pointing to
the private env file, and `hermes mcp test osm-edit` finds it after the add
process exits.

### 5. Authenticate development first

Start the OAuth helper in a PTY so its state and PKCE verifier remain alive:

```text
terminal(command="export OSM_EDIT_MCP_ENV_FILE=<env-file>; export XDG_DATA_HOME=<keyring-dir>; <uv-path> run --locked python oauth_auth.py --dev --no-browser", workdir="<checkout>", background=true, pty=true)
```

Use `process(action="poll")` to obtain the authorization URL and give it to the
user. The callback URL carries the one-time authorization code and state, so it
is a secret. Preferred: the user types it directly into the locally attached
PTY, or into a genuinely secret-capable input channel. Fallback: the agent may
forward it with `process(action="submit")` only after telling the user that the
URL will appear in the local session transcript; on a shared or exported
transcript, stop and have the user run `oauth_auth.py` in a local terminal
instead. Then use `process(action="wait")`.

Completion criterion: live `/user/details` and `/permissions` checks succeed for
the expected development account with `write_api`.

### 6. Complete development acceptance

Reconnect the MCP server, prepare representative non-writing previews, review
the exact tags and geometry, and apply only test data to the development API.
Do not move to production merely because authentication succeeded.

Completion criterion: representative create/update behavior has been exercised
against the sandbox and the resulting test objects were read back.

### 7. Switch and authenticate production deliberately

Change the private env file to:

```dotenv
OSM_USE_DEV_API=false
```

Confirm the API base is `https://api.openstreetmap.org/api/0.6`, custom API
overrides are disabled, and the production OAuth variables are present. Run the
same OAuth PTY workflow with `--prod`.

Completion criterion: the helper verifies the intended production username,
user ID, and `write_api`, then stores a production-specific token without
replacing the development token.

### 8. Verify a fresh MCP process

An already-running MCP subprocess cannot observe a changed process environment.
Start a fresh process and verify it before touching the active runtime:

```text
terminal(command="hermes mcp test osm-edit", timeout=240)
```

Call `get_edit_capabilities` and `check_authentication` through that fresh
connection and confirm all of the following:

- `environment` is exactly the intended environment;
- production target is `https://api.openstreetmap.org/api/0.6`;
- `write_profile` is `safe`;
- `raw_write_tools_registered` is `false`;
- digest-bound production confirmation is required;
- the username, user ID, and `write_api` match expectations.

Completion criterion: a freshly spawned process reports the intended API target
and identity.

### 9. Reload or restart the active runtime

Only now bring the running Hermes session in line with the verified
configuration. Prefer `/reload-mcp` as the cheap first step. If a live MCP call
still reports the old target or a tripped connection state after reload,
restart the Hermes runtime once; on Umbrel use the Hermes Agent app's Restart
action rather than self-updating Hermes or replacing its pinned image. After
reload or restart, call `get_edit_capabilities` and `check_authentication` again
from the active conversation.

Completion criterion: the active session, not only a diagnostic subprocess,
reports the intended API target and account.

## Pitfalls

- **`Connection closed` immediately:** inspect MCP stderr. A duplicated
  `OSM_USE_DEV_API` may have become a boolean or a string containing quote
  characters. Remove the Hermes duplicate and rely on the private env file.
- **Valid OAuth but wrong API:** check both the boolean selector and actual API
  base. Never infer environment from a token filename alone.
- **Authentication checks disagree:** one process may still be the old MCP
  subprocess. Verify with a fresh client, try `/reload-mcp`, and restart only if
  a live call remains stale.
- **Production token missing after dev success:** dev and production use separate
  OAuth applications and separate keyring records.
- **POI creation tools absent:** `safe` intentionally omits raw direct-write
  tools. Production OAuth does not enable them.
- **Redirect page fails to load:** with a localhost callback this can be normal;
  the full address bar URL still contains the one-time code and state.
- **Overpass timeouts:** they affect search requests, not OAuth identity checks.
  Retry read-only discovery later rather than weakening write safeguards.

## Verification

Before declaring setup complete, record non-secret evidence for:

1. Secret-file mode and unique key presence.
2. `hermes mcp test <name>` connection success and discovered tool count.
3. `get_edit_capabilities` API environment, target, profile, and confirmation
   mechanism.
4. `check_authentication` expected username, user ID, and `write_api`.
5. A post-reload (or post-restart) check from the active Hermes session.

Never include client secrets, access tokens, authorization codes, raw keyring
records, or full redirect URLs in the verification report.

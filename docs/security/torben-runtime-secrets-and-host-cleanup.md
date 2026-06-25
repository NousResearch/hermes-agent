# Torben Runtime Secrets And Host Cleanup

Status: Draft reset runbook
Date: 2026-06-25
Related PRD: `docs/prds/hermes-native-signal-coo-operator-prd.md`

## Current Remote Cutline

Executed from SSH on 2026-06-24:

- Torben profile route is configured as OpenAI default for Torben/EA/Finance and Grok/xAI for Magnus/GTM.
- Torben runtime env template validates as 1Password refs only, with no plaintext secret values.
- Torben profile now uses `openai-codex` for the default OpenAI OAuth lane, `xai-oauth` for Magnus/GTM, and registered MCPs for finance.
- Direct remote-safe wrapper installed at `/Users/ericfreeman/.hermes/profiles/torben/bin/torben-hermes`.
- Remote-safe wrapper installed at `/Users/ericfreeman/.hermes/profiles/torben/bin/torben-op-hermes`.
- Wrapper fails closed with exit `78` when `OP_SERVICE_ACCOUNT_TOKEN` is absent, instead of entering an interactive 1Password flow from SSH.
- Launchd template staged but not loaded at `/Users/ericfreeman/.hermes/profiles/torben/launchd/ai.hermes.gateway-torben.plist`; it points at the direct wrapper, not the `op` wrapper.
- Finance MCP endpoints are configured in the Torben profile:
  `robinhood-agentic-mcp` at `https://agent.robinhood.com/mcp/trading` and
  `monarch-money-mcp` at `https://api.monarch.com/mcp`, both with
  `auth: oauth`. Browser OAuth login has been completed for both connectors.
- Torben cron delivery has `cron.wrap_response=false`; background jobs should
  send only actionable output or failures, not generic `Cronjob Response`
  wrappers.
- Legacy OpenClaw, Floki/Magnus/Ratatosk gateway, GBrain, Magnus keepalive, Ratatosk, Hermes auto-update, and Hermes operational-health launchd jobs were copied to `/Users/ericfreeman/.hermes/profiles/torben/decommission/20260624T185951Z/launchagents`, unloaded, and disabled.
- Signal Desktop and `com.hermes.keepawake` were intentionally left running for remote safety.

Remaining live-runtime bootstrap: none for current OAuth/MCP access.
1Password is no longer a global launch blocker; it is optional bootstrap for
non-OAuth static values only.

Remote-safe OAuth commands:

```bash
/Users/ericfreeman/.hermes/profiles/torben/bin/torben-hermes \
  auth add openai-codex --type oauth --no-browser --manual-paste

/Users/ericfreeman/.hermes/profiles/torben/bin/torben-hermes \
  auth add xai-oauth --type oauth --no-browser --manual-paste
```

After each auth flow:

```bash
/Users/ericfreeman/.hermes/profiles/torben/bin/torben-hermes auth status openai-codex
/Users/ericfreeman/.hermes/profiles/torben/bin/torben-hermes auth status xai-oauth
```

Remote-safe hosted MCP OAuth re-login commands if tokens are revoked or expire:

```bash
/Users/ericfreeman/.hermes/profiles/torben/bin/torben-hermes \
  mcp login robinhood-agentic-mcp

/Users/ericfreeman/.hermes/profiles/torben/bin/torben-hermes \
  mcp login monarch-money-mcp
```

## Goal

Torben should run with no provider secrets in repo files, LaunchAgent plists, or long-lived `.env` files. OAuth/provider-native credentials and registered MCP sessions are the primary auth plane. 1Password can provide optional static-secret bootstrap for integrations that cannot use OAuth or MCP-native auth. Legacy OpenClaw/Hermes bridge services should be stopped or explicitly allowlisted before the first live Torben canary.

Model routing is intentionally split:

- Torben, EA, Finance, memory, policy, and orchestration use OpenAI by default.
- Magnus/GTM/X-native search and social review use Grok through the existing Magnus/xAI lane.
- X publishing credentials are separate from Grok search/review credentials and remain approval-gated.
- Finance execution uses registered MCPs, not hidden broker API secrets in Torben config.

## Runtime Auth Model

Use OAuth/provider-native auth and registered MCP sessions first:

- Torben/EA/Finance orchestration: OpenAI Codex OAuth through Hermes auth.
- Magnus/GTM/X-native work: xAI/Grok OAuth through the Magnus lane.
- Google Calendar/Gmail: Google OAuth.
- Finance/day trading: hosted OAuth MCPs, currently Robinhood Agentic MCP and
  Monarch Money MCP.
- Signal: local `signal-cli` or gateway config; Signal is not OAuth.

Use 1Password only for remaining non-OAuth static values.

Optional 1Password patterns:

- `op run --env-file <file-with-op-refs> -- <command>`
- `op run --environment <environment-id> -- <command>`
- `op read op://vault/item/field`
- `op inject --in-file <template> --out-file <private-temp-file>`

Required rule: repo-owned config can contain `op://` references, but never resolved secret values. 1Password absence must not block OAuth/MCP-native checks.

Launchd should call the direct wrapper by default:

```bash
/Users/ericfreeman/.hermes/profiles/torben/bin/torben-hermes gateway run --replace
```

The optional `op` wrapper remains available only when a specific non-OAuth static secret is required:

```bash
#!/usr/bin/env bash
set -euo pipefail

export OP_SERVICE_ACCOUNT_TOKEN="$(/usr/bin/security find-generic-password -w -s torben-op-service-account-token -a torben)"

exec /opt/homebrew/bin/op run \
  --env-file /Users/ericfreeman/.hermes/profiles/torben/runtime.env.op \
  -- /Users/ericfreeman/.hermes/hermes-agent/venv/bin/python \
    -m hermes_cli.main torben run
```

If the optional `op` wrapper is used under launchd, the `OP_SERVICE_ACCOUNT_TOKEN` bootstrap secret must live outside the repo, preferably in macOS Keychain. If interactive 1Password app auth is used instead of a service account, the `op` wrapper must fail closed after reboot or lock until 1Password is unlocked.

Current `runtime.env.op` should stay limited to local process bootstrap:

```dotenv
SIGNAL_ACCOUNT=op://Torben Runtime/Signal/account
SIGNAL_HTTP_URL=op://Torben Runtime/Signal/http-url
```

Do not add provider tokens, broker credentials, Google token JSON, X API
tokens, or Monarch credentials to this file. OpenAI/xAI use Hermes auth store,
Google uses per-account OAuth files, and Robinhood/Monarch use MCP OAuth token
storage.

## Torben Verification Commands

These commands are safe to run without resolving secret values:

```bash
/Users/ericfreeman/.hermes/hermes-agent/venv/bin/python \
  /Users/ericfreeman/.hermes/hermes-agent/hermes \
  -p torben torben route --json

/Users/ericfreeman/.hermes/hermes-agent/venv/bin/python \
  /Users/ericfreeman/.hermes/hermes-agent/hermes \
  -p torben torben secrets-check --json

/Users/ericfreeman/.hermes/hermes-agent/venv/bin/python \
  /Users/ericfreeman/.hermes/hermes-agent/hermes \
  -p torben torben auth-check --json
```

Expected route: `default.provider=openai-codex` for Torben/EA/Finance and
`gtm.provider=xai-oauth` for Magnus/GTM/X-native work. Expected auth policy:
`strategy=oauth_mcp_native_first`, `onepassword_bootstrap=optional`, and
`finance_execution=registered_mcp`. Expected MCP policy includes
`mcp_native_connectors=["robinhood-agentic-mcp","monarch-money-mcp"]` with no
missing or disabled required MCP connectors.

## Legacy Secret Migration

Known legacy secret surfaces from the read-only inventory:

- `/Users/ericfreeman/.hermes/.env`
- `/Users/ericfreeman/.hermes/profiles/floki/.env`
- `/Users/ericfreeman/.hermes/profiles/magnus/.env`
- `/Users/ericfreeman/.hermes/profiles/ratatosk/.env`
- `/Users/ericfreeman/.openclaw/.env`
- `/Users/ericfreeman/.openclaw/service-env/ai.openclaw.gateway.env`
- OpenClaw `.env` and `openclaw.json` backup files under `/Users/ericfreeman/.openclaw`

Migration rule:

1. Inventory key names without printing values.
2. Prefer OAuth/provider-native or registered MCP auth for each integration.
3. Create or map 1Password items only for static values that cannot use OAuth or MCP-native auth.
4. Quarantine old env files outside active runtime paths.
5. Re-run auth checks and secret scans to prove Torben does not load old plaintext sources.

## Host Cleanup Inventory

Read-only snapshot found these likely cleanup candidates:

- `ai.openclaw.gateway`
- `ai.openclaw.cron-receiver`
- `ai.openclaw.filter`
- OpenClaw-managed `signal-cli` daemon
- `ai.hermes.gateway-floki`
- `ai.hermes.gateway-magnus`
- `ai.hermes.gateway-ratatosk`
- `ai.gbrain.mcp`
- `com.hermes.keepawake`
- `com.hermes.operational-health`
- `com.hermes.auto-update`
- `com.magnus.x-oauth2-keepalive`
- `com.ratatosk.*` launchd jobs

Do not stop Signal Desktop or unrelated user apps as part of this cleanup.

## Dry-Run Inventory Commands

```bash
ps -axo pid,ppid,stat,%cpu,%mem,etime,command \
  | rg -i 'hermes|openclaw|heimdall|floki|magnus|ratatosk|signal-cli|gbrain|juno'

launchctl list \
  | rg -i 'hermes|openclaw|heimdall|floki|magnus|ratatosk|signal|gbrain|juno'

find "$HOME/Library/LaunchAgents" -maxdepth 1 -type f \
  \( -iname '*hermes*.plist' -o -iname '*openclaw*.plist' -o -iname '*ratatosk*.plist' \
     -o -iname '*magnus*.plist' -o -iname '*floki*.plist' -o -iname '*gbrain*.plist' \
     -o -iname '*signal*.plist' \) -print
```

## Stop Sequence

Run only after the allowlist is approved.

```bash
uid="$(id -u)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
quarantine="$HOME/.hermes/decommission/$stamp-launchagents"
mkdir -p "$quarantine"

for label in \
  ai.openclaw.gateway \
  ai.openclaw.cron-receiver \
  ai.openclaw.filter \
  ai.hermes.gateway-floki \
  ai.hermes.gateway-magnus \
  ai.hermes.gateway-ratatosk \
  ai.gbrain.mcp \
  com.hermes.keepawake \
  com.hermes.operational-health \
  com.hermes.auto-update \
  com.magnus.x-oauth2-keepalive
do
  plist="$HOME/Library/LaunchAgents/$label.plist"
  [ -f "$plist" ] && cp "$plist" "$quarantine/"
  launchctl bootout "gui/$uid/$label" 2>/dev/null || true
  launchctl disable "gui/$uid/$label" 2>/dev/null || true
done

for plist in "$HOME"/Library/LaunchAgents/com.ratatosk.*.plist
do
  [ -e "$plist" ] || continue
  label="$(basename "$plist" .plist)"
  cp "$plist" "$quarantine/"
  launchctl bootout "gui/$uid/$label" 2>/dev/null || true
  launchctl disable "gui/$uid/$label" 2>/dev/null || true
done
```

If a child process survives after its launchd parent is unloaded, inspect it before killing it:

```bash
ps -axo pid,ppid,stat,%cpu,%mem,etime,command \
  | rg -i 'openclaw|hermes|ratatosk|magnus|floki|gbrain|signal-cli'
```

## Verification

After cleanup:

```bash
launchctl list \
  | rg -i 'hermes|openclaw|heimdall|floki|magnus|ratatosk|signal|gbrain|juno' || true

ps -axo pid,ppid,stat,%cpu,%mem,etime,command \
  | rg -i 'hermes|openclaw|heimdall|floki|magnus|ratatosk|signal-cli|gbrain|juno' || true
```

Expected result: no legacy services remain except explicit allowlist entries and this Codex session.

## Non-Goals

- Do not delete source repos.
- Do not delete state until a separate retention decision exists.
- Do not migrate live broker, email, calendar, or social credentials without explicit approval.
- Do not use old `.env` values as runtime fallback after Torben launches.

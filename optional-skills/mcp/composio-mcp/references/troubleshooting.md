# Composio — extended troubleshooting

## CLI cache and pending login

The CLI stores state under `~/.composio/`:

| File | Purpose |
|------|---------|
| `user_data.json` | `ak_*` API key, base URL, org id |
| `config.json` | Developer flags, security mode |
| `pending-login-session.json` | In-flight `--no-browser` login session |
| `toolkits.json` | Known toolkit slugs cache |
| `tool_definitions/` | Cached tool schemas |
| `tool-permissions-cache.json` | Per-tool permission grants |

If `composio whoami` is stale or `execute` behaves oddly, clear the cache:

```bash
rm -rf ~/.composio/tool_definitions ~/.composio/toolkits.json ~/.composio/tool-permissions-cache.json
composio whoami   # re-fetches
```

## Org picker hangs on login

If `composio login` hangs at the org picker, pass `--yes` to use the current org:

```bash
composio login --user-api-key ak_... --yes
```

Or set the org explicitly:

```bash
composio login --user-api-key ak_... --org my_org_slug
```

## Pending login session

If a `--no-browser --no-wait` login was started but never completed, the session lingers:

```bash
cat ~/.composio/pending-login-session.json   # inspect
composio login --poll                         # resume polling up to 10 min
rm ~/.composio/pending-login-session.json     # or abandon
```

## `composio dev` (developer projects)

`composio dev` is for building agent projects with the Composio SDK — scaffolding, playground execution, logs, and developer-scoped management. It is **not** the default end-user path. Reach for it only when the user explicitly asks about SDK projects, auth configs, connected accounts, triggers, logs, orgs, or projects.

## MCP: "Bearer token rejected: not a valid AuthKit JWT"

This means the server received a bearer token but it is not a valid OAuth JWT. You are probably sending an `ak_*` key in an `Authorization: Bearer` header. The Connect MCP endpoint wants a `ck_*` consumer key in `x-consumer-api-key`, **not** a bearer token. Fix the header.

## MCP: tools list works but execute returns 401

The org has `require_mcp_api_key` enabled. Add the `ak_*` project API key in an `x-api-key` header alongside the consumer key (see `mcp-config.md` → "Optional: enforce API key").

## Connected account per toolkit

`composio link <toolkit>` opens a browser OAuth flow per app. On headless boxes:

```bash
composio link gmail --no-browser   # prints a URL; authorize in any browser
```

Multiple accounts for the same toolkit need aliases:

```bash
composio link gmail --alias work
composio link gmail --alias personal
composio execute GMAIL_SEND_EMAIL --account work -d '{ ... }'
```

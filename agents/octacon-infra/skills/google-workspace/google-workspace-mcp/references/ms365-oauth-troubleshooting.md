# ms-365-mcp-server OAuth Troubleshooting

## Architecture

The ms-365-mcp-server uses MSAL with two token cache paths:

1. **User-configured path** (env: `MS365_MCP_TOKEN_CACHE_PATH`):
   `~/.config/ms-365-mcp-server/token-cache.json`
2. **Fallback path** (module directory):
   `~/.hermes/node/lib/node_modules/@softeria/ms-365-mcp-server/.token-cache.json`

Both paths are valid. The MCP server checks the configured path first, then falls back. The `--list-accounts` CLI reads from the in-memory MSAL cache (which loads both files).

## The User.Read 403 Problem

When `--login --preset mail,calendar` completes successfully but `verify-login` returns `403`, it is **scope filtering**.

The `--preset mail,calendar` flag filters which tools are registered. Its filter pattern is `mail|attachment|draft|calendar|event` — it matches tool names, not scope requirements. The `get-current-user` endpoint uses `User.Read` scope but its tool name does not match the preset filter, so `User.Read` is never requested during consent.

### Symptoms
```json
{"success":false,"message":"Login successful but Graph API access failed: 403"}
```

The token was granted for `Mail.ReadWrite`, `Calendars.ReadWrite`, etc. but NOT `User.Read`. Graph's `/me` endpoint requires `User.Read` and returns 403.

### Fix
Re-auth **WITHOUT** the `--preset` filter so all scopes including `User.Read` are requested:

```bash
/home/kensei/.local/bin/node /home/kensei/.hermes/node/bin/ms-365-mcp-server --login
```

After completing the consent screen, `/me` returns 200.

### Note
Accounts that lack `User.Read` still have fully functional mail and calendar access. They only fail on `/me` (user profile). The mailbox agent works fine for these accounts. If `token_health.py` reports `mail functional (no User.Read /me 403 expected)`, the account is healthy for mail operations.

## Duplicate Token Cache

New accounts added via `--login` may land in the **fallback cache** instead of the configured path. Merge tokens into the configured path after auth:

```bash
cp ~/.hermes/node/lib/node_modules/@softeria/ms-365-mcp-server/.token-cache.json \
   ~/.config/ms-365-mcp-server/token-cache.json
```

If the configured path already had other accounts, merge with Python instead of overwriting (see the merge pattern in `references/ms365-oauth-troubleshooting.md` in the google-workspace-mcp skill).

## CLI Reference

| Command | Description |
|---|---|
| `--login` | Device code auth (no port forwarding needed) |
| `--login` (no --preset) | All scopes including User.Read |
| `--login --preset mail,calendar` | Limited scopes, User.Read excluded |
| `--verify-login` | Test Graph API access (requires User.Read) |
| `--list-accounts` | Show all cached accounts |
| `--select-account <id>` | Set default account |
| `--remove-account <id>` | Remove an account |
| `--list-permissions` | Show which scopes will be requested |
| `--logout` | Clear all tokens and account data |
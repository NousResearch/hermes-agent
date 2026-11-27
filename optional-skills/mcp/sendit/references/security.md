# Security And Permissions

This skill uses remote MCP OAuth. It does not require a SendIt API key, browser
cookies, platform passwords, or local social account tokens.

## Local Files

The installer may read or write:

- `$HERMES_HOME/config.yaml` when `HERMES_HOME` is set.
- `~/.hermes/config.yaml` when `HERMES_HOME` is not set.
- `~/.hermes/skills/social-media/sendit/` for the installed skill copy.

Before changing an existing config, the installer preserves unrelated YAML and
backs up the original file beside it with a timestamped `.bak` suffix. The
repair path only targets the SendIt MCP entry and upgrades older
`https://sendit.infiniteappsai.com/mcp` values to
`https://sendit.infiniteappsai.com/api/mcp`.

The OAuth starter writes temporary process state under `/tmp/sendit-hermes`:

- `oauth.pid`
- `oauth.log`

These files are operational logs only and should not be committed.

## Network And Commands

The setup scripts may invoke:

```bash
hermes mcp login sendit
```

They configure SendIt as a remote OAuth MCP server at:

```text
https://sendit.infiniteappsai.com/api/mcp
```

No script reads browser profiles, password stores, SSH keys, shell history, or
application source files.

## OAuth Callback Handling

`complete-oauth-callback.mjs` accepts only:

- `http://127.0.0.1:<port>/callback?...`
- `http://localhost:<port>/callback?...`

It rejects remote hosts, malformed URLs, and non-`/callback` paths. In output,
the helper redacts sensitive callback parameters such as `code` and `state`.

Never paste full callback URLs into public logs or issue trackers. If a URL was
exposed, restart the OAuth login flow so the old callback becomes unusable.

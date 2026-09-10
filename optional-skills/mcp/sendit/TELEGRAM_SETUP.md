# SendIt Telegram-Only Setup For Hermes

Use this playbook when the user sends this skill folder or archive to a Hermes
Agent running on a remote VPS through Telegram or another messaging gateway.

## Goal

Configure SendIt in Hermes without asking the user to SSH into the VPS.

Hermes should:

1. Install the SendIt skill into `~/.hermes/skills/social-media/sendit`.
2. Add the SendIt remote MCP server to `~/.hermes/config.yaml`.
3. Start the OAuth login flow on the VPS.
4. Send the user the authorization URL.
5. Wait for the user to paste the failed localhost callback URL from their local
   browser.
6. Replay that callback URL against the VPS-local callback listener.
7. Reload MCP and verify SendIt tools work.

## Commands

Run these commands from the directory that contains this file.

### 1. Install skill and MCP config

```bash
node scripts/install-sendit-hermes.mjs
```

This also repairs older installs that used `https://sendit.infiniteappsai.com/mcp`.
Hermes should use `https://sendit.infiniteappsai.com/api/mcp`; `/mcp` is the
ChatGPT-specific app endpoint and has a reduced tool catalog.

### 2. Start OAuth login

```bash
node scripts/start-oauth-login.mjs
```

Send the printed authorization URL to the user.

### 3. Complete OAuth after the user pastes a callback URL

When the user pastes a URL like:

```text
http://127.0.0.1:43879/callback?code=...&state=...
```

run:

```bash
node scripts/complete-oauth-callback.mjs '<PASTED_CALLBACK_URL>'
```

Do not echo the pasted URL back to the user. The helper validates the callback
host and path, replays it to the VPS-local listener, and redacts sensitive query
values from output.

### 4. Reload and verify

Ask the user to send:

```text
/reload-mcp
```

Then verify:

```text
Use the SendIt skill and list my connected social accounts.
```

## Telegram Prompt The User Can Send

After uploading this folder or archive to Hermes, the user can say:

```text
Unpack this SendIt Hermes setup folder if needed. Read TELEGRAM_SETUP.md.
Configure SendIt completely from Telegram. Install the skill, add the MCP server,
start OAuth, send me the auth URL, wait for me to paste the localhost callback
URL, then replay it on the VPS and verify SendIt tools are available.
```

## Notes

- Do not ask the user for a SendIt API key. This setup uses MCP OAuth.
- Use only `https://sendit.infiniteappsai.com/api/mcp` for the MCP server URL.
- Do not echo full OAuth `code` or `state` values back to the user.
- Only replay localhost callback URLs with host `127.0.0.1` or `localhost` and
  path `/callback`.
- If OAuth times out, rerun `node scripts/start-oauth-login.mjs` and send the
  new authorization URL.

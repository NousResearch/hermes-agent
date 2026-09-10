# Remote OAuth

Use this reference when Hermes runs on a VPS, inside a remote shell, or behind a
Telegram/file-transfer workflow where the user opens OAuth URLs on another
device.

## Canonical MCP URL

Configure SendIt at:

```text
https://sendit.infiniteappsai.com/api/mcp
```

Do not configure:

```text
https://sendit.infiniteappsai.com/mcp
```

The `/mcp` endpoint is reserved for the ChatGPT app submission profile and has a
reduced tool catalog. The `/api/mcp` endpoint exposes the standard SendIt MCP
catalog, including team tools.

## Setup Sequence

If the skill arrived through Telegram or another archive handoff, `TELEGRAM_SETUP.md`
contains the operator checklist. The full sequence is:

1. Install the skill and repair/create Hermes MCP config:

   ```bash
   node ${HERMES_SKILL_DIR}/scripts/install-sendit-hermes.mjs
   ```

   If `${HERMES_SKILL_DIR}` was not substituted, run from the skill directory:

   ```bash
   node scripts/install-sendit-hermes.mjs
   ```

2. Start the OAuth listener on the VPS:

   ```bash
   node ${HERMES_SKILL_DIR}/scripts/start-oauth-login.mjs
   ```

3. Send the printed SendIt authorization URL to the user.
4. Ask the user to open it, sign in, approve SendIt, and copy the final
   localhost callback URL from the browser address bar.
5. Replay the callback URL on the VPS:

   ```bash
   node ${HERMES_SKILL_DIR}/scripts/complete-oauth-callback.mjs '<PASTED_CALLBACK_URL>'
   ```

6. Ask the user to send `/reload-mcp`, or restart the Hermes gateway/session if
   SendIt tools do not appear.

## Callback Rules

The replay helper accepts only callback URLs with:

- protocol `http:`
- host `127.0.0.1` or `localhost`
- path `/callback`
- a valid loopback port

Reject remote hosts, malformed URLs, and non-callback paths. Treat callback URLs
as sensitive because the `code` and `state` values can complete OAuth. The
helper redacts those parameters in user-visible output.

## Logs And Recovery

OAuth helper state is stored under `/tmp/sendit-hermes`:

- `/tmp/sendit-hermes/oauth.pid`
- `/tmp/sendit-hermes/oauth.log`

If the auth URL is not printed, inspect the log and confirm `hermes mcp login
sendit` is still running. If callback replay fails because the listener timed
out or exited, run `start-oauth-login.mjs` again and use the new authorization
URL.

If SendIt tools do not appear after successful OAuth, reload MCP in Hermes or
restart the Hermes session.

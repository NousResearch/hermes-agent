# Messaging setup

Use this recipe when asked to connect a messaging platform. Setup changes who
can reach Hermes, so the command requires approval. Never request or paste a
bot token in chat.

## Telegram

1. Run with `terminal`:
   ```json
   {"command":"hermes gateway setup telegram --quick","background":true,"notify":["Link:"]}
   ```
   `notify: ["Link:"]` wakes you when the link is printed.
2. When `Link:` appears, send that link only to the user who requested setup.
   Whoever opens it becomes the bot's owner. Never post it in a group.
   Ask the user to open it in Telegram and tap **Create Bot**.
3. After sending the link, call `process_manage` with `action: "wait"` and the
   process's `session_id`; if it returns on timeout, wait again. Do not claim
   success before the command reports that the bot is connected.
4. Run the next step it printed: `hermes gateway ensure` or
   `hermes gateway restart` (restart asks for approval).

Use `--allow 111,222` to add numeric Telegram user IDs alongside the detected
owner, `--replace` only when replacing an existing bot is intended, and
`--timeout SECONDS` to change the default 180-second wait. If setup reports
Team Gateway, use the desktop app's Messaging settings instead.

## Other platforms

Ask the user to run `hermes gateway setup <platform>` in their own terminal,
or use the desktop app's Messaging settings. Never ask for a token in chat.

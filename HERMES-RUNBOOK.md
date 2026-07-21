# Hermes Runbook

## Installed location

- Repo: `/Users/steven/.openclaw/workspace/repos/hermes-agent`
- Isolated Hermes home: `/Users/steven/.openclaw/workspace/repos/hermes-agent/.hermes-home`

## Current working setup

- Model provider: `openai-codex`
- Default model: `openai-codex/gpt-5.4`
- Telegram bot: `@spak47moltbot`
- Allowed Telegram user: `8459630899`
- Gateway manager: `launchd`

## Daily-use commands

```bash
cd /Users/steven/.openclaw/workspace/repos/hermes-agent
./scripts/run-local-hermes.sh status
./scripts/run-local-hermes.sh gateway status
./scripts/run-local-hermes.sh gateway restart
```

## Logs

```bash
cd /Users/steven/.openclaw/workspace/repos/hermes-agent
tail -f .hermes-home/logs/gateway.log
tail -f .hermes-home/logs/gateway.error.log
```

## Auth

List stored auth:

```bash
./scripts/run-local-hermes.sh auth list
```

Re-run OpenAI Codex OAuth if needed:

```bash
HERMES_HOME="$PWD/.hermes-home" ./scripts/run-local-hermes.sh auth add openai-codex --type oauth --no-browser
./scripts/run-local-hermes.sh config set model.provider openai-codex
./scripts/run-local-hermes.sh config set model.default openai-codex/gpt-5.4
```

## Telegram config

Stored in repo-local file only:

- `.hermes-home/.env`

Contains:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_ALLOWED_USERS`
- proxy env vars used for this Mac environment

## Notes

- This setup is intentionally isolated from OpenClaw and uses repo-local config/state.
- Python 3.11 is used because the full `.[all]` extra path is currently blocked by a Python-3.12+-only dependency in Hermes dev extras.
- If Telegram polling becomes unstable, first check for duplicate gateway processes and then inspect `.hermes-home/logs/gateway.error.log`.

# Ops — Telegram bots + chat IDs (NGMI + WalletDB)

## Sender bot
- **Canonical sender bot for both systems:** `@Sathyrnbot`

Notes:
- Store secrets in env/session files; never commit tokens.
- Prefer per-repo env files under each repo’s `session/` directory (ignored by git).

## NGMI-terminal
- Purpose: ingest/scan and produce trend digests + lane jobs.
- DB (chat logs): `/home/yeqiuqiu/projects/ngmi-terminal/data/ngmi_terminal.db` (treat as read-only outside NGMI-terminal).

Typical env vars (names may vary by deployment):
- `TELEGRAM_BOT_TOKEN` (token for `@Sathyrnbot`)
- `NGMI_*_CHAT_ID` (lane chats / operator chat)

## WalletDB
- Purpose: entity clustering + alerts + delivery.
- DB: `data/walletdb.sqlite` (or `state/walletdb_helius_phase1.sqlite` depending on pipeline)

Typical env vars:
- `TELEGRAM_BOT_TOKEN` (token for `@Sathyrnbot`)
- `WALLETDB_OPERATOR_CHAT_ID`
- `WALLETDB_*_CHAT_ID` (alerts/watchlist destinations)

## Chat ID hygiene
- Keep a single source of truth per repo (env file) rather than scattering literals in code.
- If you must hardcode for a test/fixture, put it under `tests/fixtures/` and document it.

## Separation rule
WalletDB must not read NGMI-terminal DBs by default.

Any cross-system correlation features must be:
- disabled by default
- gated behind explicit env vars
- documented in the WalletDB ops runbooks

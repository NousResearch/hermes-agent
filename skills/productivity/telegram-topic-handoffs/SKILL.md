---
name: telegram-topic-handoffs
description: Post daily per-topic Telegram handoff summaries.
version: 1.0.0
author: Tina Marie (tmnsystems)
license: MIT
platforms:
  - macos
  - linux
metadata:
  hermes:
    tags:
      - telegram
      - handoffs
      - daily-summary
      - sqlite
      - automation
---

# telegram-topic-handoffs

The `topic_handoffs.py` script reads recent Hermes conversation history from the state database, groups it by Telegram forum topic, and writes one Markdown handoff file per topic.
When a bot token and chat id are configured, it also posts each handoff to its topic in the configured Telegram chat.
The script is executed directly as a standalone program and is not an importable module API.

> **PRIVACY WARNING:** Handoff files contain private conversation text. The output directory must never sit inside a cloud-synced folder.

## When to Use

Use this skill to generate daily per-topic handoff summaries from Hermes history and, optionally, deliver them to their Telegram forum topics.
It remains useful on machines without Telegram credentials, because it runs in a deliberate file-only mode when no token is configured.

## Prerequisites

- Python 3; the script uses only the standard library.
- Read access to the Hermes state database, which lives at `~/.hermes/state.db` by default.
- A Telegram bot token and chat id when posting is desired; both are optional for file-only runs.

## How to Run

Invoke the helper script through the terminal tool, running it directly with Python:

```bash
python3 skills/productivity/telegram-topic-handoffs/scripts/topic_handoffs.py
```

To perform a dry run that writes the files without posting anything, set `DRY_RUN` in the environment.
There is no `--dry-run` command line flag, so dry runs are controlled only through this variable:

```bash
DRY_RUN=1 python3 skills/productivity/telegram-topic-handoffs/scripts/topic_handoffs.py
```

## Quick Reference

All configuration is read from environment variables at import time.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HANDOFF_CHAT_ID` | unset | Telegram chat, meaning the forum supergroup, that handoffs are posted to. |
| `HANDOFF_DB` | `~/.hermes/state.db` | Path to the Hermes state database. |
| `HANDOFF_OUT_DIR` | `~/.hermes/topic-handoffs` | Directory where per-topic handoff files are written. |
| `HANDOFF_MIRROR_DIR` | unset | Optional second directory that receives a copy of each handoff file. |
| `HANDOFF_PUBLIC_BASE_URL` | unset | Optional base URL used to build public links to the mirrored files. |
| `HANDOFF_USER_LABEL` | `User` | Display label used for human messages. |
| `HANDOFF_BOT_LABEL` | `Assistant` | Display label used for assistant messages. |
| `HANDOFF_PREFIX` | `Daily handoff for` | Text placed before each topic name in handoff titles. |
| `HANDOFF_TOPIC_NAMES` | unset | Optional path to a JSON file mapping topic identifiers to friendly topic names. |
| `TELEGRAM_BOT_TOKEN` | unset | Bot token used to post to Telegram; `HANDOFF_BOT_TOKEN` is accepted as an alternative, and if neither is set the file at `HANDOFF_ENV_FILE` (default `~/.hermes/.env`) is checked for a token. |
| `HANDOFF_API_BASE` | `https://api.telegram.org` | Overrides the Telegram Bot API base URL. |
| `HANDOFF_RECENT_LIMIT` | `12` | Caps how many recent messages per topic are considered. |
| `HANDOFF_POST_EXCERPTS` | `6` | Controls how many excerpt lines appear in each posted handoff. |
| `HANDOFF_FILE_TRUNC` | `400` | Truncation length for individual messages in the written files. |
| `HANDOFF_POST_TRUNC` | `250` | Truncation length for individual messages in the posted excerpts. |
| `HANDOFF_MAX_POST_LEN` | `3800` | Caps the total length of a single Telegram post. |
| `HANDOFF_DB_TIMEOUT` | `30` | SQLite connection timeout in seconds. |
| `HANDOFF_SEND_DELAY` | `1.2` | Delay in seconds between Telegram sends. |
| `HANDOFF_MAX_RETRIES` | `4` | How many times a failed send is retried. |
| `HANDOFF_BACKOFF_BASE` | `2.0` | Base delay for exponential backoff between retries. |
| `DRY_RUN` | unset | When truthy, writes the handoff files and skips posting to Telegram. |

## Procedure

1. Invoke the script through the terminal tool; it reads all configuration from environment variables at import time.
2. It connects to the Hermes state database and groups recent history by Telegram forum topic.
3. A redaction pass drops any line that looks like it contains a secret entirely; dropped lines are not replaced with any placeholder or marker text.
4. One Markdown file per topic is written under a dated subdirectory at `HANDOFF_OUT_DIR/<YYYY-MM-DD>/<thread id>-<topic slug>.md`, plus a copy at `HANDOFF_OUT_DIR/latest/<thread id>.md`, and each file is titled `Handoff: <topic name> (topic <id>)`; when `HANDOFF_MIRROR_DIR` is set a copy is also written under `HANDOFF_MIRROR_DIR/<YYYY-MM-DD>/`, and when `HANDOFF_PUBLIC_BASE_URL` is set the public link for each mirrored file is reported.
5. The bot token is resolved from `TELEGRAM_BOT_TOKEN`, `HANDOFF_BOT_TOKEN`, or the env file at `HANDOFF_ENV_FILE`; if none is available, every file is still written and the run finishes successfully in the deliberate file-only mode.
6. When a token and chat id are configured, each handoff is posted to its topic, with sends spaced by `HANDOFF_SEND_DELAY` seconds and failures retried up to `HANDOFF_MAX_RETRIES` times with exponential backoff based on `HANDOFF_BACKOFF_BASE`.
7. A run summary is printed at the end, and it reports how many lines the redaction pass omitted.

## Pitfalls

- If nothing is posted to Telegram, confirm a token is available through `TELEGRAM_BOT_TOKEN`, `HANDOFF_BOT_TOKEN`, or the env file at `HANDOFF_ENV_FILE`, and confirm `HANDOFF_CHAT_ID` is set; running without a token is the intentional file-only mode, not an error.
- If expected content is missing from a handoff, it may have been dropped by the redaction pass, so check the omitted-line count in the run summary.
- If the script exits with code 1, at least one topic failed; re-running after fixing the underlying database or network problem will regenerate that topic's handoff.

## Verification

- The script exits with code 0 when every topic is processed successfully, including dry runs and file-only runs, and with code 1 when any topic fails; check with `echo $?` in the terminal.
- Confirm one Markdown file per topic exists at `HANDOFF_OUT_DIR/<YYYY-MM-DD>/<thread id>-<topic slug>.md` (with `HANDOFF_OUT_DIR` defaulting to `~/.hermes/topic-handoffs`), plus `HANDOFF_OUT_DIR/latest/<thread id>.md`, each titled `Handoff: <topic name> (topic <id>)`, and, when `HANDOFF_MIRROR_DIR` is set, that a copy exists under `HANDOFF_MIRROR_DIR/<YYYY-MM-DD>/`.
- Confirm the printed run summary includes the count of lines omitted by the redaction pass.

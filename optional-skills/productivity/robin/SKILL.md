---
name: robin
description: Save, search, and review notes in a local commonplace book.
version: 0.3.4
author: Nitin Gupta
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [productivity, knowledge-management, notes, commonplace, spaced-repetition]
    requires_toolsets: [terminal]
    category: productivity
---

# Robin Skill

Robin is a local commonplace book for saving, organizing, searching, and reviewing durable knowledge — notes, quotes, articles, links, images, and video references. It stores everything in a user-controlled state directory and needs no external services, API keys, or package installation. It is not a reminder, calendar, or secrets store.

## When to Use

- The user wants to save something for later recall.
- The user wants to organize knowledge by topic and tags.
- The user wants to search or review previously saved entries.
- The user wants to move or delete an existing entry.
- The user wants to check the health of the local Robin library.

Do not use Robin for short-lived reminders, calendar events, secrets, credentials, or files the user does not want persisted.

## Prerequisites

- Python 3.11+ and a POSIX shell (macOS or Linux) via the `terminal` toolset.
- A Robin state directory, by default `~/.hermes/data/robin`. Robin reads it from `--state-dir` or the `ROBIN_STATE_DIR` environment variable; prefer passing `--state-dir` explicitly in agent-run commands.
- All commands are run from this skill's directory so the bundled runtime under `scripts/robin/` is importable.

## How to Run

Initialize the state directory once before first use:

```bash
mkdir -p ~/.hermes/data/robin/topics ~/.hermes/data/robin/media
printf '{}\n' > ~/.hermes/data/robin/robin-config.json
python3 scripts/doctor.py --state-dir ~/.hermes/data/robin --json
```

`robin-config.json` may stay `{}`; `topics_dir`/`media_dir` default to `topics`/`media` inside the state directory and must remain relative descendants of it.

## Quick Reference

Save a text entry (`--topic` and `--description` are always required):

```bash
python3 scripts/add_entry.py --state-dir ~/.hermes/data/robin --topic "AI" --content "Useful note" --description "Short label" --tags ai,notes
python3 scripts/add_entry.py --state-dir ~/.hermes/data/robin --topic "Reading" --content "Key takeaway" --description "Article title" --source "https://example.com"
```

Save an image entry (media entries also require `--creator`, `--published-at`, `--summary`):

```bash
python3 scripts/add_entry.py --state-dir ~/.hermes/data/robin --entry-type image --topic "Design" --media-path ~/Downloads/shot.png --description "Reference screenshot" --creator "Jane Doe" --published-at 2026-01-15 --summary "Landing page layout"
```

Save a video reference (URL only — Robin never copies video files):

```bash
python3 scripts/add_entry.py --state-dir ~/.hermes/data/robin --entry-type video --topic "Talks" --media-url "https://example.com/watch" --description "Great talk" --creator "Jane Doe" --published-at 2026-01-15 --summary "Talk on X"
```

Search (the query is a **positional** argument, not a flag):

```bash
python3 scripts/search.py --state-dir ~/.hermes/data/robin "takeaway"
python3 scripts/search.py --state-dir ~/.hermes/data/robin --topic "Reading" --json
python3 scripts/search.py --state-dir ~/.hermes/data/robin --tags ai,notes
```

Review (surfaces one candidate; there is no `--limit`):

```bash
python3 scripts/review.py --state-dir ~/.hermes/data/robin
python3 scripts/review.py --state-dir ~/.hermes/data/robin --active-review --json
python3 scripts/review.py --state-dir ~/.hermes/data/robin --status --json
python3 scripts/review.py --state-dir ~/.hermes/data/robin --rate <entry-id> 5
```

Move or delete entries (`--move ID --topic TOPIC`, or `--delete ID`):

```bash
python3 scripts/entries.py --state-dir ~/.hermes/data/robin --move <entry-id> --topic "New Topic"
python3 scripts/entries.py --state-dir ~/.hermes/data/robin --delete <entry-id>
```

List topics, rebuild the review index, or check health:

```bash
python3 scripts/topics.py --state-dir ~/.hermes/data/robin --json
python3 scripts/reindex.py --state-dir ~/.hermes/data/robin --json
python3 scripts/doctor.py --state-dir ~/.hermes/data/robin
```

Add `--json` to any command for machine-readable output. See `references/guide.md` for the full CLI reference and JSON contracts.

## Procedure

1. Confirm (or initialize) the state directory and run `doctor.py` if unsure of its health.
2. When saving, choose a topic and write a concise 2-3 sentence `--description`; add tags only when they aid future search or review.
3. For images, pass a local `--media-path` and let Robin copy it into the state directory. For videos, pass a `--media-url` reference only.
4. Respect duplicate warnings — pass `--allow-duplicate` only when the user explicitly wants another copy.
5. To recall, use `search.py` with a positional query, `--topic`, or `--tags`; use `review.py` to surface an item and `--rate <id> <1-5>` to record a rating.
6. Before deleting, confirm the entry ID and the user's intent.

## Pitfalls

- The search query is positional; `--query` is not a valid flag.
- `review.py` surfaces a single best candidate and has no `--limit` option.
- Entry moves/deletes use `--move ID --topic TOPIC` and `--delete ID`; there is no `--id`/`--move-to`.
- `topics_dir`/`media_dir` in `robin-config.json` must be relative paths inside the state directory; absolute or `..` values are rejected.
- Media entries (`image`/`video`) require `--creator`, `--published-at`, and `--summary`; video entries take `--media-url`, not a local file.

## Verification

After setup or changes, confirm the library is healthy:

```bash
python3 scripts/doctor.py --state-dir ~/.hermes/data/robin --json
```

For a non-destructive, end-to-end integration check against a temporary state directory:

```bash
python3 scripts/selftest.py
```

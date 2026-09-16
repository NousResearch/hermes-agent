# Robin Guide

This guide is for advanced users who want manual control over Robin: state-directory setup, config files, CLI usage, file layout, and troubleshooting.

Important: Robin requires Python 3.11 or newer.

## Storage Model

Robin keeps both saved content and review state inside the state directory.

Recommended layout:

```text
agent-workspace/
  data/
    robin/
      robin-config.json
      robin-review-index.json
      media/
        poetry/
          20260409-a1f3c9.png
      topics/
        wisdom.md
        poetry.md
        quotes.md
```

- The Robin state directory stores config, review metadata, topic files, and copied images.
- Robin does not guess where its state lives.

Typical host examples:

- Hermes: `~/.hermes/data/robin/`
- OpenClaw: `~/.openclaw/workspace/data/robin/`

## Runtime Contract

Every Robin command needs a state directory.

Robin accepts both:

- `--state-dir /path/to/data/robin`
- `ROBIN_STATE_DIR=/path/to/data/robin`

Precedence:

1. `--state-dir`
2. `ROBIN_STATE_DIR`
3. otherwise Robin exits with an error

Expected files inside the state directory:

- `robin-config.json`
- optionally `robin-review-index.json`

If neither `--state-dir` nor `ROBIN_STATE_DIR` is present, Robin exits with:

```text
Robin state directory is not configured. Pass --state-dir or set ROBIN_STATE_DIR.
```

## Manual Setup

Create a state directory:

```bash
mkdir -p /path/to/agent-workspace/data/robin
mkdir -p /path/to/agent-workspace/data/robin/topics
mkdir -p /path/to/agent-workspace/data/robin/media
```

Create `/path/to/agent-workspace/data/robin/robin-config.json`. The file is required, but it may be an empty JSON object (`{}`):

```json
{
  "topics_dir": "topics",
  "media_dir": "media",
  "min_items_before_review": 30,
  "review_cooldown_days": 60
}
```

Robin does not need a separate content-root path. Topic files and copied media live inside the state directory under `topics/` and `media/`.

All fields inside `robin-config.json` are optional. Robin defaults to:

- `topics_dir`: `topics`
- `media_dir`: `media`
- `min_items_before_review`: `30`
- `review_cooldown_days`: `60`

Optional: create `/path/to/agent-workspace/data/robin/robin-review-index.json`:

```json
{
  "items": {}
}
```

If the review index file is missing, Robin starts with an empty index and writes the file when review state is saved.

Then either export the state dir:

```bash
export ROBIN_STATE_DIR=/path/to/agent-workspace/data/robin
```

Or pass it explicitly on each command:

```bash
python3 scripts/topics.py --state-dir /path/to/agent-workspace/data/robin
```

This is also the simplest setup verification step. A healthy empty setup returns `No topics yet. Start filing things with Robin!`

For a fuller integration check that does not touch the user's real library, run:

```bash
python3 scripts/selftest.py
```

For a non-destructive setup check against a real state directory, run:

```bash
python3 scripts/selftest.py --state-dir /path/to/agent-workspace/data/robin
```

## Topic Files

Robin stores content in topic-organized Markdown files under `topics/`.

Topic filenames use lowercase slugs with non-alphanumeric characters normalized to dashes.

Examples:

- `Song Lyrics` -> `song-lyrics.md`
- `AI/ML` -> `ai-ml.md`

Entries are separated by `***`. Each entry has frontmatter, then a blank line, then optional body text.

Text entries may omit `entry_type`; omitted `entry_type` is parsed as `text`.

Text example:

```text
id: 20260408-a1f3c9
date_added: 2026-04-08
description: A short excerpt from a Paul Graham essay about optimizing for what matters. Useful as a general reminder when making tradeoff decisions.
source: https://example.com/article
tags: [ai, reasoning]

Notable excerpt or the thing you sent.
```

Image example:

```text
id: 20260408-b7k2d1
date_added: 2026-04-08
entry_type: image
media_kind: image
media_source: media/poetry/20260408-b7k2d1.png
description: A photographed poem excerpt worth revisiting for tone and imagery.
creator: Mary Oliver
published_at: 1986
summary: An excerpt about attention and observation in everyday life.
tags: [poetry]

Opening lines from the photographed page.
```

Field meanings:

- `id`: stable entry identifier
- `date_added`: entry date
- `entry_type`: `text`, `image`, or `video`
- `media_kind`: same as `entry_type` for media entries; omitted for text entries, including text entries with image attachments
- `media_source`: copied relative path for local images or external URL for videos; may be present on text entries when `--media-path` is used
- `source`: original source URL when available
- `description`: required context for every entry
- `creator`, `published_at`, `summary`: required for media entries
- `tags`: optional tag list

## Topic and Content Policy

- Start filing by running `python3 scripts/topics.py --state-dir <state-dir> --json`.
- Choose a topic by name when there is a clear match.
- If topic names alone are ambiguous, inspect relevant topic files or use host search for more context.
- Prefer reusing an existing topic over creating a near-duplicate.
- Prefer durable, reusable topics such as `quotes`, `wisdom`, `poetry`, or `talks`.
- Create a new topic only when no existing topic clearly fits.
- Ask the user when two existing topics are both plausible.
- `add_entry.py` blocks deterministic duplicates by default when an existing entry has the same source URL, same media reference, or same normalized body text.
- Use `--allow-duplicate` only when the duplicate is intentional.
- For near-duplicates with meaningful differences, save only when the difference is worth preserving and explain the difference in `description`.
- Robin has no hard body-size limit, but agents should summarize very long articles or transcripts unless the user explicitly asks to store the full text.

Field semantics:

- `description`: required context for every entry; why the item matters and how to recognize it later.
- `summary`: required only for media entries; what the media itself contains.
- `note`: optional curation commentary, reminders, or connections to other entries.
- `tags`: pass on the CLI as one comma-separated string, for example `--tags "writing,clarity"`. Robin stores them as a frontmatter list.

## Media Rules

Robin accepts media with these rules:

- local image files: accepted and copied into `media/<topic-slug>/`; Robin creates the topic subdirectory automatically
- text entries require at least one payload: `content`, `note`, `source`, or local image attachment with `--media-path`
- text entries may attach a local image with `--media-path`; Robin keeps `entry_type` as `text`, sets `media_source`, and does not require media metadata
- remote image URLs: not supported directly by Robin's CLI
- video URLs: accepted and stored by reference
- uploaded or local video files: rejected; if the user cannot provide a shareable `http(s)` URL, save a normal `text` entry with the local path/context and tell the user Robin did not store the video file itself

Robin will not store a media entry unless the caller provides:

- `description`
- `creator`
- `published_at`
- `summary`

If a media item is rejected, Robin stores nothing and returns an error.

Robin also rejects any entry whose serialized body would contain a standalone `***` line, because `***` is Robin's internal entry separator.

## Search: Host Index vs Robin Search

If your agent supports file indexing, it should index Robin topic files like any other Markdown content.

Use host/global search for:

- broad semantic recall across all user content
- exploratory queries where Robin may be only one source

Use `robin-search` for:

- Robin-specific lookup
- topic filtering
- tag filtering
- deterministic lookup of Robin entries
- structured JSON output with stable ids, metadata, and ratings
- fallback when host indexing is unavailable or stale

`robin-search` can combine filters. If both `--topic` and `--tags` are provided, Robin first narrows to the topic and then applies the tag filter within that topic.

## CLI Reference

Default repo-local commands for agents:

- `python3 scripts/add_entry.py`
- `python3 scripts/doctor.py`
- `python3 scripts/entries.py`
- `python3 scripts/review.py`
- `python3 scripts/reindex.py`
- `python3 scripts/search.py`
- `python3 scripts/selftest.py`
- `python3 scripts/topics.py`

Optional installed entry points for advanced users:

- `robin-add`
- `robin-doctor`
- `robin-entries`
- `robin-review`
- `robin-reindex`
- `robin-search`
- `robin-topics`

All Robin commands support `--state-dir`.

Use `--json` whenever command output needs to be parsed programmatically. Without `--json`, Robin prints human-readable text for interactive use; that text is not a stable machine contract.

CLI flags by command:

- `add_entry.py`: `--state-dir`, `--topic`, `--entry-type text|image|video`, `--content`, `--description`, `--source`, `--media-path`, `--media-url`, `--creator`, `--published-at`, `--summary`, `--note`, `--tags`, `--allow-duplicate`, `--json`
- `doctor.py`: `--state-dir`, `--json`
- `entries.py`: `--state-dir`, `--delete ID`, `--move ID --topic TOPIC`, `--json`
- `review.py`: `--state-dir`, `--status`, `--active-review`, `--rate ID RATING`, `--json`
- `search.py`: `--state-dir`, optional positional `query` string, `--topic`, `--tags`, `--json`
- `selftest.py`: optional `--state-dir` for non-destructive setup checks, `--keep-temp`
- `topics.py`: `--state-dir`, `--json`
- `reindex.py`: `--state-dir`, `--json`

Recommended path for agents:

- run the repo-local `python3 scripts/*.py` commands directly

Optional path for advanced users:

- `pip install -e .`
- then use the installed `robin-add`, `robin-doctor`, `robin-entries`, `robin-review`, `robin-reindex`, `robin-search`, and `robin-topics` entry points

The repo-local `python3 scripts/*.py` commands work without `pip install -e .` or manual path setup.

Examples:

```bash
python3 scripts/search.py --state-dir /path/to/data/robin "clear thinking" --json
python3 scripts/review.py --state-dir /path/to/data/robin --json
python3 scripts/review.py --state-dir /path/to/data/robin --active-review --json
python3 scripts/review.py --state-dir /path/to/data/robin --status --json
python3 scripts/review.py --state-dir /path/to/data/robin --rate 20260408-a1f3c9 5
python3 scripts/review.py --state-dir /path/to/data/robin --rate 20260408-a1f3c9 5 --json
python3 scripts/doctor.py --state-dir /path/to/data/robin
python3 scripts/doctor.py --state-dir /path/to/data/robin --json
python3 scripts/entries.py --state-dir /path/to/data/robin --move 20260408-a1f3c9 --topic "AI Reasoning" --json
python3 scripts/entries.py --state-dir /path/to/data/robin --delete 20260408-a1f3c9 --json
python3 scripts/search.py --state-dir /path/to/data/robin --topic "AI Reasoning" --json
python3 scripts/search.py --state-dir /path/to/data/robin --tags "writing,clarity" --json
python3 scripts/search.py --state-dir /path/to/data/robin --topic "AI Reasoning" --tags "clarity" --json
python3 scripts/topics.py --state-dir /path/to/data/robin --json
python3 scripts/selftest.py
python3 scripts/selftest.py --state-dir /path/to/data/robin
python3 scripts/add_entry.py --state-dir /path/to/data/robin --topic "reasoning" --content "The most important thing is to decide what you are optimizing for." --description "A short Paul Graham line about choosing the objective before optimizing. Useful when reviewing tradeoff-heavy decisions." --json
python3 scripts/add_entry.py --state-dir /path/to/data/robin --topic "writing" --content "Write as if speaking to a smart friend." --description "A reminder to keep prose conversational and clear." --source "https://example.com/article" --note "Pair this with other writing advice." --json
python3 scripts/add_entry.py --state-dir /path/to/data/robin --topic "reasoning" --content "The map is not the territory." --description "A reminder that abstractions are not reality itself." --tags "thinking,quotes" --json
python3 scripts/add_entry.py --state-dir /path/to/data/robin --topic "wisdom" --content "Filed this screenshot to wisdom." --description "A text note with a local screenshot attached for later context." --media-path ~/Downloads/screenshot.png --json
python3 scripts/add_entry.py --state-dir /path/to/data/robin --entry-type image --topic "poetry" --media-path ~/Downloads/poem.png --description "A photographed poem excerpt worth revisiting." --creator "Mary Oliver" --published-at "1986" --summary "An excerpt about attention and observation." --json
python3 scripts/add_entry.py --state-dir /path/to/data/robin --entry-type video --topic "talks" --media-url "https://www.youtube.com/watch?v=abc123" --description "A talk to revisit for its framing and examples." --creator "Speaker Name" --published-at "2025-01-01" --summary "A concise summary of the talk." --json
python3 scripts/reindex.py --state-dir /path/to/data/robin
python3 scripts/reindex.py --state-dir /path/to/data/robin --json
```

The examples above use the repo-local `python3 scripts/*.py` path. If you installed the package with `pip install -e .`, the `robin-*` entry points are equivalent aliases.

All CLI helpers support `--json`.

Use `python3 scripts/doctor.py --state-dir <state-dir> --json` to check config, topic parsing, local media references, and review-index drift without changing the library. Doctor is read-only; use `python3 scripts/reindex.py --state-dir <state-dir> --json` when it reports review-index drift.

Use `python3 scripts/entries.py --state-dir <state-dir> --move <id> --topic <topic>` to move an entry, or `python3 scripts/entries.py --state-dir <state-dir> --delete <id>` to delete one. Delete removes the entry and its review-index item but keeps copied media files. `search.py` and `topics.py` remain read-only.

Use `python3 scripts/reindex.py --state-dir <state-dir>` after manual edits to topic files, when rebuilding review state from existing markdown, or when importing legacy entries and wanting the review index rebuilt from disk.

## Review System

Robin maintains a review index keyed by entry `id`. It stores:

- `rating`
- `last_surfaced`
- `times_surfaced`

Scheduled recall means Robin resurfaces an item for learning. It is not an active review session. Cron or scheduled recall messages should use the same recall template for every entry type and should not ask the user to reply with a bare `1`-`5` rating.

If the host scheduler supports skill metadata, scheduled recall jobs should load the `robin` skill. If it does not, the cron prompt must explicitly follow this review behavior.

Default recall text output:

```text
📚 Robin Recall

Topic: <topic>
Type: <entry_type>
Source: <source if present, else media_source if present, else "Not provided">
Creator: <creator if present, else "Not provided">
Saved on: <date_added> (<N> days ago)

Description:
<description if present, else "Not provided">

Body:
<body if present, else "Not provided">
```

Review behavior:

1. Robin waits until there are at least `min_items_before_review` items.
2. It skips items surfaced within `review_cooldown_days`.
3. It prefers items with the fewest total prior surfaces.
4. It prefers never-surfaced items before previously surfaced items, then prefers the oldest prior surface.
5. If another topic is eligible, it avoids surfacing the same topic that was surfaced most recently.
6. Ratings are only a late tie-breaker; they do not lead scheduled recall ordering.
7. Robin lightly randomizes within the best eligible pool so overloaded topics do not dominate recall.
8. Scheduled recall is the default: `python3 scripts/review.py --state-dir <state-dir>` increments `times_surfaced`, sets `last_surfaced`, and keeps `_awaiting_rating` as `false`.
9. Active review is explicit: `python3 scripts/review.py --state-dir <state-dir> --active-review` increments `times_surfaced`, sets `last_surfaced`, and marks `_awaiting_rating` as `true`.
10. A subsequent `--rate` call for that actively surfaced item overwrites the previous rating and sets `_awaiting_rating` back to `false` without incrementing `times_surfaced` again.

If `--rate` is called directly on an item that was not surfaced first, Robin still sets `last_surfaced`, increments `times_surfaced`, and keeps `_awaiting_rating` as `false`.

Preferred rating flow:

- Use `python3 scripts/review.py --state-dir <state-dir> --active-review --json` to surface an item in a live active-review session.
- After the user rates that surfaced item, call `python3 scripts/review.py --state-dir <state-dir> --rate <id> <rating> --json`.
- Use direct `--rate` without a prior surface only for manual corrections or when the user explicitly names an existing entry id.
- Do not request ratings in scheduled recall messages. Only rate during active review sessions or when the user explicitly names a Robin item id.

Setup guidance for hosts: ask the user how often recall should happen and when it should run. If the host supports scheduling, a daily or weekly recall trigger is the normal default. Otherwise, keep active review available as an on-demand command.

Example index shape:

```json
{
  "items": {
    "20260408-a1f3c9": {
      "id": "20260408-a1f3c9",
      "topic": "quotes",
      "date": "2026-04-08",
      "rating": null,
      "last_surfaced": null,
      "times_surfaced": 0,
      "_awaiting_rating": false
    }
  }
}
```

`_awaiting_rating` is an internal review-state flag. It is `true` only after Robin surfaces an item with `--active-review` and becomes `false` again after that surface is rated. Scheduled recall leaves it `false`.

## Troubleshooting

- Config not found:
  Create `robin-config.json` in the state directory and pass `--state-dir` or set `ROBIN_STATE_DIR`.
- Config has invalid JSON:
  Recreate `robin-config.json` as `{}` or with the supported config fields.
- Review index not found:
  Robin can start without it. If you want to create it manually, use `{"items": {}}`, or run `python3 scripts/reindex.py --state-dir <state-dir> --json` to rebuild from topic files.
- Review index has invalid JSON:
  Back up or recreate `robin-review-index.json` as `{"items": {}}`, then run `python3 scripts/reindex.py --state-dir <state-dir> --json` to rebuild from topic files.
- Library health is uncertain:
  Run `python3 scripts/doctor.py --state-dir <state-dir> --json` for a read-only diagnostic report.
- Media entry rejected:
  Ensure the caller provided `description`, `creator`, `published_at`, and `summary`.
- Local video rejected:
  Provide an `http(s)` video URL instead, or save a normal text entry with the local path/context and note that Robin did not store the video file.
- Image copy failed:
  Confirm the local image path exists and the `media/` directory under the state dir is writable.

---
name: apple-voice-memos
description: "Read Apple Voice Memos transcripts on macOS, preferring Apple's embedded transcript and falling back to cached local Whisper medium transcripts."
version: 1.2.0
author: Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [VoiceMemos, Apple, macOS, transcription, note-taking]
    related_skills: [apple-notes, apple-reminders]
prerequisites:
  commands: [python3]
---

# Apple Voice Memos

Read and search transcripts of the user's Apple Voice Memos. On macOS 15+/iOS 18+
Voice Memos transcribes recordings on-device and embeds the transcript text
directly inside each `.m4a` file (a QuickTime `tsrp` atom). This skill reads that
text first, then falls back to a transcript cached by the local Whisper worker
when Apple has not embedded one.

Recording metadata (title, date, duration) comes from the Voice Memos database,
opened read-only.

## When to Use

- User asks what a voice memo said, or to summarize / search their voice memos
- User wants recent memos turned into notes, tasks, or summaries
- "What did I record yesterday?", "read my last voice memo", "find the memo about X"

## When NOT to Use

- Transcribing arbitrary audio files (not Voice Memos) → use the transcription tool
- Recording new audio → that's the built-in `/voice` mode
- Memos with no embedded transcript (very old, very short, or not yet processed) —
  these show `[ ]` in `list`; the script reports when a transcript is missing

## Quick Reference

Script: `scripts/voicememos.py` (run with `python3`). All commands take `--json`.

### List recordings (newest first)

```bash
python3 scripts/voicememos.py list                      # 30 most recent
python3 scripts/voicememos.py list --limit 10
python3 scripts/voicememos.py list --with-transcript    # only ones with text
python3 scripts/voicememos.py list --search "pickwick"  # filter by title
```

`[A]` = Apple transcript, `[W]` = cached local Whisper medium transcript,
`[ ]` = none. The leading `[n]` is the index used by `transcript`.

### Read one transcript

```bash
python3 scripts/voicememos.py transcript 1              # by list index
python3 scripts/voicememos.py transcript "Recording 45" # by title search
python3 scripts/voicememos.py transcript <filename.m4a> # by filename
```

### Dump many transcripts at once (good for summarizing a batch)

```bash
python3 scripts/voicememos.py dump --limit 10 --only-transcribed
python3 scripts/voicememos.py dump --search "meeting" --json
```

## Building automation on top of this skill

This skill is the **transcript/metadata access layer**. It is a good base for
custom workflows that sit on top of it, such as:

- a cron feeder that only emits newly transcribed memos
- a pipeline that turns memos into Apple Reminders or Apple Notes
- a durable memo-processing log

If you build that kind of workflow, prefer importing or reusing this skill's
script/functions instead of forking transcript logic in a second place. Keep the
Voice Memos parsing here, and keep inbox state, deduping, reminder creation, and
logging in your custom automation layer.

## How It Works

- **Metadata:** read-only SQLite query against
  `~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings/CloudRecordings.db`
  (`ZCLOUDRECORDING` table → title, date, duration, filename).
- **Transcript:** the `.m4a` contains a `tsrp` user-data atom holding JSON
  (`attributedString.runs`); the string elements concatenated are the transcript.
- **Whisper fallback:** if Apple has no embedded transcript, the skill reads
  `~/.voicememo-whisper/transcripts/<unique_id>.txt` when available.
- Dates are stored in Cocoa epoch (2001-01-01) and converted to local time.

## Rules

1. Prefer Apple's embedded transcript; use cached local Whisper text only as a fallback.
2. The database is opened **read-only**; never write to Voice Memos files or DB.
3. When a requested memo has no Apple or cached Whisper transcript, say so plainly.
4. To save a memo's transcript somewhere durable, combine with the `apple-notes`
   skill or the `memory` tool.
5. For automation, treat this skill as the canonical Voice Memos reader and layer
   cron/stateful workflows on top rather than duplicating transcript extraction.

## Limitations

- macOS only. Requires Voice Memos to have transcribed the recording on-device
  (governed by Apple; English and other supported languages only).
- Memos still syncing from iCloud may not have their `.m4a` downloaded locally yet
  (`exists: false` in `--json`).

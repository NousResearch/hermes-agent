---
name: hermes-session-recovery
description: >
  Recover Hermes session history when the persistent session store (sessions.db)
  is empty, missing, corrupt, or does not contain the target session.
  Covers log-based reconstruction, paste-file archaeology, and structured
  reporting so the user gets value even when --resume fails.
triggers:
  - "resume SESSION_ID"
  - "resume failed"
  - "session not found"
  - "sessions.db is empty"
  - "can't resume"
  - "what was in that session"
  - uncommitted work visible in paste files that does not match live session
dependencies:
  - terminal
  - file
  - search_files
---

# Hermes Session Recovery

When `--resume <session_id>` fails because the session store is empty or the
record is gone, this skill provides a fallback recovery path. The aim is to
reconstruct enough context from agent logs and paste files so the user can
continue work without needing the exact session restored.

## When this skill applies

- `hermes --resume <id>` returns nothing or an error.
- `hermes sessions list` shows the session but `--resume` does not load it.
- `sessions.db` exists but is 0 bytes (empty / uninitialised / post-migration).
- The user asks "what happened in session X?" after the store is gone or stale.

## Detection (always run first)

```bash
# 1. Check the raw physical store
ls -la ~/.hermes/sessions.db

# 2. Confirm Hermes sees nothing
date; hermes sessions list --limit 200 2>&1 | grep -i SESSION_ID || true

# 3. Verify the session existed in the logs
grep SESSION_ID ~/.hermes/logs/agent.log | head -5
```

If `sessions.db` is 0 bytes, the SQLite store is empty or has been reset. Session
metadata and message history are not queryable. Recovery must come from
discriminative sources.

## Recovery path A — agent.log archaeology

The main log at `~/.hermes/logs/agent.log` records every tool call, model
switch, and turn-end event tagged with the session ID.

### Quick identity check

```bash
grep "\[SESSION_ID\]" ~/.hermes/logs/agent.log | head -30
```

This reveals:
- Exact start and end timestamps.
- Models used (the `model=` field on API call lines).
- Turn counts (the `history=` field on conversation turn lines).
- Tool calls executed (the `tool_executor` lines).

### Reconstructing what was done

```bash
# All terminal commands issued in that session
awk '/\[SESSION_ID\] agent.tool_executor: tool terminal completed/ {print $0}' \
  ~/.hermes/logs/agent.log

# All tool calls of any kind with their durations
awk '/\[SESSION_ID\] agent.tool_executor/ {print $0}' ~/.hermes/logs/agent.log | \
  sed 's/.*tool \(.*\) completed.*/\1/' | sort | uniq -c | sort -rn

# Turn summaries (model, tool turns, response length)
awk '/\[SESSION_ID\]/ && /Turn ended:/ {print $0}' ~/.hermes/logs/agent.log
```

### Reconstructing user messages

When a turn starts, the log truncates the message content into the `msg=`
field. Extract these to see what the user asked:

```bash
awk '/\[SESSION_ID\] agent.conversation_loop: conversation turn/' \
  ~/.hermes/logs/agent.log | grep -o "msg='.*'"
```

The content is trimmed (often at ~100 chars), but it is usually enough to
identify the topic.

## Recovery path B — paste-file archaeology

Hermes writes multi-line pastes collapsed during input to
`~/.hermes/pastes/paste_N_TIMESTAMP.txt`. If the session involved large pasted
text, those files survive independently of sessions.db.

### Find paste files by timestamp

```bash
# Pastes created during the session window
find ~/.hermes/pastes -type f -newermt "START_TIME" ! -newermt "END_TIME" \
  2>/dev/null
```

Or, if exact time bounds are unknown, read the paste timestamps in the
filename (paste_N_HHMMSS.txt) and correlate with the session window.

### Correlate with CLI collapse log lines

Look for:
```
Collapsed paste #N: X lines, Y chars -> .../pastes/paste_N_TIMESTAMP.txt
```
in the log. That maps the paste to the exact session turn.

## Recovery path C — session_search

If the session ever entered the message store (even if sessions.db is now
corrupt), try the built-in Hermes session search:

```bash
hermes sessions list --limit 500 | grep SESSION_ID
```

If a title appears but `--resume` still does not work, the session may be
listed-only; the actual messages may be gone. Fall back to log archaeology.

## Structured recovery report

Once you have reconstructed what you can, report to the user in this shape:

| Field | Value |
|---|---|
| Session ID | the ID |
| Time window | start → end |
| Models used | list |
| Tool calls | count by type |
| User topic | from msg= snippets |
| Key output | what was produced (files, pastes, memory writes) |
| Outcome | what the session ended with |
| Cannot recover | what is genuinely lost (full turn text without paste) |

## Prevention

- The session store (sessions.db) can become 0 bytes after Hermes version
  migrations or db resets. Do not rely on it being the sole source of truth
  for past session content.
- Paste files (`~/.hermes/pastes/`) and agent logs are durable and survive
  session store wipes. Treat them as the secondary archive.
- For sessions producing durable decisions (approval rules, architecture
  choices, policy changes), always write them into the `governance/` channel or
  a skill, not just into session memory.

## Pitfalls

- **Do NOT tell the user "session not found" and stop.** If the ID appeared
  in `agent.log`, the session existed. The store is just empty.
- **Do NOT waste time trying `hermes sessions export` on an empty db.** It will
  produce nothing. Go straight to the log.
- **Do NOT try `--resume` more than once** once you know sessions.db is 0
  bytes. It will not help. Pivot to reconstruction.
- **Log truncation matters.** If `agent.log` has rotated (via logrotate or
  manual truncation), older sessions may be genuinely lost. Check
  `~/.hermes/logs/` for rotated files:
  ```bash
  ls -la ~/.hermes/logs/agent.log*
  ```
- **Do NOT confuse cron session IDs with interactive session IDs.** Cron
  sessions look like `cron_<jobid>_YYYYMMDD_HHMMSS`. They are ephemeral and
  usually irrelevant for `--resume` requests.

## Limitations of log-based recovery

What you CAN recover:
- Which tools ran and for how long.
- What file paths were touched (from tool call traces).
- What paste files were created.
- What models were used and how many turns.
- The first ~100 characters of each user message.

What you CANNOT recover:
- Full assistant response text (except truncated summary in `response_len=`).
- Exact terminal command output (unless stored in a paste).
- Intermediate reasoning steps not emitted as tool calls.
- Images or binary data.
- Interactive user feedback between turns (not captured as "turn" entries).

When recovery is incomplete, explicitly tell the user what is missing rather
than silently omitting it.

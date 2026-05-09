# Telegram Chat History — Architecture & Limitations

## The Problem

Gordon asked: "can you see the HTML pages that you recently sent me?"

**Short answer:** No — not automatically.

## Why

The Telegram Bot API delivers messages to Hermes as discrete events. Each new session starts with a **fresh context window**. Hermes never receives or stores the full Telegram conversation history — it only gets:
1. The current incoming message
2. The session transcript from `/opt/data/sessions/` (what was in the model's context at session end)

Telegram stores history server-side and on the client app. The user sees it because they're using the Telegram app. Hermes sees nothing unless:
- A session file is in context (from memory/injection)
- `session_search` is explicitly called

## Session Files vs. Telegram History

| | Telegram History | Hermes Session Files |
|---|---|---|
| **Location** | Telegram servers + client | `/opt/data/sessions/` |
| **Content** | Full raw messages, media, HTML | Model context at session end |
| **Images/HTML** | Yes — fully preserved | No — stripped/summarized |
| **Automatic** | Yes — always visible to user | No — fresh context each session |
| **Searchable by Hermes** | No (without `session_search`) | Yes (via FTS5) |

## Relevant Code Paths

- Gateway Telegram adapter: `/opt/data/repo/gateway/platforms/telegram.py`
- Session storage: `/opt/data/repo/gateway/session.py` + `hermes_state.py`
- Session search tool: `/opt/data/repo/tools/session_search_tool.py`
- Session files: `/opt/data/sessions/session_*.json`
- State DB (SQLite with FTS5): `/opt/data/state.db`

## Session Search Tool

`session_search` searches past session transcripts stored in SQLite (FTS5). It:
1. Finds sessions matching a query
2. Loads the session JSON
3. Sends truncated transcript to a fast model for summarization
4. Returns per-session summaries with metadata

**Limitation:** It searches Hermes session transcripts, NOT raw Telegram history. Media (images, documents, HTML files) is not stored in session files.

## UX Guidance

When Gordon asks about something from earlier in the chat:
1. Acknowledge the limitation directly
2. Use `session_search` proactively if relevant
3. If the content was a file/HTML, ask Gordon to re-share it

This is NOT a bug — it's by design. The `session_search` tool exists to bridge this gap in a controlled, query-on-demand way.

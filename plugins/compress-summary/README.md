# Compress Summary Plugin

Injects a structured task-state summary into the conversation tail before
context compression. The summary survives compression (tail messages are
protected) and acts as an anchor so the model retains a clear picture of
completed work and pending goals.

## Problem

When context compression fires during long sessions, the model often loses
track of what was accomplished in the middle of the conversation. It remembers
the original request (in the protected head) and recent messages (in the
protected tail), but forgets intermediate progress — leading to repeated work
or abandoned sub-tasks.

## How It Works

1. Hooks into the `pre_compress` event
2. Scans messages to extract structured state:
   - **Original request** — first substantive user message
   - **Actions taken** — chronological log from tool calls (reliable, not regex)
   - **Progress notes** — assistant's status updates
   - **Recent instructions** — latest user messages
   - **Key files** — file paths from write/patch/read tool calls
3. Appends the summary as the last message, placing it in the compressor's
   protected tail region

## Requirements

None — pure Python, no external dependencies.

## Setup

Drop this plugin into `~/.hermes/plugins/compress-summary/` or keep it in the
repo's `plugins/compress-summary/` directory. It is discovered automatically.

## Config

No configuration needed. The plugin activates automatically when context
compression is triggered and the conversation has 10+ messages.

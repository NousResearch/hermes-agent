---
name: apple-platform
description: "Apple platform integration: Notes, Reminders, FindMy, iMessage — all via native macOS CLIs."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [Apple, macOS, Notes, Reminders, FindMy, iMessage]
prerequisites:
  commands: [memo, remindctl, imsg]
---

# Apple Platform Integration

Unified access to Apple ecosystem tools from the terminal. All tools sync via iCloud to the user's other Apple devices (iPhone, iPad, Mac).

## Quick Reference

| Tool | CLI | Use Case |
|------|-----|----------|
| Notes | `memo` | Create, search, edit Apple Notes |
| Reminders | `remindctl` | Task management synced to iOS |
| FindMy | AppleScript + vision | Track devices/AirTags via screen |
| iMessage | `imsg` | Send/receive iMessages and SMS |

---

## Apple Notes (`memo`)

See `references/apple-notes.md` for full details.

**Install:** `brew tap antoniorodr/memo && brew install antoniorodr/memo/memo`

**Grant:** System Settings → Privacy → Automation → Notes.app

**Core commands:**
```bash
memo notes                        # List all notes
memo notes -s "query"             # Search (fuzzy)
memo notes -a "Title"             # Quick add
memo notes -e                     # Edit (interactive)
memo notes -ex                    # Export to HTML/Markdown
```

**Key rules:**
- Cannot edit notes with images/attachments
- macOS only — requires Apple Notes.app
- For Markdown-native KB → use `obsidian` skill instead

---

## Apple Reminders (`remindctl`)

See `references/apple-reminders.md` for full details.

**Install:** `brew install steipete/tap/remindctl`

**Grant:** Reminders permission when prompted → `remindctl status`

**Core commands:**
```bash
remindctl                    # Today's reminders
remindctl today --json       # JSON output
remindctl add "Task" --due tomorrow
remindctl list               # All lists
remindctl complete 1 2       # By ID
```

**Key rules:**
- Clarify on "remind me" — Apple Reminders (iOS sync) vs agent cronjob alert
- Accepts natural dates: `today`, `tomorrow`, `YYYY-MM-DD HH:mm`

---

## Find My (FindMy.app + vision)

See `references/findmy.md` for full details.

**Prerequisites:** Screen Recording permission + optional `peekaboo`

**Two methods:**
1. AppleScript + screenshot (basic) — opens FindMy.app, captures window
2. Peekaboo UI automation (recommended) — precise element clicking

**Workflow for AirTag tracking:**
```bash
osascript -e 'tell application "FindMy" to activate'
sleep 3
screencapture -w -o /tmp/findmy.png
# Then vision_analyze(image_url="/tmp/findmy.png", question="...")
```

**Key rules:**
- Keep FindMy app in foreground when tracking — AirTags only update while page is open
- Use `vision_analyze` to read screenshot content
- For ongoing tracking → use a cronjob

---

## iMessage (`imsg`)

See `references/imessage.md` for full details.

**Install:** `brew install steipete/tap/imsg`

**Grant:** Full Disk Access (Privacy) + Automation → Messages.app

**Core commands:**
```bash
imsg chats --limit 10 --json        # List chats
imsg history --chat-id 1 --json     # Read history
imsg send --to "+1..." --text "Hi"  # Send
imsg watch --chat-id 1              # Watch for new messages
```

**Service options:** `--service imessage` (blue), `--service sms` (green), `--service auto` (default)

**Key rules:**
- Always confirm recipient + message before sending
- Never send to unknown numbers without explicit approval

---

## Rules (All Apple Tools)

1. Prefer Apple-native tools when user wants cross-device sync (iPhone/iPad/Mac)
2. Use `memory` tool for agent-only notes (no user-facing sync needed)
3. Use `obsidian` skill for Markdown-native knowledge management
4. Respect privacy — only operate on devices the user owns
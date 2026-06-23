---
name: apple-macos-apps
description: 'macOS Apple app integrations: Apple Notes (memo), Reminders (remindctl),
  iMessage (imsg), and Find My (AppleScript + vision). All macOS-only, iCloud-synced.'
version: 1.0.0
author: Hermes Agent
license: MIT
platforms:
- macos
metadata:
  hermes:
    tags:
    - Apple
    - macOS
    - Notes
    - Reminders
    - iMessage
    - FindMy
    - AirTag
    - iCloud
    - memo
    - remindctl
    - imsg
    related_skills:
    - obsidian
    - macos-computer-use
    config:
    - key: apple-macos-apps.enabled
      description: Enable apple-macos-apps skill behavior
      default: true
      prompt: Enable apple-macos-apps skill?
    category: productivity
---


# Apple macOS App Integrations

Four Apple-native app integrations that sync across Apple devices via iCloud. All require macOS.

**Quick-pick guide:**

| Goal | Tool | Section |
|------|------|---------|
| Create/search notes synced to iPhone | `memo` CLI | Apple Notes |
| Set reminders/to-dos synced to iPhone | `remindctl` CLI | Apple Reminders |
| Send/read iMessages or SMS | `imsg` CLI | iMessage / SMS |
| Find a device, AirTag, or item | AppleScript + screenshot | Find My |

---

## Apple Notes (memo)

Use `memo` to manage Apple Notes from the terminal.

### Prerequisites

- macOS with Notes.app
- Install: `brew tap antoniorodr/memo && brew install antoniorodr/memo/memo`
- Grant Automation access to Notes.app when prompted (System Settings → Privacy → Automation)

### When to Use / Not Use

- **Use** for notes that should sync across iPhone/iPad/Mac
- **Don't use** for Obsidian vault management (→ `obsidian` skill), Bear Notes, or agent-internal notes (→ `memory` tool)

### Quick Reference

```bash
memo notes                        # List all notes
memo notes -f "Folder Name"       # Filter by folder
memo notes -s "query"             # Search notes (fuzzy)
memo notes -a                     # Interactive create
memo notes -a "Note Title"        # Quick add with title
memo notes -e                     # Edit (interactive selection)
memo notes -d                     # Delete (interactive selection)
memo notes -m                     # Move note to folder
memo notes -ex                    # Export to HTML/Markdown
```

### Limitations

- Cannot edit notes containing images or attachments
- Interactive prompts require PTY (`pty=true` in terminal calls)

---

## Apple Reminders (remindctl)

Use `remindctl` to manage Apple Reminders from the terminal.

### Prerequisites

- macOS with Reminders.app
- Install: `brew install steipete/tap/remindctl`
- Grant Reminders permission: `remindctl authorize`
- Check status: `remindctl status`

### When to Use / Not Use

- **Use** when user says "remind me" and means a phone/iCloud reminder
- **Don't use** for agent alerts (→ `cronjob` tool), calendar events, or project tasks (→ GitHub Issues, Notion)
- **Clarify first** if "remind me" could mean either an agent cronjob or an Apple Reminder

### Quick Reference

```bash
remindctl                         # Today's reminders
remindctl today / tomorrow / week / overdue / all
remindctl 2026-06-15              # Specific date

remindctl list                    # All lists
remindctl list Work               # Specific list
remindctl list Projects --create  # Create list

remindctl add "Buy milk"
remindctl add --title "Call mom" --list Personal --due tomorrow
remindctl add --title "Meeting" --due "2026-06-15 09:00"

remindctl complete 1 2 3          # Complete by ID
remindctl delete 4A83 --force     # Delete by ID

remindctl today --json            # JSON for scripting
remindctl today --plain           # TSV format
```

### Date formats

`today`, `tomorrow`, `yesterday`, `YYYY-MM-DD`, `YYYY-MM-DD HH:mm`, ISO 8601.

---

## iMessage / SMS (imsg)

Use `imsg` to read and send iMessage/SMS via macOS Messages.app.

### Prerequisites

- macOS with Messages.app signed in to iCloud
- Install: `brew install steipete/tap/imsg`
- Grant Full Disk Access for terminal (System Settings → Privacy → Full Disk Access)
- Grant Automation permission for Messages.app when prompted

### When to Use / Not Use

- **Use** when user explicitly asks to send an iMessage, text, or SMS
- **Don't use** for Telegram/Discord/Slack/WhatsApp (→ gateway channels) or bulk messaging without explicit approval

### Quick Reference

```bash
# List recent chats
imsg chats --limit 10 --json

# View conversation history
imsg history --chat-id 1 --limit 20 --json
imsg history --chat-id 1 --limit 20 --attachments --json

# Send messages
imsg send --to "+141****4567" --text "Hello!"
imsg send --to "+141****4567" --text "See this" --file /path/to/image.jpg
imsg send --to "+141****4567" --text "Hi" --service imessage   # force iMessage
imsg send --to "+141****4567" --text "Hi" --service sms        # force SMS

# Watch for incoming messages
imsg watch --chat-id 1 --attachments
```

### Rules

1. **Always confirm recipient and message** before sending
2. **Never send to unknown numbers** without explicit user approval
3. **Verify file paths** before attaching
4. **Don't mass-send**

### Example workflow

```bash
# Find the contact
imsg chats --limit 20 --json | jq '.[] | select(.displayName | contains("Mom"))'
# Confirm with user, then send
imsg send --to "+155****4567" --text "I'll be late"
```

---

## Find My (AppleScript + vision)

Track Apple devices and AirTags via FindMy.app. No CLI/API — uses AppleScript UI automation + screenshot + `vision_analyze`.

### Prerequisites

- macOS with Find My app and iCloud signed in; devices/AirTags already registered
- Screen Recording permission for terminal (System Settings → Privacy → Screen Recording)
- **Optional but recommended:** `brew install steipete/tap/peekaboo` for more reliable UI automation

### When to Use

User asks "where is my [device/AirTag/keys/cat]?", checking iPhone/iPad/Mac/AirPods locations, or monitoring item movement over time.

### Method 1: AppleScript + screenshot (basic)

```bash
osascript -e 'tell application "FindMy" to activate'
sleep 3
screencapture -w -o /tmp/findmy.png
```

Then: `vision_analyze(image_url="/tmp/findmy.png", question="What devices/items are shown and where are they located?")`

Switch tabs:
```bash
osascript -e 'tell application "System Events"
    tell process "FindMy"
        click button "Items" of toolbar 1 of window 1
    end tell
end tell'
```

### Method 2: Peekaboo (recommended if installed)

```bash
osascript -e 'tell application "FindMy" to activate'
sleep 3
peekaboo see --app "FindMy" --annotate --path /tmp/findmy-ui.png
peekaboo click --on B3 --app "FindMy"
peekaboo image --app "FindMy" --path /tmp/findmy-detail.png
```
Then use `vision_analyze` on the detail screenshot.

### Periodic AirTag tracking

Keep FindMy in the foreground (AirTags only update while the page is visible); capture every 5 minutes:

```bash
while true; do
    screencapture -w -o /tmp/findmy-$(date +%H%M%S).png
    sleep 300
done
```

### Limitations / Rules

- No CLI/API; UI automation may break across macOS versions
- AirTags only update while FindMy page is actively displayed
- Use `vision_analyze` to read screenshots
- Only track devices/items the user owns

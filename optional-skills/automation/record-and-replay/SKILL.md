---
name: record-and-replay
description: |
  Record any workflow by performing it once — the agent watches your screen,
  captures your clicks and keystrokes globally across all apps, and generates a
  replayable skill. Replay the skill later to automate the task autonomously.
  Cadillac version: real input event capture via CGEventTap (macOS) or pynput
  (Linux/Windows), not just screenshot diffing. Works across all applications
  and browser windows.
version: 2.0.0
author: Eric Woodard (Snail3D)
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [automation, record, replay, computer-use, workflow, macro, cross-platform]
    category: automation
    related_skills: [apple, macos-computer-use]
    requires_toolsets: [terminal]
---

# Record & Replay

Record any workflow by performing it once. The agent captures your real input
events (mouse clicks, keystrokes, scrolls, drags) across **all applications**
— browsers, native apps, anything on screen — synchronized with periodic
screenshots and accessibility tree snapshots. It then analyzes the recording
and generates a replayable skill. Later, invoke the skill to have the agent
repeat the task autonomously via `computer_use` (macOS) or `browser_*` tools.

This is the "Cadillac" version — it captures actual system input events, not
just screenshot diffs. This means it knows exactly where you clicked, what keys
you pressed, and in what order, producing much more precise replay skills.

**Cross-platform:** Works on macOS (CGEventTap), Linux (pynput/X11), and
Windows (pynput/Win32).

## When to Use

- The user says "record this workflow" or "watch me do this"
- The user wants to automate a repetitive GUI task
- The user wants to create a reusable skill from a manual process
- The user says "replay" or "run the <name> skill"

## How to Launch from Hermes

Just tell Hermes what you want to record. Examples:

> "record this workflow"
> "watch me upload a video to YouTube"
> "record me doing my expense reports"
> "start recording — I'm going to show you how I process orders"

The agent will:
1. Load this skill
2. Start the recording script in the background
3. Tell you to go ahead and perform the task
4. Wait for you to say "stop" or "done"
5. Analyze the recording and generate a new skill
6. Confirm the skill is saved and ready to replay

Later, to replay:
> "run the youtube-upload skill"
> "replay my expense-report workflow"

## Requirements by Platform

### macOS (CGEventTap — Cadillac mode)

```bash
pip install pyobjc-framework-Quartz mss
```

- **cua-driver** (optional, for AX trees): Install via `hermes tools` →
  enable Computer Use, or directly:
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh)"
  ```
- **Permissions:** Terminal needs:
  - System Settings → Privacy & Security → **Accessibility** ✓
  - System Settings → Privacy & Security → **Screen Recording** ✓

### Linux (pynput + X11)

```bash
pip install pynput mss
# Screenshot backends (any one):
sudo apt install scrot       # or gnome-screenshot, or imagemagick
# Window info (optional, for AX tree fallback):
sudo apt install xdotool
# AT-SPI2 (optional, for accessibility trees):
pip install pyatspi
```

- On Wayland: pynput may not work (X11 only). Check if `XDG_SESSION_TYPE=x11`.
  If on Wayland, the script falls back to screenshot-only mode.

### Windows (pynput + Win32)

```bash
pip install pynput mss pygetwindow uiautomation
```

- No special permissions needed for standard windows.
- Run as admin if recording interactions with elevated applications.
- `uiautomation` package provides the UI Automation tree (replaces macOS AX
  trees).

## Phase 1: Record

### Starting a Recording

When the user says "record this workflow" (or similar):

1. **Check prerequisites:**
   ```bash
   # macOS
   python3 -c "import Quartz; print('Quartz OK')" 2>/dev/null
   # Linux/Windows
   python3 -c "import pynput; print('pynput OK')" 2>/dev/null
   ```

2. **Start the recording script** in the background via `terminal(background=true)`:
   ```bash
   python3 <skill_dir>/scripts/record_workflow.py \
     --interval 1.0 \
     --output-dir ~/.hermes/recordings/<descriptive-name>
   ```

   Faster intervals (0.5s) capture more detail but produce more screenshots.
   Default 1.0s is a good balance.

3. **Tell the user:**
   > 🔴 Recording started. Go ahead and perform your workflow.
   > I'm capturing your clicks, keystrokes, and screenshots across ALL apps.
   > Open any browser, any app, switch between them — I see everything.
   > When you're done, tell me to stop.

4. **Wait for the user** to say "stop" / "done" / "ok stop".

5. **Stop the recording** by creating the stop flag file:
   ```bash
   touch ~/.hermes/recordings/<descriptive-name>/.stop
   ```

### What Gets Captured (all platforms)

| Data | Method | Purpose |
|------|--------|---------|
| Mouse clicks (down/up, button, position) | CGEventTap / pynput | Know exactly where user clicked |
| Keystrokes (key, character, modifiers) | CGEventTap / pynput | Know exactly what was typed |
| Scrolls (direction, amount, position) | CGEventTap / pynput | Know scroll behavior |
| Mouse drags | CGEventTap / pynput | Know drag operations |
| App switches | NSWorkspace / xdotool / win32 | Know when user changed apps |
| Screenshots | mss (cross-platform) or screencapture | Visual context for each step |
| AX/AT/UI tree snapshots | cua-driver / pyatspi / uiautomation | Element names + structure |

### Recording works across ALL apps

The recording captures events at the OS level, not per-window. While recording
you can:
- Open any browser (Chrome, Safari, Firefox, Arc — doesn't matter)
- Switch between any apps (Finder, Mail, Figma, Terminal, anything)
- Click anywhere on screen, type in any field, scroll, drag
- Switch Spaces / virtual desktops (macOS) or workspaces (Linux)

### Password fields — handled, not captured

macOS **Secure Input Mode** deliberately blocks keyloggers from seeing
keystrokes in password fields (`AXSecureTextField`). This is a security
feature — the recording script cannot and should not bypass it.

When the recording encounters a password field, keystrokes are silently
skipped. The analysis script detects this pattern (click into a field →
no keystrokes → click elsewhere) and marks it as a password step.

During **replay**, the generated skill handles password fields via **password
manager integration** instead of recording the actual password:

1. **macOS Keychain** (built-in, no install):
   ```bash
   security find-generic-password -s "YouTube" -w
   ```
2. **1Password CLI** (`op`):
   ```bash
   op item get "YouTube Login" --field password
   ```
3. **Bitwarden CLI** (`bw`):
   ```bash
   bw get password "YouTube Login"
   ```

The generated skill includes a password step like:

```markdown
### Step 3: Enter password
- App: Safari
- Field: AXSecureTextField (password)
- Action: retrieve from keychain and type

# Retrieve password (never logged, never stored in the skill)
PASSWORD=$(security find-generic-password -s "YouTube" -w 2>/dev/null)
computer_use(action="type", text="$PASSWORD")
```

**Security rules for password steps in generated skills:**
- Never store the actual password in the SKILL.md — always reference the
  keychain/service name
- Never log the password value in events.jsonl
- The `type` action for passwords is not captured during recording — it's
  reconstructed during generation from context (the field was a
  `AXSecureTextField`, the user clicked into it, paused, then proceeded)
- If no password manager is configured, the replay pauses and asks the user
  to type the password manually

### Recording Output Structure

```
~/.hermes/recordings/<name>/
├── metadata.json              # Duration, event count, platform, etc.
├── events/
│   └── events.jsonl           # Streaming event log (one JSON per line)
├── screenshots/
│   ├── 0001.png               # Numbered screenshots
│   ├── 0002.png
│   └── ...
└── ax_trees/
    ├── 0001.txt               # AX/AT/UI tree at each screenshot
    ├── 0002.txt
    └── ...
```

## Phase 2: Generate Skill

After the recording stops, analyze it and generate a SKILL.md.

### Step 1: Run the analysis script

```bash
python3 <skill_dir>/scripts/analyze_recording.py \
  ~/.hermes/recordings/<name>
```

This produces a JSON summary with:
- Logical steps (grouped by 2s+ pauses or app switches)
- Each step's interactions (clicks, keys, scrolls)
- Associated screenshots
- Frontmost app for each step

### Step 2: Review screenshots with vision

For key steps, load the before/after screenshots and use `vision_analyze` to
understand what visually changed:

```
vision_analyze(
  image_url="file:///path/to/recording/screenshots/0001.png",
  question="What application is this? What UI elements are visible? What action is about to happen?"
)
```

### Step 3: Generate the SKILL.md

Write a new skill using `skill_manage(action='create')` with:

```markdown
---
name: <descriptive-name>
description: <one line summary>
version: 1.0.0
author: generated by record-and-replay
license: MIT
platforms: [<platform>]
metadata:
  hermes:
    tags: [automation, recorded]
    category: automation
    created_by: agent
    source_recording: ~/.hermes/recordings/<name>
---

# <Task Name>

## When to Use
Auto-generated from recording on <date>. Replay this workflow when the user
asks to <description of what the task does>.

## Prerequisites
- computer_use toolset enabled (macOS) or browser tools (web tasks)
- <any app-specific requirements>

## Procedure

### Step 1: <Action description>
- App: <app name>
- Action: click on <element description> at position (x, y)
- Verify: <what should appear/change>

computer_use(action="capture", mode="som", app="<app>")
computer_use(action="click", element=<N>)

### Step 2: <Action description>
...

## Verification
After all steps, capture a screenshot and verify:
- <expected final state>

## Pitfalls
- <known issues from the recording>
- <elements that might have different indices on replay>
```

### Key generation principles

1. **Use element indices, not pixel coordinates.** The recording has pixel
   positions, but replay should use `computer_use(action="capture", mode="som")`
   to get element indices, then click by index. Pixel positions break when
   windows move.

2. **Include verification steps.** After each action, re-capture and verify
   the expected state appeared. This makes replay robust to timing issues.

3. **Note app context.** Each step should specify which app to target via
   `app="AppName"` parameter.

4. **Handle variations.** If the recording shows the user retrying a click
   (clicked wrong element first), note the correct target and skip the retry.

5. **Group related actions.** If the user typed text into a field, combine
   "click field" + "type text" into one step.

6. **Web tasks → use browser tools.** If the recording shows browser
   interactions, the replay should use `browser_*` tools (headless Chromium)
   rather than `computer_use` — more reliable for web automation.

## Phase 3: Replay

When the user says "run the <name> skill" or "replay <name>":

1. **Load the skill** (it loads automatically if named correctly).

2. **For macOS desktop apps** — follow each step using `computer_use`:
   ```
   computer_use(action="capture", mode="som", app="<app>")
   → Find the target element by matching the description
   → computer_use(action="click", element=<N>)
   → computer_use(action="capture", mode="som")  # verify
   → If verification fails, retry or ask user
   ```

3. **For web tasks** — use `browser_*` tools:
   ```
   browser_navigate(url="...")
   browser_snapshot()
   → Find the target element by ref ID
   → browser_click(ref="@e5")
   → browser_snapshot()  # verify
   ```

4. **Verify after each step.** Take a screenshot/snapshot and compare to the
   expected state described in the skill. If something looks wrong:
   - Re-capture and try finding the element again
   - If the app state is fundamentally different (wrong page, dialog appeared),
     stop and ask the user

5. **Report completion.** When all steps are done, take a final screenshot
   and send it to the user with a summary.

## Pitfalls

### Recording Issues

- **macOS CGEventTap creation fails:** Terminal needs Accessibility permission.
  System Settings → Privacy & Security → Accessibility → add Terminal.
- **Blank screenshots (macOS):** Terminal needs Screen Recording permission.
  System Settings → Privacy & Security → Screen Recording → add Terminal.
- **Linux pynput not capturing:** Ensure X11 session (not Wayland). Check with
  `echo $XDG_SESSION_TYPE`. If Wayland, falls back to screenshot-only mode.
- **Linux screenshots fail:** Install `scrot`, `gnome-screenshot`, or
  `imagemagick`. The script tries all three.
- **Windows missing window info:** Install `pygetwindow` and `uiautomation`
  for full window/element tree capture.
- **cua-driver not found (macOS):** AX trees fall back to osascript (less
  detail). Install cua-driver for best results.
- **Password fields not captured:** This is intentional — macOS Secure Input
  Mode blocks key logging in `AXSecureTextField` fields. This is a security
  feature. During replay, password fields are handled via password manager
  integration (Keychain, 1Password, Bitwarden) or manual entry. See the
  "Password fields" section above.

### Generation Issues

- **Too many steps:** Increase the pause threshold in the analysis script
  (default 2.0s). Or manually merge related steps when writing the SKILL.md.
- **Screenshots don't match replay:** Screenshots are for the vision model to
  understand context, not for pixel-perfect replay matching. Always use
  element indices during replay.
- **Keystrokes captured include modifier presses:** The analysis script filters
  out standalone modifier presses (shift, cmd, etc.) and only keeps the
  combined keystroke (e.g., "cmd+s" not "cmd" then "s").

### Replay Issues

- **Element indices changed:** Always re-capture before clicking. Never assume
  element indices from a previous capture are still valid.
- **App not frontmost (macOS):** Use `computer_use(action="focus_app", app="AppName")`
  or pass `app="AppName"` to capture/click actions.
- **Timing-sensitive UI:** Add `computer_use(action="wait", seconds=1.0)`
  between steps if the UI needs time to load.
- **Dialog appeared unexpectedly:** Re-capture, check if it's a known dialog
  (save prompt, permission request), and handle per the skill instructions.
  If unknown, stop and ask the user.
- **Web page elements shifted:** Use `browser_snapshot()` to get fresh ref IDs
  before each click. Don't cache old refs.

## Verification

To verify a generated skill works:

1. Run the replay on a fresh instance of the task
2. Compare the final screenshot to the recording's final screenshot
3. Check that all expected outcomes are met
4. If the replay fails, patch the skill with the fix and re-test

## Example: YouTube Upload Workflow

User records: opening YouTube Studio in Chrome, clicking upload, selecting a
video file, entering title/description, selecting thumbnail, setting visibility
to "Draft".

Generated skill `youtube-upload` would contain steps like:

1. Browser: navigate to studio.youtube.com
2. Click "Create" → "Upload videos" (browser_click ref="@e12")
3. Click file input, type path to video file, press Enter
4. Wait for upload progress, then fill title field
5. Fill description field
6. Click thumbnail upload, select thumbnail file
7. Click "Save" → "Draft"
8. Verify: "Draft saved" notification appears

Each step includes the app, the action, the element to find, and verification.

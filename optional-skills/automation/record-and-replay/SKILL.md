---
name: record-and-replay
description: |
  Record any macOS workflow by performing it once — the agent watches your
  screen, captures your clicks and keystrokes via CGEventTap, and generates a
  replayable skill. Replay the skill later to automate the task autonomously.
  Cadillac version: real input event capture, not just screenshot diffing.
version: 1.0.0
author: Eric Woodard (Snail3D)
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [automation, record, replay, computer-use, workflow, macro]
    category: automation
    related_skills: [apple, macos-computer-use]
    requires_toolsets: [terminal]
---

# Record & Replay

Record any macOS workflow by performing it once. The agent captures your
real input events (mouse clicks, keystrokes, scrolls, drags) via CGEventTap,
synchronized with periodic screenshots and accessibility tree snapshots. It
then analyzes the recording and generates a replayable skill. Later, invoke
the skill to have the agent repeat the task autonomously via `computer_use`.

This is the "Cadillac" version — it captures actual system input events,
not just screenshot diffs. This means it knows exactly where you clicked,
what keys you pressed, and in what order, producing much more precise
replay skills.

## When to Use

- The user says "record this workflow" or "watch me do this"
- The user wants to automate a repetitive GUI task
- The user wants to create a reusable skill from a manual process
- The user says "replay" or "run the <name> skill"

## Requirements

### For Recording

1. **macOS** (uses Quartz CGEventTap)
2. **pyobjc-framework-Quartz** — `pip install pyobjc-framework-Quartz`
3. **cua-driver** — for AX tree snapshots. Install via `hermes tools` →
   enable Computer Use, or directly:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh)"
   ```
4. **Permissions** — the Terminal (or whatever shell runs the script) needs:
   - System Settings → Privacy & Security → **Accessibility** ✓
   - System Settings → Privacy & Security → **Screen Recording** ✓

### For Replay

1. **computer_use toolset** enabled (`hermes tools` → Computer Use)
2. **cua-driver** installed (same as above)
3. A vision-capable model (for screenshot verification during replay)

## Phase 1: Record

### Starting a Recording

When the user says "record this workflow" (or similar):

1. **Check prerequisites:**
   ```bash
   python3 -c "import Quartz; print('Quartz OK')" 2>/dev/null
   which cua-driver
   ```

2. **Start the recording script** in the background:
   ```bash
   python3 ~/.hermes/skills/automation/record-and-replay/scripts/record_workflow.py \
     --interval 1.0 \
     --output-dir ~/.hermes/recordings/<descriptive-name>
   ```

   Or let the user specify the interval. Faster intervals (0.5s) capture
   more detail but produce more screenshots. Default 1.0s is a good balance.

3. **Tell the user:**
   > 🔴 Recording started. Go ahead and perform your workflow.
   > I'm capturing your clicks, keystrokes, and screenshots.
   > When you're done, tell me to stop.

4. **Wait for the user** to say "stop" / "done" / "ok stop".

5. **Stop the recording** by sending SIGINT to the process:
   ```bash
   # If running as a background process:
   kill -INT <pid>
   ```

### What Gets Captured

| Data | Method | Purpose |
|------|--------|---------|
| Mouse clicks (down/up, button, position) | CGEventTap | Know exactly where user clicked |
| Keystrokes (key code, character, modifiers) | CGEventTap | Know exactly what was typed |
| Scrolls (direction, amount, position) | CGEventTap | Know scroll behavior |
| Mouse drags | CGEventTap | Know drag operations |
| App switches | NSWorkspace | Know when user changed apps |
| Screenshots | `screencapture -x` | Visual context for each step |
| AX tree snapshots | cua-driver / osascript | Element names + structure |

### Recording Output Structure

```
~/.hermes/recordings/<name>/
├── metadata.json              # Duration, event count, etc.
├── events/
│   └── events.jsonl           # Streaming event log (one JSON per line)
├── screenshots/
│   ├── 0001.png               # Numbered screenshots
│   ├── 0002.png
│   └── ...
└── ax_trees/
    ├── 0001.txt               # AX tree at each screenshot
    ├── 0002.txt
    └── ...
```

## Phase 2: Generate Skill

After the recording stops, analyze it and generate a SKILL.md.

### Step 1: Run the analysis script

```bash
python3 ~/.hermes/skills/automation/record-and-replay/scripts/analyze_recording.py \
  ~/.hermes/recordings/<name>
```

This produces a JSON summary with:
- Logical steps (grouped by 2s+ pauses or app switches)
- Each step's interactions (clicks, keys, scrolls)
- Associated screenshots
- Frontmost app for each step

### Step 2: Review screenshots with vision

For each step, load the before/after screenshots and use `vision_analyze` to
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
platforms: [macos]
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
- computer_use toolset enabled
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

## Phase 3: Replay

When the user says "run the <name> skill" or "replay <name>":

1. **Load the skill** (it loads automatically if named correctly).

2. **Follow each step:**
   ```
   computer_use(action="capture", mode="som", app="<app>")
   → Find the target element by matching the description
   → computer_use(action="click", element=<N>)
   → computer_use(action="capture", mode="som")  # verify
   → If verification fails, retry or ask user
   ```

3. **Verify after each step.** Take a screenshot and compare to the expected
   state described in the skill. If something looks wrong:
   - Re-capture and try finding the element again
   - If the app state is fundamentally different (wrong page, dialog appeared),
     stop and ask the user

4. **Report completion.** When all steps are done, take a final screenshot
   and send it to the user with a summary.

## Pitfalls

### Recording Issues

- **CGEventTap creation fails:** Terminal needs Accessibility permission.
  System Settings → Privacy & Security → Accessibility → add Terminal.
- **Blank screenshots:** Terminal needs Screen Recording permission.
  System Settings → Privacy & Security → Screen Recording → add Terminal.
- **cua-driver not found:** AX trees will fall back to osascript (less detail).
  Install cua-driver for best results.
- **pyobjc not installed:** `pip install pyobjc-framework-Quartz`

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
- **App not frontmost:** Use `computer_use(action="focus_app", app="AppName")`
  or pass `app="AppName"` to capture/click actions.
- **Timing-sensitive UI:** Add `computer_use(action="wait", seconds=1.0)`
  between steps if the UI needs time to load.
- **Dialog appeared unexpectedly:** Re-capture, check if it's a known dialog
  (save prompt, permission request), and handle per the skill instructions.
  If unknown, stop and ask the user.

## Verification

To verify a generated skill works:

1. Run the replay on a fresh instance of the task
2. Compare the final screenshot to the recording's final screenshot
3. Check that all expected outcomes are met
4. If the replay fails, patch the skill with the fix and re-test

## Example: YouTube Upload Workflow

User records: opening YouTube Studio, clicking upload, selecting a video file,
entering title/description, selecting thumbnail, setting visibility to "Draft".

Generated skill `youtube-upload` would contain steps like:

1. Focus Safari, navigate to studio.youtube.com
2. Click "Create" → "Upload videos" (element indices from SOM capture)
3. Click file input, type path to video file, press Enter
4. Wait for upload progress, then fill title field
5. Fill description field
6. Click thumbnail upload, select thumbnail file
7. Click "Save" → "Draft"
8. Verify: "Draft saved" notification appears

Each step includes the app, the action, the element to find, and verification.

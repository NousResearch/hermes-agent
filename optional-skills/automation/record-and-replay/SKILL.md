---
name: record-and-replay
description: "Record any workflow by performing it once — the agent watches your screen, captures your clicks and keystrokes globally across all apps, and generates a replayable skill."
version: 4.0.0
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

**Cross-platform:** Works on macOS (CGEventTap), Linux (pynput/X11), and
Windows (pynput/Win32).

## When to Use

- The user says "record this workflow" or "watch me do this"
- The user wants to automate a repetitive GUI task
- The user wants to create a reusable skill from a manual process
- The user says "replay" or "run the <name> skill"
- The user says "learn how to do X" or "figure out the best way to do X" → self-learning mode
- The user wants the agent to practice a task and save the most efficient approach

## Files

| File | Purpose |
|------|---------|
| `scripts/record_workflow.py` | Recording script — captures input events, screenshots, AX trees, clipboard, window state |
| `scripts/analyze_recording.py` | Analysis script — step detection, pattern/retry detection, HTML viewer generation |
| `scripts/generate_skill.py` | Vision-based skill generation from analysis JSON |
| `scripts/replay_skill.py` | Replay engine — executes generated skills step by step |
| `scripts/self_learn.py` | Self-learning engine — agent records its own attempts, compares efficiency, saves the best |
| `scripts/view_recording.html.template` | HTML viewer template (reference for the generated viewer) |
| `scripts/test_record_replay.sh` | Integration test — full pipeline validation |

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

### Full Pipeline: Record → Analyze → Generate → Replay

```bash
# 1. Record
python3 scripts/record_workflow.py --interval 1.0 --output-dir ~/.hermes/recordings/my-task

# 2. Analyze (with optional HTML viewer)
python3 scripts/analyze_recording.py ~/.hermes/recordings/my-task --html --output analysis.json

# 3. Generate skill (vision-based if configured, text-only fallback)
python3 scripts/generate_skill.py --analysis analysis.json --recording-dir ~/.hermes/recordings/my-task --name my-task

# 4. Replay (dry-run first, then real)
python3 scripts/replay_skill.py --skill ~/.hermes/skills/automation/my-task/SKILL.md --dry-run
python3 scripts/replay_skill.py --skill ~/.hermes/skills/automation/my-task/SKILL.md --step-delay 1.0
```

Later, to replay:
> "run the youtube-upload skill"
> "replay my expense-report workflow"

## Requirements by Platform

### macOS (CGEventTap — Cadillac mode)

```bash
pip install pyobjc-framework-Quartz mss Pillow
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
pip install pynput mss Pillow
# Screenshot backends (any one):
sudo apt install scrot       # or gnome-screenshot, or imagemagick
# Window info (optional, for AX tree fallback):
sudo apt install xdotool
# AT-SPI2 (optional, for accessibility trees):
pip install pyatspi
```

### Windows (pynput + Win32)

```bash
pip install pynput mss Pillow pygetwindow uiautomation
```

## Phase 1: Record

### Starting a Recording

```bash
python3 <skill_dir>/scripts/record_workflow.py \
  --interval 1.0 \
  --output-dir ~/.hermes/recordings/<descriptive-name>
```

Stop the recording by creating the stop flag file:
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
| **Clipboard content** | pbpaste (macOS) / xclip (Linux) / tkinter (Windows) | Capture copy/paste workflows |
| **Window state changes** | NSWorkspace + AX API (macOS) | Track resize/move/minimize for replay accuracy |

### Screenshot Diffing

The recorder only saves a screenshot when pixels actually change from the
previous frame (using `PIL.ImageChops.difference` with a 5% threshold). This
reduces disk usage by 80%+ for typical workflows where the user pauses
between actions. When a screenshot is skipped, a `screenshot_skipped` event
is logged so the analysis script knows the screen was checked but unchanged.

### Clipboard Monitoring

The recorder polls the clipboard every 0.5 seconds. When clipboard content
changes, a `clipboard_copy` event is logged with:
- First 200 characters (for context — NOT full content for security)
- Content hash (for deduplication)
- Content length

This captures copy/paste workflows that pure key logging misses.

### Window State Tracking (macOS)

On macOS, the recorder monitors window position, size, and minimize state.
When changes are detected, it logs:
- `window_resized` — window was resized
- `window_moved` — window was moved
- `window_minimized` — window was minimized

These events are important for replay accuracy — if a window was resized
during the original recording, the replay needs to know the expected
dimensions.

### Password fields — handled, not captured

macOS **Secure Input Mode** deliberately blocks keyloggers from seeing
keystrokes in password fields (`AXSecureTextField`). This is a security
feature — the recording script cannot and should not bypass it.

### Recording Output Structure

```
~/.hermes/recordings/<name>/
├── metadata.json              # Duration, event count, platform, screenshots_skipped
├── events/
│   └── events.jsonl           # Streaming event log (one JSON per line)
├── screenshots/
│   ├── 0001.png               # Only saved when pixels change (diffing)
│   └── ...
├── ax_trees/
│   ├── 0001.txt               # AX/AT/UI tree at each screenshot
│   └── ...
└── viewer.html                # Generated by analyze_recording.py --html
```

## Phase 2: Analyze

After the recording stops, analyze it:

```bash
python3 scripts/analyze_recording.py ~/.hermes/recordings/<name>
# With HTML viewer:
python3 scripts/analyze_recording.py ~/.hermes/recordings/<name> --html
# Save to file:
python3 scripts/analyze_recording.py ~/.hermes/recordings/<name> --output analysis.json
```

### Analysis Output

The analysis produces a JSON summary with:
- **Logical steps** — grouped by natural boundaries (pause > 1.5s, app switch,
  screenshot with pixel changes, clipboard copy)
- **Step signatures** — action-type sequences (e.g., `["click", "type", "click"]`)
- **Pattern detection** — repeated step signatures indicate loops
- **Retry detection** — nearby clicks within 500ms = likely mis-click
- **Clipboard events** — copy/paste operations
- **Window state changes** — resize/move/minimize
- **Suggested skill name** — auto-generated from app + first action

### Recording Viewer

The `--html` flag generates a self-contained HTML file (`viewer.html`) with:
- Timeline of screenshots with timestamps
- Event markers on the timeline (clicks=red, keys=blue, scrolls=green)
- Step boundaries as vertical dividers
- Play button that auto-advances through screenshots at recording speed
- Click any screenshot to see it full-size with the AX tree alongside
- Detected patterns listed in the info panel

The viewer is fully self-contained — inline CSS, JS, and base64 images. No
external dependencies. Open it in any browser:

```bash
open ~/.hermes/recordings/<name>/viewer.html
```

## Phase 3: Generate Skill

Generate a SKILL.md from the analysis:

```bash
python3 scripts/generate_skill.py \
  --analysis analysis.json \
  --recording-dir ~/.hermes/recordings/<name> \
  --name my-task

# Preview without saving:
python3 scripts/generate_skill.py --analysis analysis.json --dry-run --no-vision

# Save to custom location:
python3 scripts/generate_skill.py --analysis analysis.json --output-dir /tmp/my-skill
```

If a vision provider is configured (Hermes auxiliary vision), the script uses
it to analyze screenshots and understand what UI elements are visible and what
changed between before/after states. If no vision provider is available, it
falls back to text-only analysis from the AX tree + event log.

The generated SKILL.md includes:
- Element descriptions (not pixel coordinates) for replay
- Verification criteria for each step
- Error recovery instructions
- Estimated replay duration
- Detected patterns noted as loop candidates

## Phase 4: Replay

Replay a generated skill:

```bash
# Dry-run (captures and finds elements but doesn't click):
python3 scripts/replay_skill.py --skill ~/.hermes/skills/automation/my-task/SKILL.md --dry-run

# Real replay:
python3 scripts/replay_skill.py --skill ~/.hermes/skills/automation/my-task/SKILL.md --step-delay 1.0 --max-retries 2
```

### Replay Engine Behavior

For each step, the replay engine:
1. Captures current screen state
2. Finds the target element by matching description against current AX tree
3. Performs the action (click, type, scroll, drag)
4. Captures post-action state
5. Verifies the expected outcome
6. If verification fails: retries once, tries alternatives, then pauses and logs

### Replay Report

The engine generates a JSON report with:
- Steps attempted, succeeded, failed
- Screenshots at each step
- Total duration
- Exit code: 0 if all succeeded, 1 if any failed

## Advanced Usage

### Self-Learning Mode — Agent Records Itself

Instead of a human recording a workflow, the **agent itself** performs the task
via `computer_use` while the recording script runs in the background. The agent
attempts the task multiple times, each time trying to be more efficient (fewer
clicks, fewer retries, faster navigation). After all attempts, the
self-learning engine compares them and saves the best one as a skill.

This is useful when:
- The user wants the agent to learn a complex app (e.g., Bambu Slicer, DaVinci
  Resolve) by trial and error
- The user wants the most efficient possible replay skill
- The user says "learn how to do X" or "figure out the best way to do X"

#### Self-Learning Workflow

```
Attempt 1: Agent explores the UI (slow, retries, wrong clicks)
Attempt 2: Agent applies lessons (faster, fewer mistakes)
Attempt 3: Agent optimizes (shortcuts, combined steps, minimal captures)
    ↓
Compare: self_learn.py ranks all attempts by efficiency score
Generate: Best attempt → SKILL.md
Verify: Replay the skill to confirm it works
```

#### Step-by-step: How the Agent Self-Learns

The agent drives this loop using its tools. Here's what it does:

**1. Initialize the session:**
```bash
python3 scripts/self_learn.py --init bambu-slice \
  --description "Open Bambu Slicer, import 3MF, choose filament, slice, save G-code"
```

**2. For each attempt (1 through N, typically 3):**

a) Start recording in the background:
```bash
python3 scripts/record_workflow.py --interval 1.0 \
  --output-dir ~/.hermes/recordings/bambu-slice-attempt-N
```

b) Perform the task using `computer_use`:
```
computer_use(action="capture", mode="som", app="Bambu Studio")
computer_use(action="click", element=5)  # File menu
computer_use(action="click", element=12) # Import
computer_use(action="type", text="/path/to/model.3mf")
computer_use(action="key", keys="return")
# ... continue until task complete
```

c) Stop the recording:
```bash
touch ~/.hermes/recordings/bambu-slice-attempt-N/.stop
```

d) Analyze this attempt:
```bash
python3 scripts/self_learn.py --analyze ~/.hermes/recordings/self-learn/bambu-slice \
  --attempt N --recording ~/.hermes/recordings/bambu-slice-attempt-N
```

e) Review the metrics. Before the next attempt, think about:
   - Which clicks were retries? Can you find the right element faster?
   - Are there keyboard shortcuts that skip menu navigation?
   - Can you combine "click field" + "type text" into fewer captures?
   - Can you skip unnecessary verification captures?

**3. After all attempts, compare:**
```bash
python3 scripts/self_learn.py --compare ~/.hermes/recordings/self-learn/bambu-slice
```

**4. View the report:**
```bash
python3 scripts/self_learn.py --report ~/.hermes/recordings/self-learn/bambu-slice
```

The report shows a table of all attempts with metrics (duration, actions,
retries, steps, score) and explains why the best attempt won.

**5. Generate skill from the best attempt:**
```bash
python3 scripts/self_learn.py --generate ~/.hermes/recordings/self-learn/bambu-slice
```

**6. Install and verify:**
```bash
cp -r ~/.hermes/recordings/self-learn/bambu-slice/best-skill/ \
  ~/.hermes/skills/automation/bambu-slice/
python3 scripts/replay_skill.py \
  --skill ~/.hermes/skills/automation/bambu-slice/SKILL.md --dry-run
```

#### Efficiency Metrics

The self-learning engine ranks attempts using a weighted score (lower = better):

| Metric | Weight | What it measures |
|--------|--------|------------------|
| Action count | 30% | Total interactions (clicks + keys + scrolls) |
| Retry count | 30% | Detected mis-clicks (nearby clicks within 500ms) |
| Duration | 20% | Wall time from first to last event |
| Step count | 20% | Logical steps detected by the analyzer |

Raw values are normalized to a 0-10 scale using soft caps (120s duration,
50 actions, 10 retries, 20 steps = score 10 each). The final score is the
weighted sum.

#### How Many Attempts?

- **2-3 attempts** for simple tasks (open app, do one thing, close)
- **3-5 attempts** for complex tasks (multi-step workflows with menus, dialogs)
- Stop early if the last 2 attempts have the same score (no more improvement)

#### What Gets Captured During Self-Learning

The recording script captures screenshots, AX trees, and any OS-level events.
When the agent uses `computer_use` (cua-driver), the posted events flow through
the OS event system and are captured by CGEventTap. Even if some programmatic
events aren't captured, the screenshots + AX trees provide full context for the
analysis — the analyzer reconstructs what happened from visual state changes.

### Flags for analyze_recording.py

| Flag | Description |
|------|-------------|
| `--html` | Generate self-contained HTML timeline viewer |
| `--output FILE` | Write JSON to file instead of stdout |
| `--pause-threshold SECS` | Pause threshold for step boundaries (default: 1.5) |

### Flags for generate_skill.py

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview the generated skill without saving |
| `--no-vision` | Skip vision analysis, use text-only mode |
| `--output-dir DIR` | Output directory (default: `~/.hermes/skills/automation/<name>/`) |
| `--name NAME` | Skill name (default: auto-generated from analysis) |

### Flags for replay_skill.py

| Flag | Description |
|------|-------------|
| `--dry-run` | Capture and find elements but don't click (for testing) |
| `--step-delay SECS` | Seconds between steps (default: 1.0) |
| `--max-retries N` | Max retries per step (default: 2) |
| `--output-dir DIR` | Output directory for replay report |

### Flags for self_learn.py

| Flag | Description |
|------|-------------|
| `--init TASK_NAME` | Initialize a new self-learning session |
| `--description TEXT` | Task description (used with `--init`) |
| `--analyze SESSION_DIR` | Analyze an attempt (requires `--attempt` and `--recording`) |
| `--attempt N` | Attempt number (used with `--analyze`) |
| `--recording DIR` | Path to the recording directory (used with `--analyze`) |
| `--compare SESSION_DIR` | Compare all attempts and pick the best |
| `--generate SESSION_DIR` | Generate skill from the best attempt |
| `--report SESSION_DIR` | Print the comparison report |
| `--dry-run` | Preview skill generation without saving (used with `--generate`) |

### Testing the Full Pipeline

Run the integration test:

```bash
bash scripts/test_record_replay.sh
```

This creates a synthetic recording, runs the full analyze → generate → replay
pipeline, verifies all outputs, and cleans up test artifacts.

## Pitfalls

### Recording Issues

- **macOS CGEventTap creation fails:** Terminal needs Accessibility permission.
- **Blank screenshots (macOS):** Terminal needs Screen Recording permission.
- **Linux pynput not capturing:** Ensure X11 session (not Wayland).
- **cua-driver not found (macOS):** AX trees fall back to osascript.
- **Password fields not captured:** Intentional — macOS Secure Input Mode blocks this.

### Analysis Issues

- **Too many steps:** Increase the pause threshold (`--pause-threshold 2.0`).
- **Screenshots don't match replay:** Screenshots are for context, not
  pixel-perfect matching. Always use element indices during replay.
- **False retry detection:** Adjust `RETRY_DISTANCE_PX` and `RETRY_TIME_MS`
  in the script if needed.

### Replay Issues

- **Element indices changed:** Always re-capture before clicking.
- **App not frontmost:** Use `computer_use(action="focus_app", app="AppName")`.
- **Timing-sensitive UI:** Add `--step-delay 2.0` between steps.
- **Dialog appeared unexpectedly:** The engine retries, then pauses and logs.

## Verification

To verify a generated skill works:

1. Run the replay in `--dry-run` mode first
2. Run the full replay on a fresh instance of the task
3. Compare the final screenshot to the recording's final screenshot
4. Check the replay report for any failed steps
5. If the replay fails, patch the skill and re-test

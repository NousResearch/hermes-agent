---
name: wallpaper-engine
description: Generate and cycle desktop wallpapers via ComfyUI.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [wallpaper, desktop, comfyui, image-generation, creative, personalization]
    related_skills: [comfyui]
    category: creative
---

# Wallpaper Engine

Generate beautiful, personalized desktop wallpapers using ComfyUI and learn the
user's aesthetic preferences over time via Hermes memory.  Each wallpaper is
generated at native desktop resolution, set automatically, and tracked for
feedback — so the agent's taste improves with every generation.

This skill delegates all image generation to the **comfyui skill** — it doesn't
duplicate any of that pipeline.  What it adds is wallpaper-specific resolution
tuning, cross-platform desktop wallpaper setting, a local generation/feedback
history log, and the workflow for feeding preferences back into Hermes memory.

## When to Use

Load when the user says things like:

- "generate a new wallpaper"
- "change my desktop background"
- "give me a dark sci-fi wallpaper"
- "I want a new wallpaper every morning"
- "schedule daily wallpapers"
- "I like this wallpaper" / "I don't like that one"
- "what wallpapers have you generated for me?"
- "summarize my wallpaper preferences"

## Prerequisites

1. **ComfyUI running** — local (`comfy launch --background`) or
   Comfy Cloud (paid subscription, `COMFY_CLOUD_API_KEY` set).  See the
   comfyui skill's `## Setup & Onboarding` for full setup instructions.

2. **SDXL model** — at minimum `sd_xl_base_1.0.safetensors` (or any SDXL
   checkpoint) installed in ComfyUI.  The `check_deps.py` script from the
   comfyui skill will verify this.

3. **The wallpaper-engine skill installed** — `hermes skills install
   official/creative/wallpaper-engine`.  After install the scripts under
   `optional-skills/creative/wallpaper-engine/scripts/` are on the skill path.

## How to Run

Every workflow below starts from the **skill directory** (`SKILL_DIR`).  The
comfyui skill scripts are at `SKILL_DIR/../../skills/creative/comfyui/scripts/`
relative to this skill, or `skills/creative/comfyui/scripts/` from the repo
root.

### Step 0: Verify ComfyUI is ready

```bash
python3 skills/creative/comfyui/scripts/health_check.py
```

If the check fails, follow the comfyui skill's setup path (hardware check →
install → launch) before continuing.

### Step 1: Check the workflow's dependencies

```bash
python3 skills/creative/comfyui/scripts/check_deps.py \
  optional-skills/creative/wallpaper-engine/workflows/wallpaper_txt2img.json
```

If models or nodes are missing, use `auto_fix_deps.py` from the comfyui skill.

### Step 2: See what parameters are available

```bash
python3 skills/creative/comfyui/scripts/extract_schema.py \
  optional-skills/creative/wallpaper-engine/workflows/wallpaper_txt2img.json \
  --summary-only
```

The key parameters: `positive_prompt` (node 6), `negative_prompt` (node 7),
`width` (node 5, default 1920), `height` (node 5, default 1080), `seed` (node
3, use -1 for random), `steps` (node 3, default 30), `cfg` (node 3, default
7.5), `ckpt_name` (node 4).

### Step 3: Generate the wallpaper

```bash
python3 skills/creative/comfyui/scripts/run_workflow.py \
  --workflow optional-skills/creative/wallpaper-engine/workflows/wallpaper_txt2img.json \
  --args '{"positive_prompt": "your prompt here", "negative_prompt": "ugly, blurry, ...", "seed": -1}' \
  --output-dir ~/.hermes/wallpaper-engine/images/
```

### Step 4: Set the wallpaper

```bash
python3 optional-skills/creative/wallpaper-engine/scripts/set_wallpaper.py \
  ~/.hermes/wallpaper-engine/images/wallpaper_00001_.png
```

### Step 5: Record the generation

```bash
python3 optional-skills/creative/wallpaper-engine/scripts/wallpaper_history.py \
  add ~/.hermes/wallpaper-engine/images/wallpaper_00001_.png \
  "the prompt used" "wallpaper_txt2img.json"
```

### Step 6: Collect feedback (later, when the user reacts)

```bash
python3 optional-skills/creative/wallpaper-engine/scripts/wallpaper_history.py \
  feedback <id-from-step-5> 5 "dark" "moody" "mountains"
```

## Quick Reference

| Action | Tool / Command |
|--------|---------------|
| Generate wallpaper | `run_workflow.py --workflow wallpaper_txt2img.json --args '{...}'` |
| Check what params exist | `extract_schema.py wallpaper_txt2img.json` |
| Set image as wallpaper | `set_wallpaper.py <path>` |
| Record generation | `wallpaper_history.py add <path> <prompt> <workflow>` |
| Record feedback | `wallpaper_history.py feedback <id> <rating> [tags...]` |
| List recent generations | `wallpaper_history.py list [--limit N] [--rated-only]` |
| Get preference stats | `wallpaper_history.py stats` |
| Check user preferences | `memory(action=read)` — look in USER.md for aesthetic notes |
| Store learned preference | `memory(action=add, content="...")` |

## Procedure

### One-Shot Generation

When the user asks for a new wallpaper:

1. **Read existing preferences** from memory.  `memory(action=read)` returns
   MEMORY.md and USER.md.  Look for aesthetic notes about wallpaper styles the
   user has liked or disliked.  If none exist, ask the user what kind of
   wallpaper they want — mood, colors, subject matter.

2. **Pick a prompt template** from `references/prompt-library.md` that matches
   the user's stated or learned preferences.  Layer any additional guidance
   from memory on top.  If the user's request doesn't match any template,
   improvise a prompt that follows the same structure (positive prompt +
   negative prompt, SDXL-optimized).

3. **Detect the native resolution.** On macOS run
   `osascript -e 'tell application "Finder" to get bounds of window of desktop'`
   and parse the width/height.  On Linux try `xdpyinfo | grep dimensions` or
   `swaymsg -t get_outputs`.  On Windows check the primary monitor via
   PowerShell: `Add-Type -AssemblyName System.Windows.Forms;
   [System.Windows.Forms.Screen]::PrimaryScreen.Bounds`.  Fall back to a
   sensible default (1920x1080) if detection fails.

4. **Run the workflow** via comfyui's `run_workflow.py`, injecting the
   resolution and crafted prompt.  If the user has a capable GPU, suggest 4K
   (3840x2160) for sharper results — but note it takes longer.

5. **Set the wallpaper** via `set_wallpaper.py`.

6. **Record it** via `wallpaper_history.py add`.

7. **Report to the user** what was generated, and ask if they like it.

### Gathering Feedback

When the user expresses like or dislike about a wallpaper:

1. **Find the most recent entry** via `wallpaper_history.py list --limit 1`
   (or search by ID if they referenced a specific one).

2. **Map sentiment to rating**: "love it" / "amazing" → 5, "nice" / "good" →
   4, "it's okay" / "fine" → 3, "not great" / "meh" → 2, "hate it" /
   "terrible" → 1.

3. **Extract tags** from their feedback: adjectives like "dark", "moody",
   "bright", "minimalist"; subjects like "mountains", "space", "abstract";
   qualities like "too busy", "too simple", "perfect composition".

4. **Record the feedback** via `wallpaper_history.py feedback`.

5. **If the rating is strong (1-2 or 4-5), update memory** to persist the
   preference.  Use `memory(action=add)` with a summary like:
   > "User strongly prefers dark, moody wallpapers with natural landscapes
   > (rated mountain wallpaper 5/5 on 2026-07-24). Dislikes busy abstract
   > compositions (rated 2/5)."

6. **If the user disliked it**, offer to generate a replacement immediately.

### Checking Preferences

When the user asks about their taste profile:

1. **Run stats**: `wallpaper_history.py stats`
2. **Read memory**: `memory(action=read)` and extract the aesthetic notes
3. **Summarize**: present the top loved tags, top disliked tags, average
   rating, and a prose summary of their taste from memory.

### Scheduling Recurring Wallpapers

When the user wants regular wallpaper changes:

1. **Read preferences** from memory and history stats.

2. **Construct a self-contained cron prompt** that includes the preferences
   and instructs the agent to follow the One-Shot Generation procedure above.
   Example:

   > "Generate a new desktop wallpaper using the wallpaper-engine skill.
   > Use the comfyui skill's run_workflow.py with the wallpaper_txt2img.json
   > workflow.  The user's aesthetic preferences from memory are: [insert
   > preferences].  After generating, set the wallpaper via
   > wallpaper-engine/scripts/set_wallpaper.py and record it via
   > wallpaper_history.py add."

3. **Create the cron job** via terminal:
   ```bash
   hermes cron add \
     --schedule "0 7 * * *" \
     --name "daily-wallpaper" \
     --skills "comfyui,wallpaper-engine" \
     --prompt "<the constructed prompt>"
   ```

4. **Confirm** the schedule with the user and tell them how to manage it:
   `hermes cron list`, `hermes cron pause daily-wallpaper`,
   `hermes cron remove daily-wallpaper`.

## Preference Learning Loop

Over time, as the user rates wallpapers, the agent builds a profile:

```
Generate → Set → User sees it → Feedback → History → Memory
    ↑                                                      │
    └──────────────────────────────────────────────────────┘
            (next generation uses updated preferences)
```

The agent should periodically (e.g. every 5 ratings) read
`wallpaper_history.py stats` and consolidate new insights into memory via
`memory(action=add)`.  This closes the loop: each generation benefits from all
past feedback.

## Pitfalls

1. **ComfyUI must be running.**  The `run_workflow.py` script needs a live
   ComfyUI server at `http://127.0.0.1:8188` (or the cloud host).  If the
   server is down, the agent cannot generate.  Start it with
   `comfy launch --background` or point the user to the comfyui skill's setup.

2. **Workflow JSON must be clean API format.**  The workflow file must contain
   only node objects keyed by node ID — no top-level `_comment`, strings, or
   other non-node keys.  ComfyUI's `validate_prompt()` rejects these.  The
   bundled `wallpaper_txt2img.json` is already clean; if you create custom
   workflows, verify with:
   ```bash
   python3 -c "import json; wf=json.load(open('your_workflow.json'));
   nodes={k:v for k,v in wf.items() if isinstance(v,dict) and 'class_type' in v};
   print(f'{len(nodes)} nodes, {len(wf)-len(nodes)} non-node keys stripped')"
   ```

3. **SDXL resolution limits.**  SDXL's native resolution is 1024x1024.  Going
   much above 1920x1200 with the base workflow may produce repetition artifacts
   (the "tiling effect").  For true 4K without artifacts, the user needs Flux
   Dev or an upscale stage (add an UpscaleModelLoader + ImageUpscaleWithModel
   node to the workflow).

4. **Wallpaper setting is desktop-environment specific.**  The
   `set_wallpaper.py` script tries GNOME, KDE, XFCE, Sway, feh, and nitrogen
   on Linux; osascript on macOS; and the Win32 API on Windows.  If none work
   (e.g. a niche window manager), the generated image is still saved — tell the
   user where to find it and how to set it manually.

5. **First-run setup is heavy.**  The first time through, the agent may need to
   install ComfyUI, comfy-cli, and download the SDXL model (~6.5 GB).  This is
   a one-time cost — subsequent runs are fast.  Warn the user that the first
   generation will take extra time for setup.

6. **Generation takes time.**  An SDXL generation at 1920x1080 takes 30-90
   seconds on a modern GPU.  Warn the user that it's in progress.  For the
   first generation, the agent should not go silent — send a progress message.

7. **Cloud costs.**  If using Comfy Cloud, every generation costs credits.
   Remind the user of this before setting up a recurring schedule.

8. **Preferences need multiple data points.**  A single rating doesn't define
   a user's taste.  The agent should avoid over-generalizing from one or two
   ratings — wait for at least 3-5 before making strong claims about the user's
   aesthetic profile.

9. **History file location.**  `wallpaper_history.py` stores data at
   `$HERMES_HOME/wallpaper-engine/history.json`.  If the user switches
   profiles, each profile has its own history.

## Verification

- [ ] ComfyUI health check passes (`health_check.py`)
- [ ] Workflow dependencies are satisfied (`check_deps.py wallpaper_txt2img.json`)
- [ ] `set_wallpaper.py /path/to/test.png` returns `{"status": "ok", ...}`
- [ ] `wallpaper_history.py add /tmp/x.png "test" "w.json"` creates a record
- [ ] `wallpaper_history.py list` shows the record
- [ ] `wallpaper_history.py feedback <id> 4 "test-tag"` works
- [ ] `wallpaper_history.py stats` includes the rated entry
- [ ] End-to-end: agent generates, sets, records, and reports on a wallpaper

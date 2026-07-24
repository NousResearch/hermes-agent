---
name: youtube-automation-agent
description: Run a Hermes-native YouTube automation workflow that mirrors the darkzOGx/youtube-automation-agent stages: strategy, script, thumbnail, SEO, production, publishing, and analytics. Also supports inspecting and probing the upstream repo locally.
version: 1.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [youtube, automation, media, nodejs, express, seo, thumbnails, publishing, analytics]
    related_skills: [youtube-content, google-workspace]
    category: media
    homepage: https://github.com/darkzOGx/youtube-automation-agent
---

# YouTube Automation Agent

This optional skill now does two related jobs:

1. It helps Hermes inspect, configure, and troubleshoot the external repo `darkzOGx/youtube-automation-agent`.
2. It gives Hermes a native interactive workflow that mirrors the repo's claimed pipeline:
   - strategy
   - script
   - thumbnail
   - SEO
   - production
   - publishing
   - analytics

The goal is to capture the essence of the upstream project as a reusable Hermes workflow without pretending Hermes has magically replaced YouTube OAuth, video rendering, or upload APIs.

## What this skill is good for

Use this skill when the user wants Hermes to:
- turn a channel concept into a structured YouTube production workflow
- manage a stage-by-stage content pipeline inside Hermes
- generate a strategy brief, script brief, thumbnail brief, SEO package, publishing checklist, and analytics review plan
- inspect whether the upstream repo is actually runnable
- validate the local server and its main endpoints

## The Hermes-native workflow

This skill mirrors the upstream repo's main agent flow:

1. Content Strategy Agent
2. Script Writer Agent
3. Thumbnail Designer Agent
4. SEO Optimizer Agent
5. Production Management Agent
6. Publishing & Scheduling Agent
7. Analytics & Optimization Agent

In Hermes, those stages are represented as an interactive run workspace that can persist across sessions.

## Locate the helper script

```bash
SCRIPT="$(find ~/.hermes/skills -path '*/youtube-automation-agent/scripts/youtube_automation_helper.py' -print -quit)"
```

## Start a new Hermes workflow run

```bash
python3 "$SCRIPT" init-run \
  --channel "Ladera Labs" \
  --niche "AI productivity" \
  --audience "founders and operators" \
  --style "educational" \
  --frequency daily \
  --topic "AI workflow automations"
```

This creates a run workspace JSON file under the skill data directory and sets the current stage to `strategy`.

## Check workflow status

```bash
python3 "$SCRIPT" status --workspace /path/to/run.json
```

## Get the next stage brief

```bash
python3 "$SCRIPT" brief --workspace /path/to/run.json
```

Or a specific stage:

```bash
python3 "$SCRIPT" brief --workspace /path/to/run.json --stage seo
```

Each brief includes:
- the goal for that stage
- the contextual channel inputs
- a Hermes-ready prompt
- expected artifacts to produce

## Complete a stage

After Hermes or the user finishes a stage, save the outcome:

```bash
python3 "$SCRIPT" complete-stage \
  --workspace /path/to/run.json \
  --stage strategy \
  --notes "Selected a founder-focused AI automation angle" \
  --artifacts-json '{
    "selected_topic": "AI workflow automations for founders",
    "angle": "replace repetitive ops work with reusable agents",
    "content_type": "Explainer",
    "keywords": ["ai automation", "workflow automation", "founder productivity"]
  }'
```

The helper automatically advances the current stage to the next incomplete stage.

## Export a finished run

```bash
python3 "$SCRIPT" export --workspace /path/to/run.json
```

This gives Hermes a portable summary of all completed deliverables.

## How to use this as an interactive Hermes skill

Recommended operator loop:

1. create a run with `init-run`
2. ask for the current stage brief with `brief`
3. use Hermes to generate the requested deliverables for that stage
4. save the result with `complete-stage`
5. repeat until analytics is complete

This turns the repo's abstract pipeline into a real session-by-session Hermes workflow.

## Upstream repo inspection mode

The skill still supports repo validation.

### Inspect a local clone

```bash
python3 "$SCRIPT" inspect --repo /path/to/youtube-automation-agent
```

Checks include:
- required files
- missing package script targets
- config file presence
- `node_modules/` presence
- `node --check` syntax checks for key entry files

### Probe a running local server

```bash
python3 "$SCRIPT" probe --base-url http://localhost:3456
```

This probes:
- `/health`
- `/schedule`
- `/analytics`

## Known upstream caveats

Before claiming the upstream repo is turnkey, remember:

1. `package.json` references missing script targets in the inspected repo:
   - `workflows/daily-content-pipeline.js`
   - `workflows/weekly-strategy-review.js`
   - `database/init.js`
2. The README mentions a `workflows/` directory, but it is absent in the inspected repo.
3. The docs market Gemini as an option, but `utils/credential-manager.js` currently validates `youtube` and `openai`, so Gemini-only setup is not turnkey without upstream changes.
4. A dashboard alone is not enough; confirm `/health`.

See also:
- `references/repo-caveats.md`
- `references/setup-notes.md`
- `references/manual-ops.md`
- `references/hermes-native-flow.md`

## Practical boundaries

This skill can help Hermes achieve the same claimed flow structure as the upstream project:
- strategy
- scripting
- thumbnail planning
- SEO packaging
- production planning
- publishing prep
- analytics feedback loop

But it does not claim that Hermes alone can, without credentials or infrastructure:
- upload to YouTube automatically
- render final videos out of thin air
- authenticate external APIs without setup
- replace all manual production steps

Instead, it gives a strong, reusable workflow layer that Hermes can run interactively.

## Verification checklist

Before telling the user a workflow is ready:

1. the run workspace exists
2. the current stage brief is clear
3. each completed stage has notes and artifacts saved
4. the export summary contains the expected deliverables
5. if using the upstream repo, `inspect` and `probe` have been run successfully

---
name: youtube-automation-agent
description: Orchestrate staged YouTube production workflows.
version: 1.2.0
author: Haithum Abdelfattah (@darkzOGx)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [youtube, automation, media, seo, publishing, analytics]
    related_skills: [youtube-content, google-workspace]
    category: media
    homepage: https://github.com/darkzOGx/youtube-automation-agent
---

# YouTube Automation Agent Skill

Run a persistent, interactive workflow from channel strategy through
post-publish analytics, or inspect the related Node.js project. This skill
organizes deliverables; it does not replace YouTube credentials, rendering,
or upload infrastructure.

## When to Use

Use this skill when the user wants to:

- develop a channel idea through a stage-by-stage content pipeline;
- produce strategy, script, thumbnail, SEO, production, publishing, and
  analytics deliverables across multiple sessions;
- inspect a local clone of `darkzOGx/youtube-automation-agent`; or
- probe that project's local health, schedule, and analytics endpoints.

Do not use it for a one-off transcript or summary when the `youtube-content`
skill is sufficient.

## Prerequisites

- Python 3.9 or newer for the bundled helper.
- An installed copy of this optional skill.
- For the external Node.js app only: Node.js, npm dependencies, YouTube OAuth,
  and the AI-provider credentials expected by that project.

Resolve paths from the loaded skill rather than assuming the default profile.
Use the `skill_dir` returned by `skill_view` as `SKILL_DIR`. When only the
active Hermes home is available, use this fallback:

```bash
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
SKILL_DIR="${SKILL_DIR:-$HERMES_HOME/skills/media/youtube-automation-agent}"
SCRIPT="$SKILL_DIR/scripts/youtube_automation_helper.py"
```

Named profiles set `HERMES_HOME` to their profile directory, so this resolves
the category-preserving official install without searching another profile.

## How to Run

Initialize a workflow workspace:

```bash
python3 "$SKILL_DIR/scripts/youtube_automation_helper.py" init-run \
  --channel "Ladera Labs" \
  --niche "AI productivity" \
  --audience "founders and operators" \
  --style "educational" \
  --frequency daily \
  --topic "AI workflow automations"
```

The helper writes a JSON workspace beneath the active profile's installed
skill directory and starts at `strategy`. Pass `--output /path/to/run.json`
to choose another location, or `--json` for machine-readable output.

## Quick Reference

| Goal | Command |
|---|---|
| Create a run | `python3 "$SCRIPT" init-run <channel options>` |
| Show progress | `python3 "$SCRIPT" status --workspace RUN.json` |
| Get current brief | `python3 "$SCRIPT" brief --workspace RUN.json` |
| Get another brief | `python3 "$SCRIPT" brief --workspace RUN.json --stage seo` |
| Save a stage | `python3 "$SCRIPT" complete-stage --workspace RUN.json --stage STAGE --notes TEXT` |
| Export deliverables | `python3 "$SCRIPT" export --workspace RUN.json` |
| Inspect Node repo | `python3 "$SCRIPT" inspect --repo /path/to/repo` |
| Probe local app | `python3 "$SCRIPT" probe --base-url http://localhost:3456` |

Workflow stages, in order:

1. `strategy`
2. `script`
3. `thumbnail`
4. `seo`
5. `production`
6. `publishing`
7. `analytics`

Use `read_file` on these references only when their branch applies:

- `references/hermes-native-flow.md` — stage artifact contracts;
- `references/repo-caveats.md` — grounded upstream limitations;
- `references/setup-notes.md` — external Node.js setup; and
- `references/manual-ops.md` — command examples and local endpoints.

## Procedure

### 1. Initialize the run

Collect the channel name, niche, audience, style, frequency, and optional
topic. Run `init-run`, record the returned workspace path, and verify its
current stage is `strategy`.

### 2. Produce the current stage

Run `brief` against the workspace. Use its goal, context, prompt, and expected
artifacts as the contract for the stage; do not silently omit required
artifacts.

### 3. Persist the result

After the user accepts the deliverable, save it with `complete-stage`:

```bash
python3 "$SCRIPT" complete-stage \
  --workspace /path/to/run.json \
  --stage strategy \
  --notes "Selected a founder-focused automation angle" \
  --artifacts-json '{
    "selected_topic": "AI workflow automations for founders",
    "angle": "replace repetitive operations with reusable agents",
    "content_type": "Explainer",
    "keywords": ["ai automation", "workflow automation"]
  }'
```

Run `status` and confirm the completed stage is recorded and the next
incomplete stage is `in_progress`.

### 4. Repeat and export

Repeat the brief, production, acceptance, and completion loop through
`analytics`. Run `export` and verify every completed stage appears under
`deliverables`.

### 5. Inspect the external project when requested

Run `inspect --repo` before claiming a local clone is runnable. A `blocked`
report can include missing required files, unreadable or malformed
`package.json`, or package scripts whose targets do not exist. A
`needs-setup` report indicates missing configuration or dependencies.

After the app starts, run `probe`. Success requires 2xx responses from
`/health`, `/schedule`, and `/analytics`, plus `{"status":"healthy"}` from
`/health`.

## Pitfalls

1. **Default-profile paths.** Never hardcode the installed helper beneath the
   default home. Resolve `SKILL_DIR` from `skill_view` or active `HERMES_HOME`.
2. **Turnkey claims.** The inspected upstream revision references missing
   `workflows/daily-content-pipeline.js`,
   `workflows/weekly-strategy-review.js`, and `database/init.js` targets.
3. **Gemini-only setup.** Upstream credential validation currently expects
   `youtube` and `openai`; do not present Gemini-only configuration as tested.
4. **Planning versus execution.** Thumbnail and production stages create
   briefs and assembly plans unless an external image, audio, or video tool is
   actually available and exercised.
5. **Premature completion.** Do not mark a stage complete before its artifacts
   are accepted and persisted in the workspace.

## Verification

Before reporting success:

- [ ] `status` identifies the expected current and completed stages.
- [ ] Every completed stage contains notes and accepted artifacts.
- [ ] `export` contains each completed stage under `deliverables`.
- [ ] The workspace is beneath the active profile or the requested output path.
- [ ] External-project claims are backed by a fresh `inspect` report.
- [ ] Running-server claims are backed by a successful `probe` report.
- [ ] Credential, rendering, and upload dependencies are stated honestly.

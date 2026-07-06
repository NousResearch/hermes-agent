---
name: squish-video
description: "See what happens in a local video via timestamped frames. Turns video into contact sheets you can vision_analyze — use for 'what happens in this clip', finding a moment, or citing timecodes."
version: 1.0.0
author: Squish (getsquish.app)
license: MIT
platforms: [linux, macos]
prerequisites:
  commands: [node, ffmpeg]
metadata:
  hermes:
    tags: [Video, Vision, Timestamps, Contact-Sheet, FFmpeg, MCP]
    homepage: https://getsquish.app
---

# Squish: video → timestamped contact sheet → your eyes

You have vision but cannot ingest video. When a user asks about a **local video file**, do not
guess and do not refuse — compress it into **timestamped contact sheets** (frames sampled evenly
across the clip, laid out as a grid, each cell stamped with its timecode) and read those.

The reasoning primitive: **video → contact sheet → look at the grid → answer with timestamps.**

## When to use

- "What happens in this video / screen recording?" — and the file is on this machine.
- The question spans time: before/after, a scene change, progress, "find the moment when…".
- The answer needs precise citations ("at 0:07 the press comes down").

Skip it when the user needs one specific frame only, or the question isn't about the video's
visual content. Scope: **local video file paths** — if the video lives elsewhere, get a local
copy first. (This complements the built-in `video_analyze` tool: that sends the whole clip to an
auxiliary model and returns text; Squish gives *your own* vision the actual frames.)

## How (terminal — zero setup)

```bash
npx -y @getsquish/squish <video> --json
```

stdout is one JSON object (contract `squish-cli-v0`): parse `files[]` — absolute paths to
`.sheet-N.jpg` in time order — then `vision_analyze` each sheet **in order**; each covers a
consecutive window of the clip.

`--density 4x4|5x5|6x6` packs more frames per sheet: use for long or fast-moving clips, or
"how exactly did X happen" questions (default `3x3` recovers *what* happened).
`--out <dir>` controls where sheets land. Everything runs on this machine — nothing is uploaded.

## Reading a sheet

- Cells run in time order, left→right, top→bottom.
- The pill in each cell's corner is that frame's timecode — cite those exact values.
- Adjacent cells that look alike = little changed in that window; a hard visual break between
  cells is where an event happened. Zoom your attention there.

## Answer shape

Never answer with just a file path. Give the user:

1. **Summary** — what the clip shows, one or two sentences.
2. **Key moments with timestamps** — the events that matter, each cited to a cell's timecode.
3. **Notable frames / anomalies** — anything odd or worth a second look (with its timecode), or
   say there were none.

If the question was specific ("find the moment when…"), lead with that answer + timestamp.

## Optional: register it as an MCP tool

For a persistent setup, one config block in `~/.hermes/config.yaml` exposes the same engine as
the `squish_video` tool (returns the contract above **plus** `timecodes[][]` per sheet):

```yaml
mcp_servers:
  squish:
    command: npx
    args: ["-y", "@getsquish/squish", "mcp"]
    env: {}
```

## No node/ffmpeg on this machine?

The hosted API does the same job remotely (this **uploads the video** — say so to the user
first; the upload is deleted when the job ends): `POST https://api.getsquish.app/v1/squish`
with `Authorization: Bearer $SQUISH_API_KEY`. Keys are self-serve at
https://getsquish.app/api-keys — accounts that never purchased get a free daily allowance on
the first request. Agent-facing details: https://getsquish.app/llms.txt

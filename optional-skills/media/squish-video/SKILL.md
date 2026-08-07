---
name: squish-video
description: Read a local video as timestamped contact sheets.
version: 1.0.0
author: Natthawut Phurahong (v1b3x0r), Squish (getsquish.app)
license: MIT
platforms: [linux, macos]
prerequisites:
  commands: [npx, ffmpeg]
metadata:
  hermes:
    tags: [Video, Vision, Timestamps, Contact-Sheet, FFmpeg, MCP]
    homepage: https://getsquish.app
---

# Squish Video Skill

Compresses a local video file into timestamped contact sheets — frame grids where every
cell carries its absolute timecode — so `vision_analyze` can read the clip with the main
model's own vision. Visual content only: no audio is decoded, and nothing is uploaded.
Complements the built-in `video_analyze` tool, which sends the whole clip to an auxiliary
model and returns text.

## When to Use

- "What happens in this video / screen recording?" — and the file is on this machine.
- The question spans time: before/after, a scene change, progress, "find the moment when…".
- The answer needs precise citations ("at 0:07 the press comes down").

Skip it when only one known frame matters, when the question isn't about visual content
(speech, music — no audio is read), or when the video isn't a local file (get a local
copy first).

## Prerequisites

- `npx` (ships with Node.js ≥ 20, the engine's floor) and `ffmpeg` on PATH. The helper script shells out to
  the open-source Squish CLI (`@getsquish/squish`, Apache-2.0 —
  [github.com/getsquish/squish](https://github.com/getsquish/squish)), fetched by `npx`
  on first use; frames are extracted locally by ffmpeg.
- Helper script: `~/.hermes/skills/media/squish-video/scripts/squish_video.py`
  (stdlib only).
- Optional persistent alternative — the same engine speaks MCP. One block in
  `~/.hermes/config.yaml` exposes the tool `squish_video` (same contract plus
  `timecodes[][]` per sheet):

  ```yaml
  mcp_servers:
    squish:
      command: npx
      args: ["-y", "@getsquish/squish", "mcp"]
  ```

## How to Run

Invoke through the `terminal` tool. The script passes the path as a single argument, so
whitespace in filenames is safe:

```bash
python3 ~/.hermes/skills/media/squish-video/scripts/squish_video.py "<video-path>"
```

stdout is one JSON object (contract `squish-cli-v0`); `files[]` lists absolute sheet
paths in time order. Then `vision_analyze` each sheet **in order** — each covers a
consecutive window of the clip.

## Quick Reference

```bash
squish_video.py <video>                          # 3x3 overview: WHAT happened
squish_video.py <video> --density 5x5            # denser grid: HOW it happened, long clips
squish_video.py <video> --start 1:00 --end 1:30  # zoom into a range
squish_video.py <video> --out <dir>              # choose where sheets land
```

- `--density`: `3x3` (default) · `4x4` · `5x5` · `6x6`.
- `--start` / `--end`: seconds (`90`) or a timecode exactly as stamped on a sheet
  (`1:30`, `1:07.3`). Timecodes stay **absolute to the source video** at every depth.

## Procedure

1. **Overview.** Run the script with defaults; `vision_analyze` every sheet in `files[]`,
   in order.
2. **Read the grid.** Cells run in time order, left→right, top→bottom; the pill in each
   cell's corner is that frame's timecode. Adjacent near-identical cells mean little
   changed; a hard visual break between cells is where an event happened.
3. **Zoom when the question needs precision.** Re-run with `--start`/`--end` bracketing
   the timecodes around the event (add a denser `--density` for fast motion); repeat
   until the moment is pinned.
4. **Answer.** Lead with the asked-for moment and its timestamp, then: a one-two sentence
   summary · key moments, each cited to a cell's timecode · anomalies worth a second look
   (or say there were none). Never answer with just a file path.

## Pitfalls

- **No audio.** Sheets carry visuals only — say so if the user asks about speech or sound.
- **Cite pill timecodes only**; never claim precision finer than the spacing between cells.
- Long clips emit multiple sheets — read all of them before concluding.
- A too-narrow window is rejected with a teaching error (addressable floor ~2 ms/cell) —
  widen the range instead of retrying.
- No `npx`/`ffmpeg` on this machine: the hosted API does the same job remotely
  (`POST https://api.getsquish.app/v1/squish`, self-serve keys at
  https://getsquish.app/api-keys), but it **uploads the video** — tell the user before
  using it. Agent-facing details: https://getsquish.app/llms.txt

## Verification

- The script on a short clip exits `0` and prints JSON with
  `"contract": "squish-cli-v0"` and a non-empty `files[]`.
- Every path in `files[]` exists (`search_files target='files'`) and renders in
  `vision_analyze` with readable timecode pills.
- With prerequisites missing, the script fails fast with an actionable message
  (`npx not found…`), not a stack trace.
- Hermetic tests: `scripts/run_tests.sh tests/skills/test_squish_video_skill.py -q`.

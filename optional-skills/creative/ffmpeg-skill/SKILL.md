---
name: ffmpeg-skill
description: Local FFmpeg video and audio editing scripts, upstream-kept.
version: 2.5.1
author: kajisho5
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [ffmpeg, video, audio, captions, subtitles, transcode, loudness, reels, shorts, editing, local]
    category: creative
    related_skills: [ascii-video, manim-video, ai-presenter-video]
    upstream:
      repo: kajisho5/ffmpeg-skill
      path: .
---

# ffmpeg-skill (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [kajisho5/ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill): the
> repository root is the skill directory (`SKILL.md` + `scripts/`,
> `templates/`, `references/`). `hermes skills install
> official/creative/ffmpeg-skill` pulls the current tree live from that repo
> (quarantined and scanned like any hub install) — this directory holds only
> the catalog metadata, so the 42 scripts can never lag upstream's release
> cadence.

ffmpeg-skill turns a natural-language editing request ("make it vertical and
60 seconds", "caption this reel", "normalise to -14 LUFS for YouTube") into
runs of 42 standard-library Python scripts that drive a local `ffmpeg`:
probe, cut/join, fit/crop/pad, speed ramps, captions and subtitles (SRT/ASS,
karaoke), overlays and lower-thirds, silence removal, multicam and external
mic sync, loudness, HDR→SDR and LUTs, music ducking, platform exports, scene
detection and highlight reels, contact sheets, and a `render.py` project file
for multi-step edits. Every writing script takes `--dry-run --json` to plan
before it encodes, and `check.py --platform` verifies the deliverable against
YouTube / Reels / TikTok / X constraints before you report. Nothing leaves the
machine: no cloud, no API keys.

## Prerequisites

- `ffmpeg` and `ffprobe` on `PATH` (FFmpeg 5+; Hermes' own tool store ships
  FFmpeg 9 under `~/.hermes/tools/`) and Python 3.9+. Run
  `python3 <skill-dir>/scripts/_contract.py doctor` only after a failure,
  per the upstream workflow.
- Installs pull the whole repository (~320 files, ~19 MB; `docs/` and
  `evals/` are most of it) from GitHub, which takes a few minutes; the fetch
  is pinned to one tree SHA, recorded in the bundle metadata.
- Upstream's `bin/install.js` and `mcp/server.py` are for other harnesses
  (npm / other-harness plugin install, MCP exposure); Hermes runs the scripts
  directly and needs neither. Nothing in the skill phones home.
- Upstream publishes several releases a week; `hermes skills update
  ffmpeg-skill` re-pins to the current tree. `check.py` accepts
  `--platform instagram` as an alias for `reels`.

## When to prefer it

- Any request that names a media file (mp4, mov, mkv, wav, m4a), footage, a
  clip, captions, a reel/short, LUFS, sync, or transcoding — even when the
  user does not say "edit".
- `ascii-video` and `manim-video` generate footage and `ai-presenter-video`
  builds a presenter pipeline; ffmpeg-skill is the general-purpose editor for
  footage you already have.

Full documentation: https://github.com/kajisho5/ffmpeg-skill#readme

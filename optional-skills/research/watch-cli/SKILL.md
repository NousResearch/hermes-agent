---
name: watch-cli
description: Turn a video URL into still frames and a transcript.
version: 1.0.0
author: Son Piaz (sonpiaz)
license: MIT
platforms: [linux, macos]
prerequisites:
  commands: [yt-dlp, ffmpeg, ffprobe, jq, curl, python3]
metadata:
  hermes:
    tags: [Video, Transcription, Frames, Research, Multimedia]
    related_skills: [youtube-content, whisper]
    homepage: https://github.com/sonpiaz/watch-cli
---

# watch-cli Skill

watch-cli downloads a video from a URL (YouTube, X, LinkedIn, TikTok, Reddit, Vimeo, Facebook), extracts evenly spaced frames, and transcribes the audio, then prints one versioned text block or JSON object. It supplies raw materials only: the agent reads the frames as images and the transcript as text, and does the reasoning itself. It does not summarize, and it does not read browser cookies unless the user opts in.

## When to Use

- The user shares a video URL and asks to summarize, explain, or walk through it.
- The user wants to implement, clone, or replicate what is shown on screen in a video.
- The user wants a diagram, notebook, or step-by-step guide built from a talk or tutorial.
- The user pastes a video URL with no instruction: ask what they want first, then run it.

## Prerequisites

Install watch-cli (MIT, bash) with either method:

```bash
# macOS: Homebrew tap
brew tap sonpiaz/tap && brew install watch-cli

# Linux or macOS: pinned release installer (verifies the tarball SHA256)
curl -fsSL https://github.com/sonpiaz/watch-cli/releases/download/v0.3.4/install.sh | WATCH_CLI_VERSION=0.3.4 bash
```

The installer puts the scripts in `~/.watch-cli/bin/` (or `$WATCH_CLI_HOME/bin/`) and symlinks them into `~/.local/bin/`. Homebrew links them under `$(brew --prefix watch-cli)/bin/`.

Transcription needs a backend. Hosted mode reads `KYMA_API_KEY` from the environment or from `~/.config/watch-cli/env`. Offline mode runs whisper.cpp: install with `install.sh --with-local`, then set `WATCH_AUDIO_MODE=local`. With neither, frames are still extracted and the run exits `4`.

## How to Run

The command is named `watch`, which collides with the procps `watch` command present on most Linux systems. Never run a bare `watch`. Resolve watch-cli's own path once through the `terminal` tool and confirm it before use:

```bash
WATCH_BIN="${WATCH_CLI_HOME:-$HOME/.watch-cli}/bin/watch"
if [ ! -x "$WATCH_BIN" ] && command -v brew >/dev/null 2>&1; then
  WATCH_BIN="$(brew --prefix watch-cli)/bin/watch"
fi
ARCHIVE_BIN="$(dirname "$WATCH_BIN")/watch-archive"
case "$("$WATCH_BIN" --version 2>/dev/null)" in
  watch-cli*) echo "using $WATCH_BIN" ;;
  *) echo "watch-cli not found at $WATCH_BIN; install it first" >&2 ;;
esac
```

Then call it by that path:

```bash
"$WATCH_BIN" "<url>" 8 --format json
```

## Quick Reference

| Task | Command |
|---|---|
| Watch, JSON output | `"$WATCH_BIN" "<url>" [frames] --format json` |
| Watch, labeled text block | `"$WATCH_BIN" "<url>" [frames]` |
| Many URLs, one JSON line each | `"$WATCH_BIN" --pipe < urls.txt` |
| Ignore the cache | add `--no-cache` |
| Search earlier transcripts | `"$ARCHIVE_BIN" find "<phrase>"` |
| List or reprint earlier runs | `"$ARCHIVE_BIN" ls` / `"$ARCHIVE_BIN" get <id-or-url>` |

Exit codes: `0` success · `2` missing dependency · `3` download failed (stderr carries `tag=…`) · `4` transcription failed, frames present · `64` usage error.

## Procedure

1. Resolve and confirm `WATCH_BIN` as shown in How to Run.
2. If the user asks about something already watched, search the archive first with `"$ARCHIVE_BIN" find "<phrase>"`. Re-watching a URL is a cache hit, but the answer may already be on disk.
3. Pick a frame count: 8 by default, 16 for fast-cut UI demos, 24 to 32 for long talks.
4. Run `"$WATCH_BIN" "<url>" <frames> --format json` through `terminal` and parse the JSON: `version`, `video_path`, `duration_sec`, `frame_paths` (earliest first), `transcript` (string or `null`), `exit_code`.
5. Pass each path in `frame_paths` to `vision_analyze`; treat `transcript` as plain text.
6. Produce what the user asked for. Prompt templates for common outputs (implement from video, extract architecture, paper to code, tutorial walkthrough, clone UX) live in the watch-cli repo under `prompts/`.

## Pitfalls

- A bare `watch "<url>"` can start procps `watch`, which re-runs the URL as a shell command every two seconds and never returns. Always call the resolved `WATCH_BIN`.
- Exit `4` is partial success: frames exist and `transcript` is `null`. Continue with frames instead of failing.
- Login-walled posts fail with `tag=download-auth`. Ask the user before retrying with `WATCH_BROWSER=auto` (reads cookies from the local browser) or `--cookies <file>`; never set either silently.
- Do not parse stderr or frame filenames. The JSON fields and the order of `frame_paths` are the contract.
- In hosted mode the audio track is uploaded for transcription; frames and the video stay local. Use offline mode when audio must not leave the machine.

## Verification

```bash
"$WATCH_BIN" --version
```

It prints `watch-cli v0.3.4` or newer. Anything else, such as `watch from procps-ng`, means `WATCH_BIN` points at the wrong program.

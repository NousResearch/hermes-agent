---
name: watch-me
description: "Record or live-watch a performance and judge it."
version: 0.2.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Recording, Video, Feedback, Performance, Screen]
    related_skills: [songsee]
---

# Watch Me Skill

Watch what the user is doing on screen — playing a game, performing music,
designing a sound — and judge it. Two modes: **review** a recorded take, or
**live** commentary while they work. Capture is always explicit: it starts and
stops on request, and nothing is observed in between.

## When to Use

- "Watch me play/perform this and tell me what you think"
- "Watch how I use my synths" — live, while they tweak
- "Record my screen for the next few minutes, then review it"
- The user already has a recording and wants a critique of it
- Judging timing, pacing, technique, or UX from motion rather than a still

Don't use for: single-moment questions (use `vision_analyze` on a screenshot),
ambient observation the user did not ask for, or reading text off a window (use
`read_window_below` on the desktop).

## Prerequisites

- The `watch` plugin enabled in `config.yaml` under `plugins.enabled`.
- `ffmpeg` and `ffprobe` on PATH.
- A video-capable model. Gemini-family models and several open VL models accept
  video or image frames; **Claude does not accept video input at all**. Live
  mode pins its own default (`auxiliary.watch.model` in config.yaml) rather than
  inheriting the conversation's, because a non-vision model produces an endlessly
  silent loop that looks like a quiet session.
- Audio (review mode only) needs a device the OS exposes:
  - **Windows** — a dshow device name, e.g. `--audio-device "Stereo Mix (Realtek Audio)"`.
  - **macOS** — an avfoundation device *index*, not a name.
  - **Linux** — a PulseAudio source; a `.monitor` source captures system output.

## How to Run

In a chat surface (desktop app, TUI, messaging) use the slash command — it needs
no terminal:

```
/watch live how I use my synths
/watch status
/watch stop
/watch replay
```

From a terminal, or when you want the flags, drive the CLI through `terminal`:

```
terminal(command="hermes watch live -b \"how I use my synths\"", timeout=3600)
terminal(command="hermes watch start --label solo-take")
terminal(command="hermes watch stop")
terminal(command="hermes watch review -q \"how was my timing?\"", timeout=600)
terminal(command="hermes watch replay --sweep")
```

Prefer `/watch live` when the user is in a GUI: it detaches, so the session stays
usable while watching continues, and `/watch stop` reports everything it said.

## Quick Reference

| Command | Purpose |
|---|---|
| `/watch live <brief>` | Start live commentary from any chat surface |
| `/watch stop` / `status` | Stop it, or check in |
| `/watch replay` | Compare thresholds against the last session |
| `hermes watch live -b "<brief>"` | Same, with flags |
| `hermes watch live --duration 600` | Stop automatically after N seconds |
| `hermes watch live --min-salience 0.4` | Speak less (higher bar to look) |
| `hermes watch replay --sweep` | Re-tune the last session for free |
| `hermes watch start --label <name>` | Begin recording a take |
| `hermes watch stop` | Finalize the take |
| `hermes watch status` / `list` | What's running / what's recorded |
| `hermes watch cost` | Token estimate — no API call |
| `hermes watch review -q "<q>"` | Prepare and analyze a take |
| `hermes watch review --speed 2` | Halve the token cost |
| `hermes watch review --low-res` | ~4x cheaper frames |
| `hermes watch review --no-titles` | Keep app names, drop window titles |

## Procedure

### Live commentary

1. **Get a brief.** "Watch me" with no brief produces generic commentary; "watch
   whether I'm clipping cooldowns" produces useful commentary. Ask if unstated.
   Completion: you can state what it is watching for.
2. **Start it** with a generous `timeout` — this runs until stopped. Relay each
   comment as it arrives. Completion: the loop reported its frame count and call
   rate on exit.
3. **Tune for free afterwards.** `hermes watch replay --sweep` re-runs the
   recorded decisions against a range of thresholds without spending anything.
   If it talked too much, raise `--min-salience`; too little, lower it.
   Completion: a threshold that matches what the user wanted.

### Reviewing a take

1. **Confirm what to record and the question.** Completion: both stated.
2. **`hermes watch start`**, adding `--audio-device` when audio matters (it does
   for music). Read the command's notes back — a missing audio device means a
   silent take, worth knowing before performing.
3. **Wait for the user to say stop.** Do not guess a duration.
4. **`hermes watch stop`.** If it reports capture lag, mention it: the backend
   dropped frames and a lower `--fps` helps next time.
5. **`hermes watch cost`** before spending. Anything over ~10 minutes deserves a
   mention of `--speed` or `--low-res`.
6. **`hermes watch review -q "<their question>"`** with a generous timeout.
   Relay the answer with its timestamps intact.

## Pitfalls

- **Wayland without XWayland cannot be captured.** Both modes refuse and name a
  portal recorder instead. A compositor restriction, not a bug — do not retry.
- **Providers sample video at ~1 fps.** A higher `--fps` on review does not show
  the model more; retiming does. `--speed 0.5` gives two samples per real second
  at double the tokens.
- **Resolution does not change the token bill.** Width affects file size only.
  Use `--low-res` or `--speed` to spend less.
- **Audio is ~8x cheaper per second than video.** For a music take, keep the
  audio and consider a lower frame rate rather than the reverse.
- **Live mode is quiet on purpose.** Most seconds produce no model call at all:
  the screen must have changed, a refractory window must have passed, and the
  comment must not repeat something already said. A session with two comments in
  ten minutes is working, not broken — check the call rate on exit.
- **Window titles carry private text** (documents, URLs, message previews). Use
  `--no-titles` when the recording is shared.
- **A retimed clip's timestamps are clip time, not real time.** The prompt says
  so; keep that framing when relaying the answer.
- **Takes are not cleaned up automatically.** They accumulate under the profile's
  `workspace/watch/`; offer to remove old ones.
- **For audio-only questions, a spectrogram may beat the video.** The `songsee`
  skill renders tempogram/chroma panels that `vision_analyze` reads for a
  fraction of the tokens.

## Verification

- `hermes watch status` reports not-recording after a stop.
- `hermes watch list` shows the take with `+timeline` when the window track was
  captured.
- Live mode prints frames, model calls, and call rate on exit. **Zero frames
  means capture failed** — it says so rather than reporting a quiet session.
- The review output cites specific moments; if it is generic, the question was
  too vague — re-ask with a sharper one rather than re-recording.


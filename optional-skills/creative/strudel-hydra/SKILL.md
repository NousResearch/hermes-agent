---
name: strudel-hydra
description: Live-code and self-evolve synchronized Strudel audio + Hydra visuals.
version: 1.0.0
author: mrev-lab
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [strudel, hydra, live-coding, audio-visual, generative-audio, generative-visuals, creative-coding, sse, self-improving, evolution, tidalcycles, vj]
    related_skills: [supercollider, pd-patching, songwriting-and-ai-music, darwinian-evolver]
---

# Strudel + Hydra Skill

Drive a synchronized audio/visual liveset in a browser and hot-swap it while it
runs. A tiny standard-library HTTP server hosts one page that runs Strudel
(pattern-based audio) in the page and Hydra (audio-reactive visuals) in an
isolated iframe; the agent pushes new "sets" over Server-Sent Events (SSE) and
the running page evaluates them without a reload. Same DNA as `pd-patching` and
`supercollider` — no MCP, no npm, no compiled bridge, driving a live instance
over a thin protocol — here the transport is SSE and both sound *and* picture
come from one push. The page also **measures its own audio output** and reports
it back, closing the live-coding loop so the agent can score a set and evolve it.
It exports a standalone self-contained `.html`. It does **not** cover the Strudel
CodeMirror editor, the TidalCycles Haskell stack, or compiling Hydra.

## When to Use

Use when the user wants a generative or live-coded audio/visual set: a techno
pulse with a strobing kaleidoscope, an ambient drone under drifting noise,
audio-reactive visuals, a VJ loop — described in words, produced as running
sound and picture. Skip it for audio-only DSP (use `supercollider` or
`pd-patching`) or for a static rendered video file.

## Example Requests

What a user might say to Hermes, and what the agent does:

| User says | Agent does |
|-----------|------------|
| "Play a dark techno set with a strobing kaleidoscope" | boot the server, open the page, push an `--audio`/`--visual` set |
| "Make it an ambient drone with slow drifting visuals" | hot-swap a calm pad + slow Hydra set (one push) |
| "More red, and speed up the beat" | re-push the running set with tweaked color/tempo |
| "Export this acid set as a web page" | `sh_examples.py acid --export acid.html` |
| "VJ on your own and keep making it more energetic" | headless browser + audio-fitness `sh_evolve.py` on `/loop` |
| "Too loud — stop" | `sh_client.py --hush` |

## Prerequisites

- A **modern browser** (Chrome/Firefox) open on a desktop session — this drives
  real Web Audio + WebGL, so it needs a display and speakers, not a headless box.
- **Internet on first load**: the page fetches the pinned `@strudel/web` and
  `hydra-synth` bundles from a CDN (the browser fetches them, not npm).
- The `terminal` tool, to run the server and push scripts. No Python packages —
  standard library only.
- A way to open a URL: `open <url>` (macOS) / `xdg-open <url>` (Linux).

## How to Run

Run every command through the `terminal` tool. Locate the installed skill first
(the scripts import each other by name, so run them from `scripts/`):

```bash
SKILL="${HERMES_HOME:-$HOME/.hermes}/skills/creative/strudel-hydra"
cd "$SKILL/scripts"
```

Boot the server once (leave it running) and open the page:

```bash
python3 sh_server.py &                 # http://127.0.0.1:8765
open http://127.0.0.1:8765             # Linux: xdg-open http://127.0.0.1:8765
```

**Click "start" in the page.** Browsers require a user gesture before audio can
begin; the overlay handles that once. No microphone is used or requested.

Push a gallery set, or arbitrary code, in one stdin-free command:

```bash
python3 sh_examples.py pulse                              # push a gallery set
python3 sh_client.py \
  --audio 'note("c3 e3 g3 b3").s("sawtooth").lpf(700).gain(.5)' \
  --visual 'osc(20, 0.1, 0.8).kaleid(5).rotate(0, 0.1).out()' \
  --label riff
python3 sh_client.py --hush                               # panic: stop + clear
```

> Pass code as **arguments**, never via a heredoc — an agent terminal often
> fails to deliver heredoc stdin and the call hangs. For a long set, write the
> code to a file and use `--audio-file` / `--visual-file`.

## Quick Reference

Helper scripts (all in `scripts/`, standard library only):

| File | Purpose |
|------|---------|
| `sh_server.py` | HTTP host + SSE broker. Serves the page, fans the latest set out to every browser, retains it for late joiners. `--host`/`--port`. |
| `sh_client.py` | Push a set (`--audio`/`--visual`/`--audio-file`/`--visual-file`/`--label`), `--hush`, `--status`, `--observe [--target … --wait …]`. Exposes `push_set()`, `observe()`, `score()`. |
| `sh_examples.py` | Gallery (`pulse`, `bells`, `drone`, `acid`) + `--list` and `--export FILE`. |
| `sh_evolve.py` | Score + rank one generation of candidate sets against a fitness `--target` (the evaluation/selection step of the self-improving loop). |
| `templates/page.html` | The host page: runs Strudel in the page and Hydra in an isolated iframe, opens the SSE stream, hot-swaps each set, and posts audio telemetry. Also the export scaffold. |

A **set** is JSON — `audio` (Strudel code), `visual` (Hydra code), both, or
neither with `cmd:"hush"`:

| Field | Runs via | Example |
|-------|----------|---------|
| `audio` | Strudel `evaluate()` | `note("c3 e3 g3").s("sawtooth")` |
| `visual` | Hydra (globals) | `osc(20).kaleid(5).out()` |
| `label` | HUD tag | `"riff"` |
| `cmd:"hush"` | stop audio + black screen | via `--hush` |

## Procedure

The agent turns a request into a running liveset like this:

1. **Ensure the server is up and a browser is attached.** `python3 sh_client.py
   --status` reports `subscribers`; `0` means no browser is on the page yet
   (open it and click start).
2. **Pick the cheapest path.** Canned vibe → a gallery name; a tweak → push a
   modified `--audio`/`--visual`; novel → write the set from scratch.
3. **Push audio and visual together.** Keep them synchronized by sending one set;
   drive visuals from the sound with `a.fft[0..3]` in the Hydra code.
4. **Verify it landed.** `--status` and the server's `subscribers` count confirm
   delivery; the page HUD shows `applied · <label>` or an error string.

**Share an artifact.** Export a standalone page that plays without the server
(the set is baked into the HTML, no SSE):

```bash
python3 sh_examples.py acid --export acid.html      # open acid.html anywhere
```

## Self-Improving Loop

Live-coding is already an observe→modify→repeat loop; the page closes it by
reporting **telemetry** — features it measures from the running set — which the
agent uses as a fitness signal. Reported each ~0.75 s:

| Field | Meaning | Source |
|-------|---------|--------|
| `level` | overall audio energy (mean of bins) | Strudel output tap |
| `bands` | per-band FFT magnitudes (8) | Strudel output tap |
| `centroid` | spectral brightness, 0–1 | Strudel output tap |

**Telemetry is audio-only** — the output tap works headless, muted, and mic-free
(verified in a live browser). There is no visual telemetry: Hydra's WebGL canvas
can't be read back in-page (the drawing buffer isn't preserved), so visuals are
judged by eye. An unattended evolving loop can therefore run the browser
headless and score on the audio features alone.

The agent runs the loop (it is the mutation operator; the scripts measure,
score, and select):

1. **Set a fitness target** — an audio feature vector for the mood, e.g.
   `{"level":0.6,"centroid":0.4}`. `score()` is `1/(1+distance)` over the shared
   scalar keys (`level`, `centroid`); higher is closer.
2. **Read the current set:** `python3 sh_client.py --observe --target '{…}'`.
3. **Mutate:** write 2–4 variations of the best set (nudge patterns, filters,
   Hydra ops) into a candidates JSON list.
4. **Evaluate + select:** `python3 sh_evolve.py cands.json --target '{…}'
   --wait 1.5` pushes each, lets it settle, and prints them ranked best-first.
5. **Keep the winner and repeat.** For novelty (open-ended evolution with no
   fixed target), also reward distance from recent sets — track that history
   yourself between rounds.

Run it unattended with the `/loop` skill so the VJ set evolves on its own; use
`--hush` to stop. This is the `darwinian-evolver` shape with the code as
organism, the agent as mutator, and telemetry as the evaluator — but here the
fitness signal comes free from the analyser.

## [CRITICAL] Safety

Two failure modes need active care — both have a one-command escape via
`--hush`:

- **Loud audio.** Keep master level sane: chain `.gain(<= 0.8)` on audio
  patterns and never stack many full-gain voices. `python3 sh_client.py --hush`
  cuts all sound immediately. If a request implies danger ("make it deafening"),
  keep the gain bounded.
- **Photosensitive epilepsy.** Full-field luminance flashing faster than ~3 Hz
  can trigger seizures. Do not build rapid full-screen black↔white/red strobes.
  Prefer motion, hue shifts, and partial-frame changes; keep fast `a.fft`-driven
  brightness swings local, not full-frame. `--hush` also blacks the screen.

## Pitfalls

- **Audio needs the start click.** Strudel's AudioContext stays suspended until
  the user gesture; a set pushed before clicking "start" is retained and applies
  on click. If nothing sounds, confirm the overlay was clicked.
- **Hydra is isolated in an iframe on purpose.** Strudel's `initStrudel()` spins
  up a WebGL context that invalidates Hydra's shader program in the *same*
  document — the canvas goes blank/gray with `useProgram: program not valid` in
  the console. Hydra therefore runs in its own `srcdoc` iframe (separate GL
  context); visual code and `a.fft` cross via `postMessage`. Do not move Hydra
  back into the main page.
- **Visuals react to the music with no mic.** Hydra runs `detectAudio:false`; the
  page feeds `a.fft` from the Strudel-output tap into the iframe every frame, so
  `a.fft[...]` in visual code reacts to the actual sound. No mic is requested, and
  `a.fft` is always an 8-element array — a visual referencing it can't throw and
  blank the render (the mic-less failure mode this replaced).
- **A bad set shows in the HUD, not the terminal.** Evaluation happens in the
  browser; a syntax error prints `visual error:` / `audio error: …` on the page
  HUD. Watch it, or keep sets small and incremental.
- **Function names track the pinned versions.** The gallery uses common Strudel
  (`note`, `s`, `lpf`, `room`, `sine.range`) and Hydra (`osc`, `voronoi`,
  `kaleid`, `color`, `rotate`, `modulateRotate`, `modulateScale`,
  `modulatePixelate`) names verified against the bundles pinned in `page.html`.
  Not every documented Hydra op exists in that build — `scale`, `repeat`,
  `colorama` and the `noise`/`shape` sources threw `… is not a function` — so
  the HUD names any bad op; prefer the ops the gallery already uses, and adapt
  if you repin.
- **One set = one screen.** Pushing a new set replaces the whole liveset on
  every connected browser; there is no per-client targeting.
- **Telemetry is audio-only.** Features tap Strudel's output directly (an
  analyser before the speakers), so they work muted and need no mic — but they
  read low while a pattern sits between hits, so give each candidate enough
  `--wait` to settle. There is **no visual telemetry**: Hydra's WebGL drawing
  buffer isn't preserved, so in-page readback is black on every browser tried
  (real GPU included) — judge visuals by eye. Read scores as a relative ranking,
  not truth.

## Verification

1. **Server up:** `curl -s 127.0.0.1:8765/status` returns `{"subscribers": N}`;
   no answer means `sh_server.py` is not running.
2. **Browser attached:** after opening the page, `subscribers` is `>= 1`.
3. **Push delivered:** `sh_client.py` / `sh_examples.py` print
   `{"ok": true, "subscribers": N}`; the page HUD flips to `applied · <label>`.
4. **Stream sanity (headless):** `curl -sN 127.0.0.1:8765/events` prints
   `: connected`, then each pushed set as a `data: {…}` frame — proves the SSE
   relay without a browser.
5. **Telemetry flowing:** with a browser on the page, `sh_client.py --observe`
   returns `data` (not null) with a small `age`; a null `data` means the page
   is not posting (not started, or blocked). The endpoint itself round-trips
   headlessly: `POST` a JSON body to `/telemetry`, then `--observe` echoes it.
6. **Export:** `sh_examples.py <name> --export out.html` writes a file
   containing `window.__SET__`; opening it plays the set with no server.

## Security

The server binds `127.0.0.1` only — keep it there; do not bind `0.0.0.0`. The
page **evaluates pushed code** (Strudel `evaluate`, and `eval` of Hydra code),
so anything reaching `/push` runs in the browser: only push sets you author, and
never expose the port beyond loopback. The CDN bundles are version-pinned in
`templates/page.html`; review a bump before trusting it.

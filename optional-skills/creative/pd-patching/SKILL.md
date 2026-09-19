---
name: pd-patching
description: Build and control Pure Data audio patches live.
version: 1.0.0
author: mrev-lab
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [puredata, pd, dsp, audio, synthesis, dynamic-patching, netreceive, creative-coding, generative-audio, midi, osc]
    related_skills: [supercollider, songwriting-and-ai-music, comfyui]
---

# Pure Data Patching Skill

Generate and control sound in a running Pure Data instance by sending its own
canvas-editing messages (`obj`, `connect`, `clear`, `dsp`) to a `[netreceive]`
socket over TCP, entirely from Python's standard library — no MCP server, no
externals, no compiled bridge. It builds patches, plays and live-controls them,
exports standalone `.pd` files, and records WAV. It does **not** compile Pd
externals or drive the Pd GUI editor.

## When to Use

Use when the user wants to generate or control audio/DSP in Pure Data: a synth,
an effects chain, an LFO/filter sweep, a generative patch, or MIDI/OSC
processing — described in words, produced as sound. Skip it for tasks that need
compiled externals or hand-editing in the Pd GUI.

## Prerequisites

- **Pd-vanilla** installed (provides the `pd` binary):
  - macOS: `brew install --cask pd`
  - Linux: `sudo apt install puredata`
- The `terminal` tool, to launch Pd and run the helper scripts.
- **The receiver stub must be open in Pd** so the TCP socket is listening (see
  How to Run). No Python packages — standard library only.

## How to Run

Run everything through the `terminal` tool. Helpers import each other by name, so
run them from `scripts/`:

```bash
SKILL="${HERMES_HOME:-$HOME/.hermes}/skills/creative/pd-patching"
cd "$SKILL/scripts"
```

Open the receiver stub once (it opens `[netreceive 9999]`), then confirm the
handshake:

```bash
# macOS cask: the pd binary lives inside the .app bundle
PD="$(ls -d /Applications/Pd-*.app 2>/dev/null | sort -V | tail -1)/Contents/Resources/bin/pd"
# Linux: PD=pd
"$PD" "$SKILL/templates/receiver_stub.pd" &     # add -nogui -stderr for headless
python3 pd_client.py "hello"                     # -> "pd-net-in: hello" in Pd; nonzero if down
```

Play a gallery patch, or an arbitrary chord, in one stdin-free command:

```bash
python3 pd_examples.py wobble_bass --send
python3 pd_examples.py sines --freqs 120 254 1201 --send
python3 pd_client.py "pd dsp 0"                  # stop DSP
```

> Pass everything as **arguments**, never via a heredoc — an agent terminal
> often fails to deliver heredoc stdin and the call hangs. To shape a custom
> patch, write a real `.py` file in `scripts/` and run it.

> **No leading `;`.** Over `[netreceive]` a leading semicolon does not route to a
> receiver (that is a message-box-only feature; verified empirically), so send
> `pd-target_canvas obj 100 100 osc~ 440` — no `;`. The client strips a stray one
> and appends the trailing terminator.

## Quick Reference

Helper scripts (all in `scripts/`, standard library only):

| File | Purpose |
|------|---------|
| `pd_patch.py` | `PdPatch` builder (auto layout, stable indices); DSP safety linter; `.pd` export (`to_pd_file`); WAV recorder (`to_dac(record=True)`). |
| `pd_client.py` | TCP socket bridge to `[netreceive]` (`PureDataClient`) + CLI. |
| `pd_examples.py` | Gallery of 7 patches + CLI (`--send`, `--export file.pd`, `--wav out.wav --seconds N`). |
| `templates/receiver_stub.pd` | Base patch `[netreceive 9999] → [route] → [send]`. Open in Pd first. |

Dynamic-patching messages (first atom is the receiver; no leading `;`):

| Action | Message |
|--------|---------|
| Clear canvas | `pd-target_canvas clear` |
| Create object | `pd-target_canvas obj {x} {y} {name} {args…}` |
| Create message box | `pd-target_canvas msg {x} {y} {text}` |
| Wire | `pd-target_canvas connect {srcIdx} {outlet} {dstIdx} {inlet}` |
| DSP on/off | `pd dsp 1` / `pd dsp 0` |

Object indices are 0-based in creation order (reset by `clear`); `pd_patch.py`
tracks them. Gallery: `sine`, `sines --freqs …`, `fm_bell`, `wobble_bass`,
`supersaw_drone`, `random_blips`, `resonator` (`python3 pd_examples.py --list`).

| Name | What you hear | Technique |
|------|---------------|-----------|
| `sine` | 440 Hz tone | minimal `osc~ → *~ → dac~` |
| `sines --freqs …` | chord of sines | N `osc~` chain-summed |
| `fm_bell` | metallic bell | FM — one `osc~` bends another's frequency |
| `wobble_bass` | dubstep "wob" | `[vcf~]` band-pass swept by an LFO |
| `supersaw_drone` | fat drone | three detuned `phasor~` saws mixed |
| `random_blips` | random melody | `[samphold~]` latches noise on a clock |
| `resonator` | string drone | Karplus-Strong `[delwrite~]`/`[delread~]` loop |

## Procedure

The agent turns a request into sound like this:

1. **Ensure the stub is up.** `python3 pd_client.py "hello"`; a `Connection
   refused` means Pd or the stub is not running — open it.
2. **Pick the cheapest path.** Canned sound → a gallery name; a tweak → call a
   gallery builder with params; novel routing → build a `PdPatch`.
3. **Build and check safety.** End every patch with `to_dac(...)` and confirm
   `is_safe()` before sending.

Build a custom patch in a `.py` file under `scripts/`:

```python
from pd_patch import PdPatch
from pd_client import PureDataClient

p = PdPatch()                      # targets pd-target_canvas
p.clear()
saw = p.obj("phasor~", 110)
lop = p.obj("lop~", 1000)          # low-pass
p.connect(saw, 0, lop, 0)
p.to_dac(lop, 0.1)                 # REQUIRED: hip~ 5 -> *~ 0.1 -> clip~ -1 1 -> dac~
p.dsp(True)
if not p.is_safe():
    raise SystemExit(p.safety_violations())
PureDataClient().send_batch(p.messages())
```

**Share an artifact.** Export a standalone `.pd` (openable in any Pd, no socket):
`python3 pd_examples.py sines --freqs 120 254 1201 --export chord.pd`. Record a
WAV of the exact (post-limiter) output with `--wav out.wav --seconds 8` (needs Pd
running with audio on); it exits nonzero if the record controls fail to land.

## [CRITICAL] Audio DSP safety

Output must reach `[dac~]` through the protective master chain:

```
source → [hip~ 5] → [*~ 0.1] → [clip~ -1 1] → [dac~]
```

`PdPatch.to_dac(source, gain)` builds exactly this, and `safety_violations()`
enforces all three guards (check `is_safe()` before sending):

1. **Limiter at the DAC** — `[dac~]` must be fed by `[clip~ -1 1]` so overshoots
   can't hit the DAC as full-scale bursts. Never wire a raw signal to `[dac~]`.
2. **Bounded master gain** — the `[*~]` feeding that `[clip~]` must be `≤ 0.1`.
3. **DC / subsonic block** — a `[hip~ 5]` must sit upstream of `[dac~]`.

Always end a patch with `to_dac(...)` rather than wiring `[dac~]` by hand. If a
request implies danger ("blast it", direct-to-dac), keep the gain `≤ 0.1`.

## Pitfalls

- **Summing/scaling two *signals*? Use the no-argument form.** `[+~]`/`[*~]`/`[-~]`
  with a creation argument (`[+~ 110]`) turn the **right inlet into a control
  (float) inlet** — a signal wired there is rejected (`audio signal outlet
  connected to nonsignal inlet`) when DSP starts. To sum two signals use `[+~]`
  with no arg; keep `[+~ base]` only for a fixed scalar on the left inlet.
- **`osc~` frequency by signal goes in the left inlet (0)**; the right inlet is
  phase. `[vcf~]`: inlet 0 audio, inlet 1 centre freq (signal), inlet 2 Q (float).
- **The linter checks output *safety*, not signal-flow *correctness*.** It will
  not catch a signal wired to a control inlet or a mis-indexed `connect` — watch
  Pd's console for `nonsignal` / `couldn't create` after `pd dsp 1`.

## Verification

1. **Before sending:** `PdPatch.is_safe()` / `safety_violations()` catch a
   missing `[clip~]`, a master gain above `0.1`, or a missing `[hip~]`;
   `build_simple_synth()` is a known-safe reference.
2. **Socket handshake:** `python3 pd_client.py "hello"` prints `pd-net-in: hello`
   in Pd (a plain word falls out `[route]`'s last outlet to `[print pd-net-in]`);
   `Connection refused` means Pd or the stub is not running.
3. **Canvas edits landed:** an intentionally invalid object
   (`pd-target_canvas obj 10 10 nope_xyz`) makes Pd log `nope_xyz … couldn't
   create` — proof the message reached the editor. Valid objects create silently.
4. **Regression tests:** `scripts/run_tests.sh tests/skills/test_pd_patching_skill.py -q`
   (mocked, no Pd needed).

## Security

Communication is localhost-only (`127.0.0.1:9999`); Pd's `[netreceive]` has no
authentication, so keep it on loopback. Dynamic patching can instantiate any Pd
object, including ones that read/write files or run shell (`[shell]`, `[pdlua]`);
only send patches you trust.

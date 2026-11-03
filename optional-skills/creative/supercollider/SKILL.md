---
name: supercollider
description: Build and control SuperCollider synths over OSC.
version: 1.0.0
author: mrev-lab
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [supercollider, scsynth, synthdef, osc, audio, synthesis, dsp, generative-audio, creative-coding, live-coding]
    related_skills: [pd-patching, songwriting-and-ai-music]
---

# SuperCollider Skill

Make and control sound in a running SuperCollider audio server (`scsynth`) by
compiling SynthDefs and sending OSC over UDP, entirely from Python's standard
library — no `sclang`, no `python-osc`, no compiled bridge. It builds synth
graphs, plays and live-controls them, exports `.scsyndef` files, and records WAV.
It does **not** cover the SuperCollider language, patterns, or the IDE.

## When to Use

Use when the user wants to generate or control audio/DSP in SuperCollider: a
synth, an FM bell, a filter sweep, a drone, or a generative melody — described in
words, produced as sound. Skip it for tasks that need the `sclang` language
(patterns, `Pdef`, GUI); this skill drives the bare server directly.

## Prerequisites

- **SuperCollider** installed (provides the `scsynth` server binary):
  - macOS: `brew install --cask supercollider`
  - Linux: `sudo apt install supercollider-server`
- The `terminal` tool, used to boot the server and run the helper scripts below.
- No Python packages — standard library only.

## How to Run

Run every command through the `terminal` tool. First locate the installed skill
(helpers import each other by name, so run them from `scripts/`):

```bash
SKILL="${HERMES_HOME:-$HOME/.hermes}/skills/creative/supercollider"
cd "$SKILL/scripts"
```

Boot a bare server once (leave it running), then confirm it answers:

```bash
"$SKILL/templates/boot_scsynth.sh" &      # listens for OSC on UDP :57110
python3 sc_client.py --status             # -> ugens=0 synths=0 ... sr=44100
```

Play a gallery synth, or an arbitrary chord, in one stdin-free command:

```bash
python3 sc_examples.py wobble_bass --play
python3 sc_examples.py sines --freqs 120 254 1201 --play
python3 sc_client.py --free-all           # stop everything
```

> Pass everything as **arguments**, never via a heredoc — an agent terminal
> often fails to deliver heredoc stdin and the call hangs. To shape a custom
> graph, write a real `.py` file in `scripts/` and run it.

## Quick Reference

Helper scripts (all in `scripts/`, standard library only):

| File | Purpose |
|------|---------|
| `sc_synthdef.py` | `SynthDef` graph builder → SynthDef v2 bytes (`compile`, `to_scsyndef_file`); UGen palette; `to_out` safety chain; safety linter. |
| `sc_client.py` | OSC-over-UDP bridge to `scsynth` (`play`/`set`/`free`/`free_all`/`sync`/`status`/`record`) + CLI. |
| `sc_examples.py` | Gallery of 7 synths + CLI (`--play`, `--export file.scsyndef`, `--freqs`). |
| `templates/boot_scsynth.sh` | Boots `scsynth` on UDP :57110 (finds the macOS app-bundle binary + plugins). |

Server commands (CLI: an OSC address then args):

| Action | Command |
|--------|---------|
| Load a compiled SynthDef | `/d_recv <blob>` (`SCClient.load` / `play`) |
| Start a synth | `/s_new {name} {id} {addAction} {target} [ctrl val …]` |
| Set controls live | `/n_set {id} {ctrl} {val} …` |
| Free all synths (stop) | `--free-all` (`/g_freeAll 0`) |
| Server status | `--status` |

Gallery: `sine`, `sines --freqs …`, `fm_bell`, `wobble_bass`, `supersaw_drone`,
`random_blips`, `resonator`. List them with `python3 sc_examples.py --list`.

## Procedure

The agent turns a request into sound like this:

1. **Ensure the server is up.** `python3 sc_client.py --status`; if it does not
   answer, boot `templates/boot_scsynth.sh`.
2. **Pick the cheapest path.** Canned sound → a gallery name; a tweak → call a
   gallery builder with params, or `/n_set` a running node; novel routing →
   build a `SynthDef`.
3. **Build and check safety.** End every graph with `to_out(...)` and confirm
   `is_safe()` before playing.
4. **Play, then verify it landed.** `--play` calls `/sync` and reports failure
   if the server does not confirm.

Build a custom synth in a `.py` file under `scripts/`:

```python
from sc_synthdef import SynthDef
from sc_client import SCClient

sd = SynthDef("my_synth")
freq = sd.control("freq", 220)          # live-settable parameter
sig  = sd.rlpf(sd.saw(freq), 800, 0.2)  # resonant low-pass on a saw
sd.to_out(sig, 0.15)                    # LeakDC -> *0.15 -> Limiter -> Out.ar(0)
if not sd.is_safe():
    raise SystemExit(sd.safety_violations())
c = SCClient()
node = c.play(sd, node_id=1000)
c.sync()                                # confirm it started
# c.set(1000, freq=110)                 # live control
# c.free_all()                          # stop
```

Signal math reads naturally: `a * 0.2`, `a + b` build `BinaryOpUGen` nodes.

**Share an artifact.** Export a standalone `.scsyndef` (loadable by any
`scsynth`, no Python): `python3 sc_examples.py fm_bell --export bell.scsyndef`.
Record a WAV of the live output with `SCClient.record(path)` /
`record_stop()` (needs the server booted with audio).

## [CRITICAL] Audio safety

Every `Out` must be reached through the protective master chain:

```
source -> LeakDC.ar -> [* <= 0.2] -> Limiter.ar -> Out.ar(0)
```

`SynthDef.to_out(source, gain)` builds exactly this (dual-mono), and
`safety_violations()` enforces it: a `Limiter`/`Clip` must feed `Out`, its
master gain `*` must be a constant `<= 0.2` (`MAX_MASTER_GAIN`), and a `LeakDC`
must sit upstream. Never wire a raw signal straight to `Out`; always end with
`to_out(...)`. If a request implies danger ("blast it"), keep the gain `<= 0.2`.

## Pitfalls

- **UGen names must be exact.** The palette (`sinosc`, `saw`, `pulse`, `rlpf`,
  `lpf`/`hpf`/`bpf`, `resonz`, `combl`, `lfnoise0/1`, `whitenoise`, `dust`,
  `impulse`, `leakdc`, `limiter`, `clip`, `pan2`) plus `ugen(name, rate, inputs)`
  cover the rest. A misspelled UGen loads with no Python error, but `scsynth`
  logs `UGen 'X' not installed` and the def never registers — watch its output.
- **Calc rates matter.** Audio helpers default to `.ar`, LFOs to `.kr`; mixing an
  audio and control signal yields audio (the builder picks the higher rate).
  Scalar-only inputs (`combl` `maxdelay`, `limiter` `dur`) take a number, not a
  live signal.
- **Controls need `sd.control(...)`.** A bare constant cannot be changed with
  `/n_set`; declare a named control to make a parameter live-settable.
- **The linter checks output *safety*, not signal-flow *correctness*.** It will
  not catch a wrong input order or a bad rate — watch the `scsynth` console.

## Verification

1. **Before playing:** `SynthDef.is_safe()` / `safety_violations()` catch a
   missing `Limiter`, a master gain above 0.2, or a missing `LeakDC`;
   `build_simple_synth()` is a known-safe reference.
2. **Server handshake:** `python3 sc_client.py --status` returns a
   `/status.reply` line; no reply means `scsynth` is not running.
3. **Def loaded & node running:** after `--play`, `--status` shows `synths` and
   `ugens` climb (the sine reports `ugens=6 synths=1`); after `--free-all` they
   return to `synths=0`. A def that fails to load leaves `defs` unchanged and
   logs an error in the `scsynth` output.
4. **Regression tests:** `scripts/run_tests.sh tests/skills/test_supercollider_skill.py -q`
   (mocked, no server needed).

## Security

Communication is localhost-only (`127.0.0.1:57110`); `scsynth`'s OSC port has no
authentication, so keep it on loopback (do not boot with `-B 0.0.0.0`). Server
commands and UGens can read and write files (`/b_write`, `DiskOut`); only load
and play defs you trust.

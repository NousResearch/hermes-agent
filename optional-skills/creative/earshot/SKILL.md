---
name: earshot
description: Hear and score audio files locally on CPU.
version: 0.3.0
author: Leon Lin (Leonxlnx), ported to Hermes by Nous Research
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [audio, sound, sfx, sound-design, video, loudness, clap, mcp, cpu]
    category: creative
    related_skills: [songsee, hyperframes, audiocraft-audio-generation]
    upstream: https://github.com/Leonxlnx/earshot (pinned 925ebe5, v0.3.0)
---

# earshot

earshot gives an agent "ears". It reads a sound or a video's audio track and
returns evidence: a plain-language description with calibrated z-scores, how
well a text description fits ("strong match, z +3.7"), the best moments inside
a long recording as trims, real-recording vs code-synthesized, a second-by-
second listening timeline synced to picture cuts, an annotated PNG sheet, and
a scorecard against professional launch films. It also renders sound effects
from a bank of 201 code-defined sounds (`earshot sfx`) and can make new ones in
code, improving them by listening (`earshot forge`). Everything runs locally on
CPU. Audio never leaves the machine, except through the opt-in `llm` command.

The full upstream workflow (five loops, how to read the numbers, how to write
queries, limits) is in `references/upstream-skill.md`, pinned at the commit
above. Read it before your first real task. The notes below override it where
they differ. Its repo-relative paths aren't installed with this skill; read
them at the pinned commit instead:
`docs/SFX.md` → https://github.com/Leonxlnx/earshot/blob/925ebe5/docs/SFX.md,
`skills/real-sound` (recorded-audio pipeline) →
https://github.com/Leonxlnx/earshot/tree/925ebe5/skills/real-sound,
`eval/semantics/RESULTS.md` →
https://github.com/Leonxlnx/earshot/blob/925ebe5/eval/semantics/RESULTS.md.

## When to use it

Use it whenever the task needs a judgement about sound you can't hear. That
covers picking an SFX or music bed from candidates, trimming the clean take,
checking a rendered video's mix and sync (hyperframes, manim and ffmpeg
pipelines), A/B-ing a mix change, or confirming that generated audio matches
its prompt. Quote the number when you report ("strong match z +3.9; -14.2
LUFS"). Never say how audio sounds without a reading behind it.

## Install (once)

The PyPI package named `earshot` is an unrelated project, so a bare
`pip install earshot` installs the wrong thing. Install from the repo at the
pinned commit into its own venv, with CPU-only torch:

```bash
python3 -m venv "$HOME/.local/share/earshot-venv"
"$HOME/.local/share/earshot-venv/bin/pip" install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "earshot[mcp] @ git+https://github.com/Leonxlnx/earshot@925ebe5"
EARSHOT="$HOME/.local/share/earshot-venv/bin/earshot"
"$EARSHOT" doctor
```

Expect about 1.5 minutes and 1.7 GB for the venv (Python 3.10+). Shell
variables don't carry over between terminal calls, so repeat the `EARSHOT=`
line in every call that uses it, or call the full path. If the user already
has `uv`, `uv tool install` with the same requirement string and index works
too.

The first model-using command downloads two pinned Hugging Face models (CLAP
`laion/larger_clap_general` and the AudioSet AST), about 1 GB, into earshot's
own cache. Set `EARSHOT_HOME` to choose where that cache lives. `HF_TOKEN` is
not needed; expect an unauthenticated-Hub warning on stderr. `measure`, `sfx`
and `kit` load no models and answer in a second or two.

**Shared machine.** Set `EARSHOT_THREADS=2` or `3` so one call doesn't take
every core.

## CLI or one script, not the MCP server

The upstream text suggests `earshot mcp` for repeated calls. Under Hermes that
server exposes 16 tools, about 6.4k tokens of schema, in every turn of every
session it's enabled for. For a handful of calls use the CLI (each call
reloads the models in about 20-40 s). For more, write one script with
`from earshot import api` and run it once with
`"$HOME/.local/share/earshot-venv/bin/python" script.py`; after the first load,
repeat calls take well under a second. Add the MCP server only when a long
audio session justifies it, and only with the user's go-ahead, since it
changes their config:

```bash
hermes mcp add earshot --command "$HOME/.local/share/earshot-venv/bin/earshot" --args mcp
```

## Working in Hermes

- **Paths.** Pass absolute paths. Any ffmpeg-readable file works, including
  the audio track of an mp4.
- **Looking at sheets.** `earshot sheet FILE -o out.png` draws frames,
  labelled events, spectrogram and loudness. Read it with `vision_analyze`
  and ask specific questions ("does the hit at 0:01.2 land on the cut?").
- **Long jobs.** `critique` and `timeline` on a full film, and `forge` loops,
  can take minutes. Run them with `terminal(background=true, notify=true)`.
- **Timelines are dense.** One whoosh can show up as several rows, and faint
  `HIT?` events can appear. Read the sections and sync lines first.
- **The optional `llm` command** sends audio to Gemini only with an explicit
  `--yes-upload` and the user's own `GEMINI_API_KEY`. Don't run it without the
  user's consent.

## Verification

```bash
EARSHOT="$HOME/.local/share/earshot-venv/bin/earshot"
"$EARSHOT" sfx render click/double -o "$PWD/click.wav"   # no models
"$EARSHOT" measure "$PWD/click.wav"                      # LUFS / true peak / attack
"$EARSHOT" match "$PWD/click.wav" "a snare hit" --alts "thunder" "a dog barking"
```

The first two print a one-line summary. `match` loads the models (the first
run downloads them) and should print `strong` with a z around +3.4: the bank's
`click/double` reads closest to a snare hit. A z below about 2 is a `no match`
verdict about the sound, not an install failure.

## Updating the pin

Upstream ships several releases a week. To move the pin, replace
`references/upstream-skill.md` with `skills/earshot/SKILL.md` from the new
commit, update the commit in the frontmatter and the install line, then rerun
Verification.

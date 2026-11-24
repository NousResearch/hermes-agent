---
name: cinematic-film
description: Narrated documentary-style films from code, upstream-kept.
version: 1.0.0
author: Moh4696
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [video, documentary, explainer, narration, motion-graphics, captions, skia, ffmpeg]
    category: creative
    related_skills: [manim-video, hyperframes, ai-presenter-video, brag]
    upstream:
      repo: Moh4696/cinematic-skill
      path: skills/cinematic-film
---

# Cinematic film (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [Moh4696/cinematic-skill](https://github.com/Moh4696/cinematic-skill): the
> project ships a self-contained skill directory (`skills/cinematic-film/`)
> with the Skia film engine, the narration/SFX/caption scripts, two runnable
> example projects and the references. `hermes skills install
> official/creative/cinematic-film` pulls the current tree live from that repo
> (quarantined and scanned like any hub install); this directory holds only the
> catalog metadata, so the vendored copy can never go stale.

The skill makes narrated "one camera over one evidence board" films entirely
from code: photos, documents, cards, maps with routes and big numbers laid out
on one world, with every stamp, highlight and number timed to the narrator's
word, then synthesised sound effects, an optional score and burned-in captions
(four styles plus an SRT). It suits documentaries and history, product and
founder stories, science explainers and true crime. Renders are deterministic
and re-time themselves when the voice changes.

## Prerequisites

- Python 3.9+ with `skia-python numpy pillow` (scipy optional); a venv works.
- `ffmpeg` and `ffprobe` on `PATH` (libx264, loudnorm).
- A voice: macOS `say`, `espeak-ng`, or any command-line TTS through
  `tts_say.py --cmd '<tts> ... {txt} ... {out}'` (it must write a WAV).
- Network only for the optional helpers: Wikimedia Commons (`fetch_commons.py`,
  refuses non-free files and writes `credits.md`), Google Fonts on Linux and
  Windows (`get_fonts.py`), and jsdelivr map data (`make_map.py`).

After install, `manim-video` stays the pick for maths/algorithm animation and
`hyperframes`/`brag` for HTML-composited launch videos; prefer this skill for a
voice-led documentary or explainer built from stills, documents and maps.

Full documentation: https://github.com/Moh4696/cinematic-skill#readme

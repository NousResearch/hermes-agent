---
name: wake-word-training
description: "Train a custom on-device wake word from synthetic speech."
version: 1.0.0
author: Brooklyn <brooklyn> & Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Voice, Wake Word, openWakeWord, On-Device, Training]
    category: productivity
    related_skills: []
---

# Wake Word Training Skill

Trains a custom openWakeWord model so Hermes answers to a phrase other than the
bundled "hey hermes" — e.g. "hey morgane" or "hey <profile>". Detection stays
100% on-device; only the optional voice synthesis (OpenAI TTS) touches the
network, and only during training. This skill drives openWakeWord's own
training pipeline through the `terminal` tool — it does not reimplement it.

Custom models are trained on **synthetic speech**: you never record yourself.
A TTS engine speaks the phrase a few thousand times across many voices, the
clips are augmented with noise/room impulses, and a small classifier is trained
on top of openWakeWord's frozen feature extractor. The output is a sub-100 KB
`.onnx` the existing wake runtime loads with zero code changes.

## When to Use

- The user wants Hermes to wake on a different phrase ("hey morgane", a
  nickname, "hey <profile>").
- The user asks to "make/train my own wake word".

Do **not** use this for the default "hey hermes" — that model already ships.
For a phrase you already have an `.onnx` for, skip training and jump to
**Wire It Up**.

## Prerequisites

- **Time & hardware.** Fast on Linux/GPU (~1 hr) or a free Colab GPU. On macOS
  it is single-threaded and slow (~3–5 hr) — see `references/platforms.md`.
  Prefer Colab if the user just wants the file quickly.
- **Disk.** Several GB for augmentation/negative datasets (room impulses,
  AudioSet/FMA noise, LibriSpeech negatives). Use a scratch dir, not the repo.
- **Voices.** `OPENAI_API_KEY` for high-quality OpenAI TTS positives (~$0.04 a
  run); otherwise the offline Piper generator openWakeWord ships with.
- **A scratch training venv**, created ad hoc via `terminal` — never install the
  heavy training stack into Hermes's own environment.

## How to Run

Confirm the phrase and where the model belongs, then train, then wire it up.
Resolve the profile-aware model directory once and reuse it:

```bash
python -c "from hermes_constants import get_hermes_home; print(get_hermes_home() / 'wakewords')"
```

Default the phrase to `hey <profile>` (the active profile's name; `hey hermes`
is the default profile) unless the user gives one. The model file is
`<phrase-slug>.onnx` (lowercase, spaces → underscores), e.g. `hey_morgane.onnx`.

## Quick Reference

| Step | Command / action |
| --- | --- |
| Model dir | `get_hermes_home() / 'wakewords'` (profile-aware) |
| Positives (OpenAI) | `scripts/generate_positives.py --phrase "hey morgane" --out-dir <dir>` |
| Training config | `scripts/make_training_config.py --phrase "hey morgane" --out <cfg.yml>` |
| Train | openWakeWord's `train.py --training_config <cfg.yml>` (in the scratch venv) |
| Wire up | set `wake_word.openwakeword.model` + `phrase`, then restart the listener |

## Procedure

1. **Agree on the phrase and paths.** Slugify the phrase for the filename.
   Resolve the model dir with the one-liner above and `mkdir -p` it.

2. **Make a scratch training dir + venv** somewhere with room (not the repo):

```bash
mkdir -p /tmp/oww-train && cd /tmp/oww-train
python -m venv .venv && . .venv/bin/activate
pip install openwakeword
```

3. **Generate positives.**
   - *OpenAI TTS (default when `OPENAI_API_KEY` is set):* run
     `generate_positives.py` — it speaks the phrase across many voices with
     small text/pacing variations and writes 16 kHz mono WAVs.
   - *Offline:* let openWakeWord's own Piper generator produce them (the
     training config's `target_phrase` drives it). See `references/platforms.md`.

4. **Fetch augmentation + negative data.** Follow openWakeWord's automatic
   training notebook (room impulses, AudioSet/FMA noise, LibriSpeech/precomputed
   negatives). `references/platforms.md` has the exact dataset sources and the
   Colab shortcut that downloads them for you.

5. **Write the training config** with `make_training_config.py` (point it at the
   positives/negatives/output dirs), then run openWakeWord's `train.py` against
   it. Cross-check the emitted YAML against openWakeWord's `custom_model.yml` for
   your installed version.

6. **Place the model.** Copy the exported `<slug>.onnx` into the resolved
   `wakewords` dir.

7. **Wire It Up.** In the active profile's `~/.hermes/config.yaml`, set:
   - `wake_word.openwakeword.model` → the absolute `.onnx` path
   - `wake_word.phrase` → the human label ("hey morgane")
   - `wake_word.enabled` → `true`
   Then restart the listener so it reloads: `/wake` off then on, or restart the
   app/gateway. (Wake config is not part of the prompt, so no cache concern.)

## Pitfalls

- **Don't install the training stack into Hermes's venv.** It's large and
  macOS-flaky; keep it in the throwaway scratch venv.
- **openWakeWord needs its base feature models** (melspectrogram + embedding)
  for any model — the trainer downloads them; don't delete them.
- **macOS bus errors** from openWakeWord's threadpool + mmap: train
  single-threaded, or use Colab. See `references/platforms.md`.
- **Config `model` must be an absolute path** (or a bundled/built-in name). A
  bare filename won't resolve.
- **Sensitivity.** If the new phrase over/under-triggers, tune
  `wake_word.sensitivity` (0–1, higher = stricter) before retraining.

## Verification

- The `.onnx` exists in the profile's `wakewords` dir and is > 50 KB.
- Smoke-test detection before enabling it live:

```bash
python - <<'PY'
import wave, numpy as np
from openwakeword.model import Model
m = Model(wakeword_models=["/abs/path/hey_morgane.onnx"], inference_framework="onnx")
a = np.frombuffer(wave.open("a_clip_of_the_phrase.wav").readframes(1<<20), np.int16)
print("best:", max(max(m.predict(a[i:i+1280]).values()) for i in range(0, len(a)-1280, 1280)))
PY
```
  A clear utterance should score well above `wake_word.sensitivity`.
- After wiring up, `/wake` on and say the phrase — Hermes opens a fresh session.

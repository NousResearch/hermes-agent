# Platforms, Datasets & the Fast Path

openWakeWord's automatic trainer is heavy (torch + a synthetic-speech +
augmentation stack) and its speed depends entirely on where it runs. Pick the
environment first, then follow the data steps.

## Where to train

| Environment | Time | Notes |
| --- | --- | --- |
| **Colab (free GPU)** | ~1 hr | Easiest. Recommended when the user just wants the `.onnx` quickly. |
| Linux + CUDA GPU | ~1 hr | Full local control. |
| Linux CPU | ~2 hr | Fine, just slower. |
| **macOS** | ~3–5 hr | Single-threaded only — see the bus-error note below. |

### Colab shortcut (recommended)

openWakeWord ships an automatic-training notebook that installs the stack,
downloads every dataset below, generates positives, trains, and exports the
model — in one run on a free GPU:
<https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb>

Set the target phrase (and, if using the OpenAI-TTS positives from this skill,
upload the generated WAVs and point the config's positives dir at them), run it,
download the `.onnx`, then follow **Wire It Up** in `SKILL.md`.

### macOS caveat

openWakeWord's data pipeline uses a threadpool over memory-mapped files that
triggers bus errors on macOS. Train **single-threaded** (set the trainer's
worker/`n_jobs` to 1) or, better, use Colab. Expect 3–5 hours locally.

## Datasets the trainer needs

Four kinds of data (openWakeWord's notebook pulls them from HuggingFace):

1. **Room impulse responses** — `davidscripka/MIT_environmental_impulse_responses`
   (reverb augmentation). → `./mit_rirs`
2. **Background noise** — a slice of `agkphysics/AudioSet` (16 kHz). → `./audioset_16k`
3. **False-positive / music negatives** — `rudraml/fma` (start with ~1 hr).
   → `./fma`
4. **Precomputed openWakeWord negative features** — from the openWakeWord
   releases; used as generic negatives + validation for early stopping.

Base feature models (`melspectrogram`, `embedding`) are fetched automatically by
openWakeWord — don't delete them. Budget several GB of scratch space; keep a
local copy to reuse across runs.

## Positives: OpenAI TTS vs Piper

- **OpenAI TTS** (`generate_positives.py`, default when `OPENAI_API_KEY` is set):
  10 voices, small text/pacing variations, ~$0.04 per run, higher quality. Point
  the training config's positives dir at the generated WAVs.
- **Piper** (offline): openWakeWord's built-in generator synthesizes positives
  from the config's `target_phrase`. Fully local, no key. Note: Piper's sample
  generator is best-supported on Linux.

More positives → better accuracy, with smooth diminishing returns. A few
thousand is the practical floor; the notebook defaults to 5,000.

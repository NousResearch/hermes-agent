---
name: whisper
description: >
  Local speech-to-text transcription using Whisper models. Cross-platform:
  uses mlx-whisper on macOS Apple Silicon, faster-whisper on Linux/Windows.
  Supports 99 languages, translation to English, and word timestamps.
  Includes a ready-to-use wrapper script managed with uv.
version: 2.0.0
author: Nous Research
license: MIT
prerequisites:
  commands: [ffmpeg]
metadata:
  hermes:
    tags: [Whisper, Speech Recognition, ASR, Multilingual, Speech-To-Text, Transcription, Translation, Audio Processing]
    requires_toolsets: [terminal]
---

# Whisper — Cross-Platform Local Transcription

Local speech recognition using Whisper models. The included wrapper script
auto-detects the platform and picks the optimal backend.

## When to use

- Speech-to-text transcription (any of 99 languages)
- Podcast, video, or meeting transcription
- Translating speech to English text
- Noisy audio transcription
- Batch audio file processing

## Architecture

| Platform | Backend | Acceleration |
|----------|---------|--------------|
| macOS (Apple Silicon) | `mlx-whisper` | Metal / ANE |
| Linux / Windows (NVIDIA GPU) | `faster-whisper` | CUDA |
| Linux / Windows (CPU only) | `faster-whisper` | CPU (AVX2) |

Both backends use the same Whisper model weights and produce equivalent
transcription quality.

## Quick reference

```bash
# Basic transcription
~/whisper/bin/transcribe-audio recording.ogg

# Specify language (faster + more accurate)
~/whisper/bin/transcribe-audio --language de interview.mp3

# JSON output with word-level timestamps
~/whisper/bin/transcribe-audio --json --word-timestamps lecture.wav

# Use a larger model for higher accuracy
~/whisper/bin/transcribe-audio --model medium podcast.mp3

# Translate non-English speech to English text
~/whisper/bin/transcribe-audio --task translate spanish_audio.mp3
```

## Setup procedure

### Prerequisites

- **ffmpeg** on `PATH` (`brew install ffmpeg` / `apt install ffmpeg` / `choco install ffmpeg`)
- **uv** — Python package manager (see https://docs.astral.sh/uv/getting-started/installation/)

### Installation

1. Create the working directory:

```text
~/whisper/
  .venv/               # managed by uv
  bin/
    transcribe-audio   # wrapper script (this skill's scripts/transcribe-audio)
  cache/huggingface/   # downloaded model weights
  tmp/                 # ephemeral wav files during transcription
```

2. Create a virtual environment and install the backend:

```bash
mkdir -p ~/whisper/bin ~/whisper/cache/huggingface ~/whisper/tmp
cd ~/whisper && uv venv
```

macOS Apple Silicon:
```bash
uv pip install mlx-whisper
```

Linux / Windows:
```bash
uv pip install faster-whisper
```

3. Copy `scripts/transcribe-audio` from this skill into `~/whisper/bin/` and
   make it executable (`chmod +x ~/whisper/bin/transcribe-audio`).
   Update the shebang to point at the venv Python
   (`#!/path/to/whisper/.venv/bin/python`).

4. On first use the selected model is downloaded automatically into
   `~/whisper/cache/huggingface/`.

## CLI options

| Flag | Description |
|------|-------------|
| `--model ALIAS` | tiny, base, **small** (default), medium, turbo |
| `--language CODE` | ISO 639-1 code (`en`, `de`, `ja`, …). Auto-detects if omitted |
| `--task` | `transcribe` (default) or `translate` (to English) |
| `--json` | Structured JSON output instead of plain text |
| `--word-timestamps` | Per-word timing (in JSON mode) |
| `--initial-prompt` | Domain hint to improve accuracy |

## Model sizes

| Model | Params | Multilingual | Relative speed | VRAM |
|-------|--------|:------------:|:--------------:|------|
| tiny | 39 M | ✓ | ~32× | ~1 GB |
| base | 74 M | ✓ | ~16× | ~1 GB |
| small | 244 M | ✓ | ~6× | ~2 GB |
| medium | 769 M | ✓ | ~2× | ~5 GB |
| large | 1550 M | ✓ | 1× | ~10 GB |
| turbo | 809 M | ✓ | ~8× | ~6 GB |

Use **turbo** for the best speed / quality trade-off on capable hardware.

## Best practices

1. **Specify language** when known — faster and more accurate than auto-detect.
2. **Use turbo** for long recordings when speed matters.
3. **Split long audio** — accuracy degrades beyond ~30 minutes.
4. **Add an initial prompt** for domain-specific terms or proper nouns.
5. **Pre-convert to WAV** if you hit format issues (ffmpeg handles most formats).

## Pitfalls

- **Hallucinations on silence**: Whisper may invent text for silent segments.
  Trim silence before transcription if this occurs.
- **No speaker diarization**: Whisper does not identify speakers. Use
  whisperX or pyannote-audio if diarization is needed.
- **Long audio drift**: Accuracy degrades on files longer than ~30 min.
  Split into chunks and transcribe individually.
- **Model download size**: First run downloads the model (small ≈ 500 MB,
  turbo ≈ 1.5 GB). Ensure sufficient disk space.

## Verification

After setup, verify with a quick test:

```bash
# Record or download a short audio clip, then:
~/whisper/bin/transcribe-audio --language en test.wav
# Should print the transcript to stdout.
```

## Resources

- **Whisper**: https://github.com/openai/whisper
- **faster-whisper**: https://github.com/SYSTRAN/faster-whisper
- **mlx-whisper**: https://github.com/ml-explore/mlx-examples/tree/main/whisper
- **Paper**: https://arxiv.org/abs/2212.04356

---
name: local-tts-optimization
description: Use when tuning local TTS latency/stability (MLX Qwen3-TTS).
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [TTS, MLX, Apple Silicon, voice, performance]
---

# Local TTS Optimization (Apple Silicon)

Deploy and tune a **local** text-to-speech voice-clone engine (e.g. Qwen3-TTS-12Hz-1.7B) on macOS Apple Silicon. Distilled from a production deployment that took a local TTS service from 377s to 100s for 822-char Chinese text without changing the model or sacrificing user-accepted quality.

## When to Use

- User runs a local TTS / voice-clone engine (via a `tts.providers.<name>.type: command` provider) and hits timeouts, slowness, or broken Chinese output
- User asks why local TTS is slower than cloud TTS, or how to speed it up without replacing the model
- Tuning a local `StreamingTTSProvider` / command provider on Apple Silicon

## Key Numbers (validated, M5 Max, Qwen3-TTS-12Hz-1.7B)

| Engine | 822-char | short text | model load | memory |
|---|---|---|---|---|
| PyTorch CPU single process | 377s (RTF 2.17) | ~20s | 7-11s | 9GB |
| PyTorch CPU 4-process parallel | 230.7s (RTF 1.25) | 11.5s | ~11s | 9GB×4 |
| MLX single process | 266.5s (RTF 1.88) | 8-11s | 1-2s | 2-3GB |
| **MLX 4-process pool** | **100.2s (RTF 0.53)** | 7.5-8s | 1-2s | 4×2-3GB |

- **RTF < 1.0 means generation is faster than playback** — the UX goal.
- MLX single process is NOT faster than tuned PyTorch parallelism. The win is **MLX GPU inference × controlled 4-process parallelism**.
- 18-process parallel dies (memory 36GB+ and GPU init storm) — **always cap concurrency with a process pool**.

## Architecture (production-tested)

```
dalu_tts.py --engine mlx          # command provider entry (Hermes: tts.providers.<name>.command)
  ├─ split text (160-char cap, CJK punctuation preferred break)
  ├─ round-robin into 4 worker groups
  ├─ each group = separate mlx-tts-venv subprocess (model load 1-2s)
  │     └─ load MLX model → generate chunk → write temp wav
  ├─ main waits all (retry failed blocks once)
  └─ strip per-chunk WAV headers → concat PCM → 150ms silence gaps → unified RIFF header
```

Why subprocess workers: the Hermes runtime venv must not carry MLX's `transformers 5.x` (conflicts with `qwen_tts`'s pinned `4.57.3`). Keep MLX in its own venv and bridge via subprocess — same pattern as command providers.

## Stability Checklist (all validated)

1. **Timeout** — command provider `timeout: 600` in config (default 120s kills 822-char local synthesis)
2. **Backlog** — if using a daemon: `request_queue_size=64` (default 5 refuses connections during long generation)
3. **Liveness probe** — use `socket.bind()` (port in use = alive), NOT TCP connect (backlog-full gives false negatives → spawn storms)
4. **Spawn race** — flock file lock around lazy-spawn
5. **Failover** — retry POST on other daemon ports; retry whole synthesis 3×
6. **Proxy** — `no_proxy=127.0.0.1,localhost` so local HTTP never goes through Surge/Clash
7. **Concurrency cap** — process pool (4) for MLX; never unbounded processes

## Chinese Chunking (CJK punctuation)

Chinese has no spaces, so space-based sentence splitting hard-cuts mid-word (`发现三个隐|患点`). Split on CJK punctuation `。！？；，、：”’` as hard breaks, then merge short sentences up to the cap. Latin punctuation still breaks only after whitespace (`3.14`/`Dr.` untouched). Hermes core fix: `_split_text_for_tts` in `tools/tts_tool.py` (see PR #84622).

## MLX Setup (venv isolation is mandatory)

```bash
python3 -m venv ~/work/mlx-tts-venv
# git+mlx-audio needs proxy for GitHub; deps via TUNA mirror in CN:
env -u PYTHONPATH https_proxy=http://127.0.0.1:6152 http_proxy=http://127.0.0.1:6152 \
  ~/work/mlx-tts-venv/bin/pip install --timeout 120 -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git@9349644ccbd62eb10900852228f7b952c566def3"
```

Model: `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` (hf-mirror in CN; `snapshot_download` may miss `config.json` — patch `"quantization": {"group_size": 64, "bits": 8}` onto the PyTorch config; copy `merges.txt/vocab.json/tokenizer_config.json/speech_tokenizer/*` from the PyTorch model dir).

```python
from mlx_audio.tts.utils import load_model
model = load_model("~/.hermes/models/Qwen3-TTS-12Hz-1.7B-Base-8bit-mlx")
for result in model.generate(text, ref_audio="ref.wav", ref_text="transcript",
                             lang_code="Chinese"):   # ref_audio: path or mx.array, NOT numpy
    ...  # result.audio is mx.array PCM at 24kHz
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Received 500 parameters not in model` | config.json missing `quantization` | patch `{"group_size": 64, "bits": 8}` |
| `audio must be str or mx.array` | ref_audio passed numpy | pass path string or mx.array |
| `check_model_inputs() missing func` | PYTHONPATH shadows venv (transformers 5.x) | run with `env -u PYTHONPATH` |
| 18-process parallel hangs | unbounded concurrency + GPU init storm | 4-process pool |
| output ZCR > 0.3 | 0.6B Lite model + 8bit = noise | use 1.7B 8bit |
| Chinese breaks mid-word | space-based sentence split | use CJK punctuation breaks |
| command provider timeout | 120s default < long local synth | config `timeout: 600` |

## Quality Gate

- ZCR 0.06-0.10 healthy; 8-segment check for tail degradation
- 8bit MLX spectrum skews low-freq (~78% 100-500Hz vs PyTorch 54%) but users accept by listening — **listening is the final gate, spectra are advisory**
- 0.6B Lite: faster only marginally (221s vs 266s) and ZCR 0.52 (broken) — reject

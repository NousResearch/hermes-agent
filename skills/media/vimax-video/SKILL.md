---
name: vimax-video
description: "Agentic video generation pipeline (HKUDS ViMax). Use when the user wants Paimon to turn an idea or script into an end-to-end generated video (storyboard → shots → audio → mux), GPU-backed."
version: 1.0.0
platforms: [linux]
metadata:
  hermes:
    tags: [video, generation, agentic, ml, gpu, vimax]
    related_skills: [ffmpeg-media, hyperframes-render]
---

# ViMax — Agentic Video Generation

Cloned at `~/paimon/tools/ViMax`. Heavy stack: PyTorch CUDA 12.8, ffmpeg,
opencv. Bootstrapped via `uv sync` in its own `.venv`.

## Bring up the venv (first time only)
```bash
cd ~/paimon/tools/ViMax
~/snap/code/237/.local/bin/uv sync          # downloads torch CUDA wheels (~5 GB)
```

## Run a pipeline
```bash
cd ~/paimon/tools/ViMax
.venv/bin/python main_idea2video.py --idea "A 30-sec teaser for a coffee shop"
.venv/bin/python main_script2video.py --script ./script.txt
```
Outputs land under `outputs/` (or wherever `configs/` points).

## When to invoke
- User asks for a generated video from text/idea/script.
- Pair with `ffmpeg-media` to post-process and `hyperframes-render` for HTML
  overlays (titles, lower-thirds).

## Hardware
- This box has an RTX 2080 (8 GB) — fine for short clips at lower res.
- For >720p 30s+ clips, consider running on a beefier GPU or splitting shots.

## Configuration
- LLM provider for the script/storyboard agents: edit
  `~/paimon/tools/ViMax/configs/`.
- Ensure API keys are actually set (non-empty) for the configured providers before running.
  This environment does not guarantee `~/paimon/home/.env` has populated keys by default.
- Voice/TTS modules pull additional weights on first run; allow ~10–15 min.

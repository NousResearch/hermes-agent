# Creative Media Stack

This document records the advanced creative/video/image capability set imported from the Paimon physical design agent into the canonical Hermes/SitioUno repos.

## Purpose

Enable Zeus and inherited agents to produce higher-quality UGC reels, avatar/lipsync workflows, generated video, programmatic motion graphics, deterministic text overlays, background cleanup, media QA, and video intelligence without relying on ad-hoc local-only notes.

The repo ships the reusable skills and a bootstrap script. Large third-party checkouts and binaries are intentionally **not** committed; agents install or mount them per machine.

## Imported skills

### Video generation and UGC

- `media/app-feature-ugc-reel` — end-to-end UGC reel planning and execution for app/agent/product demos.
- `media/vimax-video` — ViMax agentic video generation pipeline: storyboard, shots, audio, mux.
- `creative/vimax` — operational ViMax runner guidance and provider config expectations.
- `media/hyperframes-render` — HTML/CSS/JS motion graphics to MP4/WebM via HyperFrames.
- `media/reel-text-overlay` — deterministic Pillow + FFmpeg text overlays for reels.

### Editing, compositing, and QA

- `media/ffmpeg-media` — FFmpeg recipes for trim, concat, mux, overlays, GIF/HLS/audio handling, QA pitfalls.
- `creative/image-background-cleanup` — cleanup/recreate background workflows and quality gates.
- `media/video-intel-pipeline` — download/transcribe/analyze video with yt-dlp, ffmpeg, faster-whisper.
- `media/creative-media-api-validation` — validation protocol for image/video/avatar/lipsync APIs.

### Hosted media providers

- `media/higgsfield-cli` — Higgsfield CLI operation and routing boundaries.
- `media/mmx-cli` — MiniMax CLI operation.
- `higgsfield/higgsfield-generate` — general Higgsfield generation.
- `higgsfield/higgsfield-soul-id` — Soul Character training.
- `higgsfield/higgsfield-product-photoshoot` — product/brand photoshoots.
- `higgsfield/higgsfield-marketplace-cards` — marketplace product image cards.

## Tools and libraries from Paimon audit

Observed on Paimon (`openclaw-miami`, physical media/design agent):

- Static FFmpeg/FFprobe build on PATH via local tools directory.
- ImageMagick (`magick`, `convert`).
- Higgsfield CLI.
- HyperFrames and HyperFrames CLI.
- HKUDS ViMax checkout.
- Python media packages in Hermes venv: `moviepy`, `opencv-python/cv2`, `Pillow`, `imageio`, `imageio-ffmpeg`, `numpy`, `torch`, `faster_whisper`, `onnxruntime`, `soundfile`, `openai-whisper`, `elevenlabs`.

## Bootstrap

Run:

```bash
scripts/bootstrap_creative_media_stack.sh
```

By default it creates/uses the isolated venv `~/.hermes/tool-venvs/creative-media`, installs lightweight Python/CLI dependencies there, and verifies system tools. Heavy tool checkouts are opt-in:

```bash
INSTALL_HEAVY_CREATIVE_TOOLS=1 scripts/bootstrap_creative_media_stack.sh
```

Heavy installs should be reviewed per host because they may clone large repositories or require GPU/CUDA/provider credentials.

## Secret hygiene

Provider credentials must come from the agent's configured secret manager/runtime environment, not hardcoded paths or copied dotfiles. Relevant variables include:

- `HF_CREDENTIALS` or `HF_API_KEY` + `HF_API_SECRET` for Higgsfield programmatic access.
- `MINIMAX_API_KEY` for MiniMax CLI.
- ViMax provider credentials such as `VIMAX_IMAGE_API_KEY`, `VIMAX_VIDEO_API_KEY`, and model/provider-specific env vars.

Do not commit OAuth caches, provider tokens, generated media caches, or local node_modules/venv directories.

## Quality gates for produced media

Before delivering a video/image artifact:

1. Verify file exists and has non-zero size.
2. Probe duration/resolution with `ffprobe` for video.
3. Inspect at least first/middle/last frames or use `vision_analyze` for visual QA.
4. Verify text overlays are within safe margins and readable on 9:16 mobile.
5. Confirm audio is present and not clipped when voice/music is expected.
6. Prefer deterministic overlays for final copy; do not rely on AI video models to render exact text.
7. Keep final UGC/reel scripts under the platform time constraint and check actual duration.

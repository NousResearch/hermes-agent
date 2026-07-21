# Video Generation Reference

## Models

### Veo (Google)
- **veo/3.1** — Highest quality, frame interpolation for smoother output
- **veo/3** — Previous gen, still excellent
- **veo/2** — Budget-friendly
- **veo/fast** — Speed-optimized variants

Supports: prompt, duration, aspect_ratio, resolution.

### Seedance 2.0 (ByteDance)
- **seedance/2.0** — Multi-modal input (text + up to 9 images + 3 videos + 3 audios)
- **seedance/2.0-studio** — Portrait consistency variant
- **seedance/2.0-i2v** — Image-to-video
- **seedance/2.0-r2v** — Reference-to-video

Key fields:
- `generate_audio: true` — Synchronized audio (always enable unless silent wanted)
- `image` — Source image for i2v
- `reference_images` — Style/subject references
- `video` — Source video for extension/editing
- Duration: 4-15 seconds, up to 1080p

### HappyHorse 1.0 (Alibaba)
- **happyhorse/1.0** — Physically realistic motion, up to 15s at 1080p
- **happyhorse/1.0-i2v** — Image-to-video variant

Strong with natural scenes, physics, water, cloth.

### Other Models
- **wan/2.5-i2v** — Animate any still image
- **wan/2.5** — Text-to-video
- **grok-video** — xAI, configurable duration, fast
- **p-video** — Cheapest and fastest, with optional audio

## Enhancement
- **topaz/video-upscaler** — Upscale resolution and frame rate
- **hunyuanvideo/foley** — Add sound effects to silent video
- **media-merger** — Merge multiple clips with transitions

## Tips

- For highest quality, use Veo 3.1. For audio-synced output, use Seedance 2.0 with `generate_audio: true`.
- Image-to-video works best when source image matches the target aspect ratio.
- For long videos, generate 5-10s clips and merge, or use Seedance video extension.
- All video tasks are async — use `belt task get <id>` to poll.

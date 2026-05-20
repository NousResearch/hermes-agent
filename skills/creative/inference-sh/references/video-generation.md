# Video Generation Reference

## Model Details

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

Key features:
- `generate_audio: true` — Synchronized audio (always enable unless silent wanted)
- `image` — Source image for i2v
- `reference_images` — Style/subject references
- `video` — Source video for extension/editing
- Duration: 4-15 seconds, up to 1080p

### HappyHorse 1.0 (Alibaba)
- **happyhorse/1.0** — Physically realistic motion
- **happyhorse/1.0-i2v** — Image-to-video variant

Supports up to 15 seconds at 1080p. Strong with natural scenes, physics, water, cloth.

### Wan 2.5
- **wan/2.5-i2v** — Animate any still image
- **wan/2.5** — Text-to-video

### Other Models
- **grok-video** — xAI, configurable duration, fast
- **p-video** — Cheapest and fastest, with optional audio
- **hunyuanvideo/foley** — Add sound effects to silent video (post-processing)

## Video Enhancement

- **topaz/video-upscaler** — Upscale resolution and frame rate
- **media-merger** — Merge multiple clips with transitions

## Common Input Fields

```json
{
  "prompt": "description of the video",
  "duration": 5,
  "aspect_ratio": "16:9",
  "resolution": "1080p",
  "image": "source.png",
  "generate_audio": true
}
```

## Tips

- For highest quality, use Veo 3.1. For audio-synced output, use Seedance 2.0 with `generate_audio: true`.
- Image-to-video works best when the source image matches the target aspect ratio.
- Long videos: generate 5-10s clips and merge with media-merger, or use Seedance video extension.
- For product demos, generate the product shot first (image gen) then animate it (i2v).
- All video tasks are async — use `belt task get <id>` to poll for completion.

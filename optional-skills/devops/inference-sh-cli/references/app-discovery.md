# Discovering Apps

## List All Apps

```bash
infsh app list
```

## Pagination

```bash
infsh app list --page 2
```

## Filter by Category

```bash
infsh app list --category image
infsh app list --category video
infsh app list --category audio
infsh app list --category text
infsh app list --category other
```

## Search

```bash
infsh app search "flux"
infsh app search "video generation"
infsh app search "tts" -l
infsh app search "image" --category image
```

Or use the flag form:

```bash
infsh app list --search "flux"
infsh app list --search "video generation"
infsh app list --search "tts"
```

## Featured Apps

```bash
infsh app list --featured
```

## Newest First

```bash
infsh app list --new
```

## Detailed View

```bash
infsh app list -l
```

Shows table with app name, category, description, and featured status.

## Save to File

```bash
infsh app list --save apps.json
```

## Your Apps

List apps you've deployed:

```bash
infsh app my
infsh app my -l  # detailed
```

## Get App Details

```bash
infsh app get falai/flux-dev-lora
infsh app get falai/flux-dev-lora --json
```

Shows full app info including input/output schema.

## Popular Apps by Category

### Image Generation
- `falai/flux-dev-lora` - FLUX.2 Dev (high quality)
- `falai/flux-2-klein-lora` - FLUX.2 Klein (fastest)
- `infsh/sdxl` - Stable Diffusion XL
- `google/gemini-3-pro-image-preview` - Gemini 3 Pro
- `xai/grok-imagine-image` - Grok image generation

### Video Generation
- `google/veo-3-1-fast` - Veo 3.1 Fast
- `google/veo-3` - Veo 3
- `bytedance/seedance-1-5-pro` - Seedance 1.5 Pro
- `infsh/ltx-video-2` - LTX Video 2 (with audio)
- `bytedance/omnihuman-1-5` - OmniHuman avatar

### Audio
- `infsh/dia-tts` - Conversational TTS
- `infsh/kokoro-tts` - Kokoro TTS
- `infsh/fast-whisper-large-v3` - Fast transcription
- `infsh/diffrythm` - Music generation

## Documentation

- [Browsing the Grid](https://inference.sh/docs/apps/browsing-grid) - Visual app browsing
- [Apps Overview](https://inference.sh/docs/apps/overview) - Understanding apps
- [Running Apps](https://inference.sh/docs/apps/running) - How to run apps

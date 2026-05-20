# Discovering Apps

## Browse the App Store

```bash
belt app store
```

## Filter by Category

```bash
belt app store --category image
belt app store --category video
belt app store --category audio
belt app store --category text
```

## Search

```bash
belt app store search "flux"
belt app store search "video generation"
belt app store search "tts"
```

## Get App Details

```bash
belt app get flux/dev-lora
```

Shows full app info including input/output schema.

## Generate Sample Input

```bash
belt app sample flux/dev-lora
belt app sample flux/dev-lora --save input.json
```

## Popular Apps by Category

### Image Generation
- `flux/dev-lora` — FLUX Dev with LoRA styles
- `seedream/4.5` — ByteDance, 4K cinematic, text rendering
- `gpt-image-2` — OpenAI, editing and inpainting
- `gemini/image-3-pro` — Google, high fidelity
- `grok-imagine` — xAI, fast creative
- `reve/image` — Natural language editing
- `p-image` — Fastest and cheapest ($0.0001/image)

### Video Generation
- `veo/3.1` — Google, highest quality
- `seedance/2.0` — ByteDance, audio-synced, multi-modal
- `happyhorse/1.0` — Alibaba, physically realistic
- `wan/2.5-i2v` — Image-to-video
- `grok-video` — xAI, fast
- `p-video` — Cheapest and fastest

### Avatars
- `p-video-avatar` — Built-in TTS, 30 voices, 1080p
- `omnihuman/1.5` — ByteDance, multi-character
- `fabric/1.0` — Lipsync with built-in TTS
- `pixverse-lipsync` — Most realistic lip movement

### Audio & TTS
- `inworld/tts-2` — Emotion steering, 100+ languages
- `elevenlabs/tts` — Premium, 32 languages, 22+ voices
- `kokoro/tts` — Natural and fast
- `dia/tts` — Conversational dialogue
- `elevenlabs/music` — Music generation, up to 10 min
- `elevenlabs/scribe-v2` — Transcription, 98%+ accuracy

### Enhancement
- `topaz/image-upscaler` — Image upscaling (2x, 4x)
- `topaz/video-upscaler` — Video upscaling
- `birefnet/bg-remove` — Background removal
- `hunyuanvideo/foley` — Add sound effects to video

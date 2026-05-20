# Talking-Head Avatars Reference

## Models

### P-Video-Avatar (Recommended)
- **p-video-avatar** — 18x faster and 6x cheaper than alternatives
- Built-in TTS: 30 voices across 10 languages, 1080p
- $0.025/second

```bash
belt app run p-video-avatar --input '{"image": "headshot.png", "text": "Welcome to our product demo.", "voice_id": "en-US-male-1"}'
```

### OmniHuman 1.5 (ByteDance)
- **omnihuman/1.5** — Multi-character, audio-driven
- Bring your own audio file

```bash
belt app run omnihuman/1.5 --input '{"image": "headshot.png", "audio": "narration.mp3"}'
```

### Fabric 1.0
- **fabric/1.0** — Lipsync with built-in TTS

### PixVerse Lipsync
- **pixverse-lipsync** — Most realistic lip movement, audio-driven

## Workflows

**Avatar from scratch:**
```bash
# 1. Generate a portrait
belt app run seedream/4.5 --input '{"prompt": "professional headshot, business casual, neutral background"}'
# 2. Create talking avatar
belt app run p-video-avatar --input '{"image": "output.png", "text": "Hello, welcome to the demo.", "voice_id": "en-US-female-1"}'
```

**Avatar with custom voice:**
```bash
# 1. Generate speech with emotion
belt app run inworld/tts-2 --input '{"text": "[enthusiastic] This product will change everything!", "voice_id": "en-US-narrator-1"}'
# 2. Drive avatar with the audio
belt app run omnihuman/1.5 --input '{"image": "presenter.png", "audio": "speech.mp3"}'
```

## Tips

- **Portrait quality matters.** Clear, front-facing headshot with good lighting. AI-generated portraits from Seedream or FLUX work well.
- **P-Video-Avatar is the default choice** — built-in TTS means one step instead of two.
- **For premium lipsync,** use PixVerse Lipsync with separately generated high-quality audio.
- **All avatar tasks are async.** Poll with `belt task get <id>`.
- **Keep scripts under 60 seconds** per clip. For longer content, generate multiple clips and merge.

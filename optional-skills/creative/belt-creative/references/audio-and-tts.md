# Audio, TTS, and Music Reference

## Text-to-Speech

### Inworld TTS-2
- **inworld/tts-2** — 100+ languages, 271+ voices, emotion steering
- **inworld/tts-1.5-max** — Low latency (<200ms)
- **inworld/tts-1.5-mini** — Ultra-low latency (~120ms)

Emotion control with bracket tags: `[happy]`, `[serious]`, `[whisper]`, `[excited]`.
Delivery modes: STABLE, BALANCED, CREATIVE.

```bash
belt app run inworld/tts-2 --input '{"text": "[excited] This is amazing!", "voice_id": "en-US-narrator-1", "delivery_mode": "BALANCED"}'
```

### ElevenLabs
- **elevenlabs/tts** — 22+ premium voices, 32 languages
- **elevenlabs/tts-flash** — Flash v2.5 for ultra-fast generation

Popular voices: `rachel`, `josh`, `adam`, `bella`, `elli`, `sam`.

### Other TTS
- **kokoro/tts** — Natural and fast. Voices: `am_michael`, `af_sarah`, `bf_emma`, `bm_george`
- **dia/tts** — Conversational, expressive dialogue
- **chatterbox** — General purpose
- **higgs-audio** — Emotional control via prompting
- **vibevoice** — Long-form content (podcasts, audiobooks)

## Music Generation

- **elevenlabs/music** — Up to 10 minutes, commercial license
- **diffrythm** — Fast song generation
- **tencent/song-generation** — Full songs with vocals

## Advanced Audio (ElevenLabs)

- **elevenlabs/voice-clone** — Clone a voice from samples
- **elevenlabs/dialogue** — Multi-speaker conversation generation
- **elevenlabs/sound-effects** — Foley and SFX from text
- **elevenlabs/voice-changer** — Transform existing recordings
- **elevenlabs/voice-isolator** — Extract clean speech from noisy audio
- **elevenlabs/dubbing** — Video localization / translation

## Speech-to-Text

- **elevenlabs/scribe-v2** — 98%+ accuracy, speaker diarization, 90+ languages
- **fast-whisper/large-v3** — Fast transcription
- **whisper/v3-large** — Highest accuracy

## Tips

- For emotion-rich narration, use Inworld TTS-2 with bracket tags.
- For premium quality in English, use ElevenLabs.
- For fastest generation, use Kokoro or Inworld TTS-1.5-mini.
- Avatar models (p-video-avatar, fabric) have built-in TTS — no need to generate audio separately.

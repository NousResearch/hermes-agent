# Irodori-TTS Skill

Generate speech via a local OpenAI-compatible Irodori-TTS endpoint.

## When to Use

- Read aloud secretary summaries, alerts, or calendar briefings locally

## Prerequisites

- Irodori-TTS-Server running at `http://127.0.0.1:8088`
- Start with `scripts/windows/start-irodori-tts-server.ps1`

## How to Run

```bash
py -3 skills/audio/irodori-tts/scripts/irodori_tts.py \
  --text "Good morning." \
  --output ~/.hermes/audio/briefing.wav
```

Long input is sentence-chunked automatically. Output is JSON with file path + metadata.

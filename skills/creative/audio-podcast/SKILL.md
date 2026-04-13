---
name: audio-podcast
description: >
  Generate a NotebookLM-style two-host conversational podcast from documents, reports,
  cron output, or any text content. Use when user says "make a podcast", "podcast from",
  "generate audio", "deep dive podcast", "summarize as podcast", "Alex and Sam",
  "listen to this", or wants audio versions of written content.
version: 2.0.0
author: Hermes Agent
metadata:
  hermes:
    tags: [Creative, Audio, Podcast, NotebookLM, TTS]
    related_skills: [youtube-content]
---

# Audio Podcast Generator

Generate engaging, natural-sounding two-host podcasts from any source material. Two AI hosts — **Alex** (the knowledgeable explainer) and **Sam** (the curious co-host) — discuss the content in a conversational format.

## How to Use

Call the `podcast_generate` tool with the source content:

```
podcast_generate(source="path/to/document.md")
podcast_generate(source="path/to/folder/", style="quick_brief")
podcast_generate(source="~/.hermes/cron/output/abc123/2026-04-10_08-00-00.md", style="deep_dive")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | string | required | File path, directory, or inline text |
| `title` | string | auto | Episode title |
| `style` | string | `deep_dive` | `deep_dive` (~10 min), `quick_brief` (~3 min), `debate` (opposing views) |
| `output_format` | string | `mp3` | `mp3`, `m4a`, or `wav` |

### Styles

- **deep_dive**: Thorough exploration. Alex explains concepts in depth, Sam asks probing follow-ups. 20-30 turns, ~8-12 minutes.
- **quick_brief**: Top 3-5 highlights. Brisk, punchy. 8-12 turns, ~2-4 minutes. Great for daily briefings.
- **debate**: Structured disagreement. Alex defends the source's conclusions, Sam plays devil's advocate. 15-25 turns, ~6-10 minutes.

## Output

Generated podcasts are saved to `~/.hermes/podcasts/{episode_id}/`:

```
~/.hermes/podcasts/a1b2c3d4/
├── podcast.mp3          # Final audio
├── transcript.json      # [{speaker, text, emotion, start_time, end_time}]
└── metadata.json        # {title, source, style, duration, created_at}
```

The tool returns a `MEDIA:` tag for automatic delivery on messaging platforms (Telegram, Discord, etc.).

## Examples

**From a cron briefing:**
> "Generate a quick_brief podcast from my latest morning market scan"

**From research notes:**
> "Create a deep_dive podcast about this research paper at ~/papers/attention.pdf"

**Debate format:**
> "Make a debate podcast from this policy document"

## Dependencies

Requires `edge-tts` and `pydub`:
```bash
pip install edge-tts pydub
```

## TTS Providers

### Edge TTS (default, free, cloud)
Microsoft neural voices. No API key needed. Fast but less expressive.

### VoxCPM (local, studio-grade)
OpenBMB's 2B-param TTS. 48kHz studio quality. Requires ~8GB VRAM.
Install: `pip install voxcpm`

Features:
- **Voice Design**: Generate unique voices from text descriptions — no reference audio needed
- **Voice Cloning**: Clone a voice from a short audio sample
- **Ultimate Cloning**: Highest fidelity with reference audio + transcript
- **Emotion/Pace**: Style descriptors applied per-line from the script's emotion/pace tags
- **30 languages**: Multilingual synthesis with seamless language switching

```
podcast_generate(source="doc.md", provider="voxcpm")
```

## Voice Configuration

Voices are configured in `~/.hermes/config.yaml`:

```yaml
podcast:
  tts_provider: "auto"    # auto | voxcpm | edge
  voices:
    alex:
      description: "A warm, authoritative male voice, mid-30s, American English"
      edge_voice: "en-US-GuyNeural"
      speed: 1.0
      reference_wav: ""    # Optional: path to .wav for voice cloning
      reference_text: ""   # Optional: transcript for ultimate cloning
    sam:
      description: "An energetic, curious female voice, late-20s, American English"
      edge_voice: "en-US-JennyNeural"
      speed: 1.05
      reference_wav: ""
      reference_text: ""
  voxcpm:
    cfg_value: 2.0         # Guidance strength (1.0-5.0, higher = more precise)
    inference_steps: 10    # Diffusion steps (5-20, higher = better quality)
    device: "auto"         # auto | cpu | cuda | mps
```

### Voice Cloning

To clone your own voice as a podcast host:

1. Record a 10-30 second audio clip (WAV, clear speech, no background noise)
2. Write the transcript of what you said
3. Set `reference_wav` and `reference_text` in the voice config
4. VoxCPM will reproduce your vocal characteristics when generating that host's lines

# Xiaomi MiMo TTS plugin

Text-to-speech backend for [Xiaomi MiMo-V2.5-TTS](https://platform.xiaomimimo.com/),
implementing the `TTSProvider` plugin surface (issue #46257, hook from #30398).

## Why a dedicated plugin

MiMo-V2.5-TTS serves synthesis through the OpenAI-compatible
`/v1/chat/completions` endpoint with an `audio` output parameter — not the
standard `/v1/audio/speech` route — and returns base64 WAV in
`choices[0].message.audio.data`. Swapping `base_url` on the built-in OpenAI
TTS provider therefore cannot work; this plugin implements the dedicated
request/response shape.

## Setup

1. Get an API key from <https://platform.xiaomimimo.com/> (TTS is
   free during the limited-time promotion).
2. Add it to `~/.hermes/.env` (or your environment):

   ```bash
   XIAOMI_API_KEY=<your key>
   # Optional — defaults to https://api.xiaomimimo.com/v1.
   # Token Plan users can point at their regional endpoint, e.g.
   # XIAOMI_BASE_URL=https://token-plan-sgp.xiaomimimo.com/v1
   ```

   `MIMO_API_KEY` / `MIMO_BASE_URL` are accepted as fallback aliases.
3. Select the provider in `~/.hermes/config.yaml`:

   ```yaml
   tts:
     provider: mimo
     voice: 冰糖        # optional; default: mimo_default (cluster choice)
     # output_format: wav  # optional; default is mp3 (needs ffmpeg)
   ```

## Configuration reference

| Setting | Source | Purpose |
|---|---|---|
| API key | `XIAOMI_API_KEY` (or `MIMO_API_KEY`) | Authentication |
| Endpoint | `XIAOMI_BASE_URL` > `MIMO_BASE_URL` > `tts.mimo.base_url` | Global / Token Plan endpoint |
| Voice | `tts.voice` | Any preset voice id (see below) |
| Model | `tts.model` | Defaults to `mimo-v2.5-tts` |
| Style | `MIMO_TTS_STYLE` env or `tts.mimo.style` | Natural-language delivery instruction |
| Timeout | `MIMO_TTS_TIMEOUT` env or `tts.mimo.timeout` | Seconds, default 60 |
| Text cap | `tts.mimo.max_text_length` | Overrides the 4000-char fallback |

## Preset voices

| Voice ID | Language | Gender |
|---|---|---|
| `mimo_default` | cluster-dependent (冰糖 on China clusters, Mia elsewhere) | — |
| `冰糖` | Chinese | female |
| `茉莉` | Chinese | female |
| `苏打` | Chinese | male |
| `白桦` | Chinese | male |
| `Mia` | English | female |
| `Chloe` | English | female |
| `Milo` | English | male |
| `Dean` | English | male |

## Style control

MiMo follows natural-language delivery instructions. Set one globally via
`MIMO_TTS_STYLE` / `tts.mimo.style`; it is sent as the optional `user`
message (never spoken). Inline audio tags also work inside the synthesized
text, e.g. `(笑声)`, `[开心]`, `[语速加快]`.

## Output formats and ffmpeg

MiMo natively returns WAV. `format: wav` writes the decoded bytes
directly; other formats (the dispatcher default is mp3) are converted with
`ffmpeg`, which must be on `PATH`. `voice_compatible` is enabled, so
gateway voice-bubble delivery converts to Opus through the existing
pipeline.

## Limitations

- Preset-voice model only (`mimo-v2.5-tts`); `voicedesign` / `voiceclone`
  are not exposed here.
- No streaming (`stream()` falls back to whole-file `synthesize`).
- The numeric `speed` parameter is ignored — control pace with style
  instructions / tags instead.
- Transient failures (429 / connection errors) get exactly one retry.

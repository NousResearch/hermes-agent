"""The default ``tts`` config block (every built-in speech provider) for ``DEFAULT_CONFIG``.

Split out of :mod:`hermes_cli.config_defaults` as a topical sibling: that module is over the
code-health file cap and may only shrink, so the provider config block lives here and it imports
``TTS_DEFAULTS``. Pure data — nothing here imports or executes anything.

On-device providers in this block:

* ``phonon`` — Gradium's 31M/40M models on the ``ptts`` runtime (CPU, numpy-only wheel: no torch,
  no ONNX runtime, no espeak-ng, no API key). 102 catalogue voices across five languages.
* ``ollama`` — speech models served by an Ollama daemon >= 0.40.9 over its OpenAI-compatible
  ``POST /v1/audio/speech``. No SDK and no key either.
"""

from __future__ import annotations

TTS_DEFAULTS: dict = {
        # "edge" (free) | "elevenlabs" (premium) | "openai" | "xai" | "minimax" | "mistral" |
        # "gemini" | "deepinfra" | "neutts" (local) | "kittentts" (local) | "piper" (local)
        "provider": "edge",
        # Seconds a local engine (Piper, KittenTTS) stays loaded after the last speech toggle
        # turns off, so a quick re-activation (wake word, voice-chat restart) skips the reload.
        # 0 unloads immediately.
        "keep_warm_seconds": 60,
        "streaming": {
            # Shortest first sentence (chars) spoken on its own by streaming TTS; shorter openers
            # ride with the next sentence. 20 suits English; CJK voice setups use ~6.
            "min_len": 20,
        },
        "edge": {
            # Popular: AriaNeural, JennyNeural, AndrewNeural, BrianNeural, SoniaNeural
            "voice": "en-US-AriaNeural",
        },
        "elevenlabs": {
            "voice_id": "pNInz6obpgDQGcFmaJgB",  # Adam
            "model_id": "eleven_multilingual_v2",
        },
        "openai": {
            "model": "gpt-4o-mini-tts",
            # gpt-4o-mini-tts voices: alloy, ash, ballad, cedar, coral, echo, fable, marin, nova,
            # onyx, sage, shimmer, verse
            "voice": "alloy",
            # Forwarded verbatim in the request body for OpenAI-compatible servers whose cloned
            # voices demand it (400 consent_required otherwise); "" sends nothing.
            "consent_attestation": "",
            # Raw PCM rate for streaming playback. OpenAI emits 24 kHz; a compatible endpoint that
            # reports its rate (X-Audio-Sample-Rate header) overrides this automatically.
            "pcm_sample_rate": 24000,
        },
        "gemini": {
            "model": "gemini-2.5-flash-preview-tts",
            "voice": "Kore",
            # Gemini 3.1: aux-model rewrite inserts [audio tags] into the TTS script only.
            "audio_tags": False,
            # Optional local text file with performance direction; may include a `{transcript}`
            # placeholder, else the live transcript is appended.
            "persona_prompt_file": "",
        },
        "xai": {
            "voice_id": "eve",  # or a custom voice ID (docs.x.ai custom voices)
            "language": "en",  # BCP-47 code ("en", "pt-BR") or "auto"
            "speed": 1.0,  # 0.7–1.5
            "auto_speech_tags": False,  # insert expressive audio tags via LLM rewrite
            "optimize_streaming_latency": 0,  # 0–2, trades quality for lower latency
            "sample_rate": 24000,  # 22050 / 24000 / 44100 / 48000
            "bit_rate": 128000,  # MP3 bitrate; only applies when codec=mp3
        },
        "mistral": {
            "model": "voxtral-mini-tts-2603",
            "voice_id": "c69964a6-ab8b-4f8a-9465-ec0925096ec8",  # Paul - Neutral
        },
        "minimax": {"model": "speech-02-hd", "voice_id": "English_expressive_narrator"},
        "kittentts": {
            "model": "KittenML/kitten-tts-nano-0.8-int8",  # nano 25MB; micro 41MB; mini 80MB
            "voice": "Jasper",
        },
        "neutts": {
            "ref_audio": "",  # path to reference voice audio (empty = bundled default)
            "ref_text": "",   # path to reference voice transcript (empty = bundled default)
            "model": "neuphonic/neutts-air-q4-gguf",  # HuggingFace model repo
            "device": "cpu",  # cpu, cuda, or mps
        },
        "piper": {
            # Voice name (downloaded on first use) or absolute path to a .onnx file; list:
            # github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md. Optional keys: voices_dir
            # (~/.hermes/cache/piper-voices/), use_cuda, length_scale (2.0 = twice as slow),
            # noise_scale, noise_w_scale, volume, normalize_audio.
            "voice": "en_US-lessac-medium",
        },
        "deepinfra": {
            "model": "",  # empty = first tts-tagged model from the live catalog
            "voice": "default",
            # optional "base_url" key overrides DEEPINFRA_BASE_URL for TTS only
        },
        "phonon": {
            # Gradium Phonon on the `ptts` runtime: CPU, numpy only, 24 kHz mono WAV.
            # model: 31m-en (Apache 2.0, 64MB) | 40m-en | 40m-fr | 40m-de | 40m-es | 40m-pt |
            # 40m-multilingual. Aliases: en/fr/de/es/pt, multilingual.
            "model": "hf://gradium/phonon-31m-en",
            # 102 voices (tts.phonon.voice = a catalogue name or voice_id); "default" = the
            # checkpoint's own voice, the only one that needs no extra download.
            "voice": "default",
            "quant": "q8",  # ptts weight format: f32, q8, q8k, q6k, q5k, q4k, ...
            # Text-normalization language (numbers, dates, currency). "" = the checkpoint's own
            # language, else the selected voice's locale (multilingual checkpoint only).
            "lang": "",
            "device": "auto",  # ptts picks the first available; 0.4.x ships CPU-only backends
            "threads": 0,  # 0 = ptts default CPU thread count
            # Where an unknown voice is looked up: an "hf://owner/name" catalog, a local
            # voices.jsonl, or a folder holding one. "" = gradium/phonon-voices.
            "voice_catalog": "",
        },
        "ollama": {
            # Ollama >= 0.40.9 serves speech models over the OpenAI-compatible
            # POST /v1/audio/speech. Phonon's Ollama packages require Apple Silicon today;
            # any speech model the daemon reports works here, on any host that runs Ollama.
            "base_url": "http://localhost:11434/v1",
            "model": "phonon",  # phonon[:31m-en] | phonon:40m-en|fr|de|es|pt|multilingual
            "voice": "default",  # the voice(s) the pulled model exposes (/api/show -> voices)
            "timeout": 120,  # seconds; first call also loads the model
        },
}

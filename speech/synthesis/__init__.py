"""Text-to-speech engines behind the ``text_to_speech`` tool and the voice surfaces.

- ``delivery``: output post-processing, container repair and delivery files.
- ``providers``: cloud backends (Edge, ElevenLabs, xAI, MiniMax, Mistral, Gemini).
- ``openai_compatible``: OpenAI and DeepInfra backends and the managed-gateway route.
- ``local``: local backends (Piper, KittenTTS, NeuTTS).
- ``neutts_synth``: the NeuTTS child script, run by file path from ``local``.
- ``plugins``: dispatch to plugin TTS providers.
- ``text_normalize``: turns assistant Markdown into a spoken script.
- ``streaming``: streaming providers for voice surfaces.
- ``speaker``: local playback of streamed speech.
- ``lifecycle``: warm-up and release leases for local engines.
"""

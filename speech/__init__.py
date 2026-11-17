"""Text-to-speech, speech-to-text, voice mode and wake word for every surface.

- ``synthesis``: text-to-speech providers, delivery, streaming and local playback.
- ``transcription``: speech-to-text providers, audio preparation and live sessions.
- ``voice``: push-to-talk capture and playback, live voice sessions, the client voice config.
- ``wake_word``: the "Hey Hermes" hotword listener and its engines.
- ``audio_container``: detects an audio file's container from its magic bytes.
- ``command_provider``: user-declared shell-command providers for both directions.
- ``voice_mode_transcript``: transcript rules for voice mode: the stop phrase and the TTS-echo guard.
"""

"""Speech-to-text for voice notes, voice mode, wake word and the desktop app.

- ``transcriber``: provider resolution and dispatch; ``transcribe_audio``.
- ``settings``: constants, result envelopes and config readers shared by the STT modules.
- ``audio``: audio preparation (ffmpeg, silk, CAF, silence trim).
- ``chunking``: splits cloud input over a provider's upload cap.
- ``cloud``: cloud backends.
- ``command``: the command-provider backend.
- ``local``: the faster-whisper backend.
- ``whisper_cpp``: the managed whisper.cpp command.
- ``streaming``: live transcription sessions.
- ``lease``: warm-up and release leases for the local engine.
"""

"""``DEFAULT_CONFIG["voice"]`` — how voice conversations are wired (the user-facing docs of
that config.yaml section live in these comments). Extracted from ``config_defaults.py`` to keep
that module inside its size ratchet; the leaf-data rule is the same (no imports from
``hermes_cli.config``)."""

VOICE_DEFAULTS = {
    # How the Desktop voice conversation is wired:
    #   chained  — STT → Hermes turn → TTS (the stt.* / tts.* providers below)
    #   gpt-live — one full-duplex voice model (OpenAI GPT-Live) owns the mic and speaker and
    #              DELEGATES every real request to Hermes (any model / provider you have
    #              selected); needs an OpenAI API key. $0.05/min voice layer billing.
    "voice_chat_mode": "chained",
    "gpt_live": {
        "model": "gpt-live-1",
        "voice": "marin",  # marin | quartz | ripple | vesper | willow | stone | gleam | meridian | ...
        # Extra sentences appended to the live model's conversation persona (tone, pacing, language).
        "instructions": "",
        # optional "api_key" / "base_url" keys override the OpenAI audio credentials for this mode only
    },
    "record_key": "ctrl+b",
    "submit_mode": "direct",  # TUI: direct submits immediately; draft = editable transcript
    "max_recording_seconds": 120,
    "auto_tts": False,
    # Desktop remote clients call STT/TTS providers DIRECTLY (config + key fetched over
    # authenticated REST at session start) instead of relaying via the gateway.
    "client_direct": True,
    "beep_enabled": True,  # record start/stop beeps in CLI voice mode
    "beep_volume": 0.3,  # beep amplitude multiplier, 0.0-1.0
    "thinking_sound": True,  # ambient bubble sound while the agent works (volume = beep_volume)
    # Tuning for LOCAL live transcription (tools.voice_partial), which answers `stt.streaming`
    # for providers with no realtime wire: while the mic is open a worker re-decodes a capped
    # tail of the running capture and surfaces render each partial as a preview. The take is
    # always transcribed in full afterwards, so a partial is never the authoritative transcript.
    # Only the tuning lives here — the switch is stt.streaming itself.
    "partial": {
        "tail_seconds": 20.0,  # how much of the running capture each re-decode reads (capped)
        "interval_seconds": 2.0,  # re-decode cadence; a tick landing mid-decode is skipped
        "min_seconds": 1.5,  # skip re-decodes until at least this much audio exists
    },
    "silence_threshold": 200,  # RMS below this = silence (0-32767)
    "silence_duration": 3.0,  # seconds of silence before auto-stop
    "barge_in": True,  # interrupt the agent / stop TTS when the user starts talking
    # Trip suppression after TTS onset (mic stays live the whole turn).
    "barge_in_grace_seconds": 0.5,
    # Speech trigger = quiet-room floor x this (floor calibrated BEFORE playback).
    "barge_in_threshold_multiplier": 3.0,
    # Saying EXACTLY one of these (case-insensitive, punctuation ignored) ends the voice chat
    # instead of going to the agent. [] disables.
    "stop_phrases": ["stop"],
}

"""Defaults for the ``voice`` section of DEFAULT_CONFIG: CLI/TUI/Desktop voice and the
server-owned converse WebSocket. Pure data; the comments are the user-facing docs of config.yaml.
"""

VOICE_DEFAULTS = {
    # How the Desktop voice conversation is wired:
    #   chained  — STT → Hermes turn → TTS (the stt.* / tts.* providers)
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
    # ── server-owned converse WS (/v1/audio/converse) ──
    # Voice-only model override for the converse path. "" = use model.default; set a model
    # group here to run voice on its own deployment / reasoning_effort without changing the
    # global default. provider defaults to the configured one.
    "model": "",
    "provider": "",
    # Hard cap on one converse agent turn: a hung LLM call (no tokens, no error) is abandoned
    # here and the socket returns to listening instead of hanging "thinking" forever.
    "turn_timeout_seconds": 120.0,
    # Playback-tail fallback: between turn_done and the client's {"type":"drained"} the client
    # is still PLAYING the reply; the server holds quiet/VAD for the tail and gives up after
    # this many seconds if drained never arrives.
    "playback_tail_timeout_seconds": 45.0,
    # Smart Turn v3 semantic end-of-turn endpointer (opt-in; needs the [voice-endpoint] extra).
    # model: "" = off (fixed silence endpoint); any truthy value enables it. threshold =
    # P(turn complete) at/above which the turn commits.
    "endpoint": {"model": "", "threshold": 0.5},
    # Speech detector for the converse WS: "silero" (model-based; uses onnxruntime from the
    # [voice] extra and a 1.3 MB model fetched once) or "energy" (RMS thresholds). Silero
    # falls back to energy, with a warning, when it can't load.
    "vad": {"model": "silero"},
}

"""Native audio content-block translation shared by Gemini's sync and async clients."""

import base64

GEMINI_AUDIO_MIMES = {
    "wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/m4a",
    "aac": "audio/aac", "ogg": "audio/ogg", "flac": "audio/flac",
}


def gemini_audio_part(item: dict) -> dict:
    audio = item.get("input_audio")
    if not isinstance(audio, dict):
        raise ValueError("input_audio must contain data and format")
    mime = GEMINI_AUDIO_MIMES.get(audio.get("format"))
    data = audio.get("data")
    if not mime or not isinstance(data, str) or not data:
        raise ValueError("input_audio requires nonempty base64 data and a supported audio format")
    # Refuse malformed audio rather than silently sending only the accompanying question.
    base64.b64decode(data, validate=True)
    return {"inlineData": {"mimeType": mime, "data": data}}

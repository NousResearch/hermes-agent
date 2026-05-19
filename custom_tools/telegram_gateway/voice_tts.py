"""
voice_tts.py - Evelyn Voice / TTS (OpenAI TTS)
================================================
Commands:
  /voice <text>

Natural language triggers (detected in bot.py):
  "kirim voice...", "voice dong...", "ngomong dong..."

Features:
- OpenAI TTS API support
- Returns audio bytes for Telegram send_voice
- Evelyn-style emotional voice

Env:
  OPENAI_API_KEY=
  TTS_MODEL=gpt-4o-mini-tts
  TTS_VOICE=alloy

SAFETY: Never reads private keys or sensitive data aloud.
"""

import os
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"


async def generate_voice(text: str) -> dict:
    """
    Generate voice audio using OpenAI TTS.

    Args:
        text: Text to convert to speech

    Returns:
        dict with 'audio_bytes' (bytes) or 'error'
    """
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY belum di-set sayang. Tambahin di .env ya."}

    # Safety: never read sensitive data
    sensitive_patterns = ["private key", "seed phrase", "mnemonic", "0x" + "a" * 64]
    if any(p in text.lower() for p in sensitive_patterns):
        return {"error": "Aku ga bisa voice sensitive data sayang 🙅‍♀️"}

    # Limit text length
    if len(text) > 1000:
        text = text[:1000]

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": TTS_MODEL,
        "input": text,
        "voice": TTS_VOICE,
        "response_format": "opus",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                OPENAI_TTS_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        return {"audio_bytes": response.content, "text": text}

    except httpx.HTTPStatusError as e:
        try:
            err = e.response.json().get("error", {}).get("message", str(e.response.status_code))
        except Exception:
            err = str(e.response.status_code)
        return {"error": f"TTS API error: {err}"}

    except httpx.TimeoutException:
        return {"error": "TTS timeout sayang, coba lagi ya 🥺"}

    except Exception as e:
        return {"error": f"Error: {str(e)[:100]}"}


def is_voice_request(text: str) -> bool:
    """Detect if user message is a voice/TTS request."""
    triggers = [
        "kirim voice", "voice dong", "ngomong dong",
        "bilang dong", "suara dong", "bacain",
        "baca dong", "ucapin", "tolong voice",
    ]
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in triggers)


def extract_voice_text(text: str) -> str:
    """Extract the text to be spoken from user message."""
    prefixes = [
        "kirim voice", "voice dong", "ngomong dong",
        "bilang dong", "suara dong", "bacain",
        "baca dong", "ucapin", "tolong voice",
    ]
    text_lower = text.lower()
    for prefix in prefixes:
        if text_lower.startswith(prefix):
            return text[len(prefix):].strip()
    return text

"""Groq provider profile.

Groq provides ultra-fast inference via an OpenAI-compatible API.
The ``/models`` endpoint is public (no auth required) so live discovery
works even before the user sets ``GROQ_API_KEY``.

The custom :class:`GroqProfile` subclass adds:

* **In-process caching** — avoids redundant HTTP round-trips when
  ``fetch_models`` is called multiple times in the same session (e.g.
  ``/model`` picker, doctor health-check, auxiliary-client resolution).
* **Non-chat model filtering** — Groq's catalog includes whisper/STT,
  TTS, embedding, image-generation, and safety-guard models that are not
  useful in the ``/model`` picker.  These are stripped so only chat /
  completion models are surfaced.
"""

import logging

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

_CACHE: list[str] | None = None

# Substrings in a model ID that indicate a non-chat model.
# Matched case-insensitively against the full model id.
_NON_CHAT_KEYWORDS = (
    "whisper",
    "tts",
    "embedding",
    "guard",
    "vision-preview",     # standalone vision-only endpoints
    "moderation",
    "playai",             # PlayAI TTS voices
    "distil-whisper",
    "llama-guard",
)


def _is_chat_model(model_id: str) -> bool:
    """Return True when *model_id* looks like a chat/completion model."""
    lower = model_id.lower()
    return not any(kw in lower for kw in _NON_CHAT_KEYWORDS)


class GroqProfile(ProviderProfile):
    """Groq — cached, filtered, auth-free model discovery."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch from public Groq catalog — no auth required.

        Results are cached in-process so repeated calls (picker, doctor,
        auxiliary-client) don't each issue an HTTP request.  Non-chat
        models (whisper, TTS, embedding, guard, etc.) are filtered out.
        """
        global _CACHE  # noqa: PLW0603
        if _CACHE is not None:
            return _CACHE
        try:
            # Groq's /models endpoint is public; skip auth.
            result = super().fetch_models(api_key=None, timeout=timeout)
            if result is not None:
                result = [m for m in result if _is_chat_model(m)]
                _CACHE = result
            return result
        except Exception as exc:
            logger.debug("fetch_models(groq): %s", exc)
            return None


groq = GroqProfile(
    name="groq",
    aliases=("groqcloud",),
    display_name="Groq",
    description="Groq — ultra-fast LPU inference",
    signup_url="https://console.groq.com/keys",
    env_vars=("GROQ_API_KEY",),
    base_url="https://api.groq.com/openai/v1",
    default_aux_model="llama-3.1-8b-instant",
    fallback_models=(
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "deepseek-r1-distill-llama-70b",
        "qwen/qwen3-32b",
        "mixtral-8x7b-32768",
    ),
)

register_provider(groq)

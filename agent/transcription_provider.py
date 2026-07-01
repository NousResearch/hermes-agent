"""
Transcription Provider ABC
==========================

Defines the pluggable-backend interface for speech-to-text. Providers
register instances via
:meth:`PluginContext.register_transcription_provider`; the active one
(selected via ``stt.provider`` in ``config.yaml``) services every
:func:`tools.transcription_tools.transcribe_audio` call **when the
configured name is neither a built-in (``local``, ``local_command``,
``groq``, ``openai``, ``mistral``, ``xai``) nor disabled**.

Two coexisting STT extension surfaces — in resolution order:

1. **Built-in providers** (``BUILTIN_STT_PROVIDERS`` in
   :mod:`tools.transcription_tools`) — native Python implementations
   for the 6 backends shipped today (faster-whisper, local_command,
   Groq, OpenAI, Mistral, xAI). **Always win** — plugins cannot
   shadow them. The single-env-var shell escape hatch
   ``HERMES_LOCAL_STT_COMMAND`` is preserved via the built-in
   ``local_command`` path.
2. **Plugin-registered providers** (this ABC). For new STT backends —
   OpenRouter, SenseAudio, Gemini-STT, custom proprietary engines —
   that need a Python implementation without modifying
   ``tools/transcription_tools.py``.

Built-ins-always-win is enforced at registration time
(:func:`agent.transcription_registry.register_provider` rejects names
in ``BUILTIN_STT_PROVIDERS`` with a warning) AND at dispatch time
(:func:`tools.transcription_tools._dispatch_to_plugin_provider`
re-checks defensively).

Providers live in ``<repo>/plugins/transcription/<name>/`` (built-in
plugins, none shipped today) or
``~/.hermes/plugins/transcription/<name>/`` (user-installed).

Response contract
-----------------
:meth:`TranscriptionProvider.transcribe` returns a dict with keys::

    success      bool
    transcript   str       transcribed text (empty when success=False)
    provider     str       provider name (for diagnostics)
    error        str       only when success=False
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class TranscriptionProvider(abc.ABC):
    """Abstract base class for a speech-to-text backend.

    Subclasses must implement :attr:`name` and :meth:`transcribe`.
    Everything else has sane defaults — override only what your provider
    needs.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier used in ``stt.provider`` config.

        Lowercase, no spaces. Examples: ``openrouter``, ``sensaudio``,
        ``gemini``, ``deepgram``. Names that collide with a built-in STT
        provider (``local``, ``local_command``, ``groq``, ``openai``,
        ``mistral``, ``xai``) are rejected at registration time.
        """

    @property
    def display_name(self) -> str:
        """Human-readable label shown in ``hermes tools``.

        Defaults to ``name.title()``.
        """
        return self.name.title()

    def is_available(self) -> bool:
        """Return True when this provider can service calls.

        Typically checks for a required API key + that the SDK is
        importable. Default: True (providers with no external
        dependencies are always available).

        Must NOT raise — used by the picker and ``hermes setup`` for
        availability displays and should fail gracefully.
        """
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        """Return model catalog entries.

        Each entry::

            {
                "id": "whisper-large-v3-turbo",  # required
                "display": "Whisper Large v3 Turbo",   # optional
                "languages": ["en", "es", "fr"],        # optional
                "max_audio_seconds": 1500,              # optional
            }

        Default: empty list (provider has a single fixed model or
        doesn't expose model selection).
        """
        return []

    def default_model(self) -> Optional[str]:
        """Return the default model id, or None if not applicable."""
        models = self.list_models()
        if models:
            return models[0].get("id")
        return None

    def get_setup_schema(self) -> Dict[str, Any]:
        """Return provider metadata for the ``hermes tools`` picker.

        Used by ``tools_config.py`` to inject this provider as a row in
        the Speech-to-Text provider list. Shape::

            {
                "name": "OpenRouter STT",              # picker label
                "badge": "paid",                       # optional short tag
                "tag": "Whisper via OpenRouter API",   # optional subtitle
                "env_vars": [                          # keys to prompt for
                    {"key": "OPENROUTER_API_KEY",
                     "prompt": "OpenRouter API key",
                     "url": "https://openrouter.ai/keys"},
                ],
            }

        Default: minimal entry derived from ``display_name`` with no
        env vars. Override to expose API key prompts and custom badges.
        """
        return {
            "name": self.display_name,
            "badge": "",
            "tag": "",
            "env_vars": [],
        }

    @abc.abstractmethod
    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Transcribe the audio file at ``file_path``.

        Returns a dict with the standard envelope::

            {
                "success": True,
                "transcript": "the transcribed text",
                "provider": "<this provider's name>",
            }

        or on failure::

            {
                "success": False,
                "transcript": "",
                "error": "human-readable error message",
                "provider": "<this provider's name>",
            }

        Implementations should NOT raise — convert exceptions to the
        error envelope so the dispatcher can deliver a consistent shape
        to the gateway/CLI caller.

        Args:
            file_path: Absolute path to the audio file. The dispatcher
                has already validated existence + size before calling.
            model: Model identifier from :meth:`list_models`, or None
                to use :meth:`default_model`.
            language: Optional BCP-47 language hint (e.g. ``"en"``,
                ``"ja"``) — providers without language hints should
                ignore this argument.
            **extra: Forward-compat parameters future schema versions
                may expose. Implementations should ignore unknown keys.
        """


# ---------------------------------------------------------------------------
# Built-in fallback provider: FuelIX
# ---------------------------------------------------------------------------
#
# Not registered via ``register_provider`` / ``BUILTIN_STT_PROVIDERS`` — this
# is invoked directly by ``tools.transcription_tools.transcribe_audio`` as a
# same-request fallback when the user's configured primary STT provider
# raises or times out (Phase 4: FuelIX media fallback). It is also reachable
# explicitly via ``stt.provider: fuelix``.


class FuelIXTranscriptionProvider(TranscriptionProvider):
    """Speech-to-text fallback backed by FuelIX's ``whisper-1`` endpoint."""

    @property
    def name(self) -> str:
        return "fuelix"

    @property
    def display_name(self) -> str:
        return "FuelIX"

    def is_available(self) -> bool:
        try:
            from hermes_cli.config import load_config

            cfg = load_config()
            api_key = (
                cfg.get("providers", {})
                .get("api", {})
                .get("fuelix.ai", {})
                .get("api_key", "")
            )
            return bool(str(api_key).strip())
        except Exception:
            return False

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        try:
            import requests

            from hermes_cli.config import load_config

            cfg = load_config()
            api_key = (
                cfg.get("providers", {})
                .get("api", {})
                .get("fuelix.ai", {})
                .get("api_key", "")
            )
            if not api_key:
                return {
                    "success": False,
                    "transcript": "",
                    "error": "FuelIX API key missing in config.providers.api.fuelix.ai.api_key",
                    "provider": self.name,
                }

            resolved_model = model or "whisper-1"
            with open(file_path, "rb") as fh:
                response = requests.post(
                    "https://api.fuelix.ai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": fh},
                    data={"model": resolved_model},
                    timeout=60,
                )
            response.raise_for_status()
            data = response.json()
            text = str(data.get("text") or "")
            return {"success": True, "transcript": text, "provider": self.name}
        except Exception as exc:
            logger.error("FuelIX STT transcription failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "transcript": "",
                "error": f"FuelIX STT transcription failed: {exc}",
                "provider": self.name,
            }

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
                The dispatcher currently forwards ``prompt`` here when a
                transcription prompt is set (via the ``stt.prompt`` config
                key or a ``pre_transcription`` hook) — prompt-capable
                providers may use it as a vocabulary/context hint; others
                should ignore it.
        """

    # ------------------------------------------------------------------
    # Live streaming (optional capability)
    # ------------------------------------------------------------------

    @property
    def streaming_capable(self) -> bool:
        """Whether this provider can transcribe LIVE audio chunk-by-chunk.

        Streaming lets a caller feed audio as it is captured (microphone
        input) instead of only whole files, cutting dictation latency
        from "record, then transcribe" to "transcribe while speaking".

        Providers that return True must implement
        :meth:`open_stream_session` and accept 16 kHz mono s16le PCM in
        arbitrary chunk sizes. Default: False — the file-based
        :meth:`transcribe` path is unchanged for everyone else.
        """
        return False

    def open_stream_session(
        self,
        *,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> "TranscriptionStreamSession":
        """Open a live transcription session (streaming-capable only).

        Args:
            language: Optional BCP-47 language hint (e.g. ``"pt"``).
            prompt: Optional vocabulary/context hint — same semantics as
                the ``prompt`` extra on :meth:`transcribe`.

        Returns:
            A :class:`TranscriptionStreamSession` bound to this provider.
            Sessions are single-use; open a fresh one per utterance.

        Raises:
            NotImplementedError: this provider is not streaming-capable.
        """
        raise NotImplementedError(
            f"{self.name} does not support live streaming transcription"
        )


class TranscriptionStreamSession(abc.ABC):
    """Live audio → transcript session from a streaming-capable provider.

    Contract: audio arrives as 16 kHz mono s16le PCM in arbitrary chunk
    sizes via :meth:`push_audio`; :meth:`finalize` ends the session,
    flushes trailing audio and returns the standard transcription
    envelope (``success``/``transcript``/``provider``).

    Sessions are single-use. A session may transcribe on a background
    thread (e.g. a provider websocket reader); implementations must make
    :meth:`push_audio` and :meth:`partial_transcript` safe to call from
    the feeding thread.
    """

    @abc.abstractmethod
    def push_audio(self, chunk: bytes) -> None:
        """Feed one chunk of 16 kHz mono s16le PCM to the session."""

    @abc.abstractmethod
    def finalize(self) -> Dict[str, Any]:
        """End the session and return the transcription envelope.

        Blocks until the provider has finalized the transcript. Must not
        raise — convert failures to the error envelope, matching
        :meth:`TranscriptionProvider.transcribe`.
        """

    def partial_transcript(self) -> str:
        """Latest non-final transcript (``''`` when none available).

        Callers may poll this to show live captions while the user is
        still speaking. Default: no partials.
        """
        return ""

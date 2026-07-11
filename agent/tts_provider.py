"""
Text-to-Speech Provider ABC
============================

Defines the pluggable-backend interface for text-to-speech synthesis.
Providers register instances via
``PluginContext.register_tts_provider()``; the active one (selected via
``tts.provider`` in ``config.yaml``) services every ``text_to_speech``
tool call **only when the configured name is neither a built-in nor a
command-type provider declared under ``tts.providers.<name>``**.

Three coexisting TTS extension surfaces — in resolution order:

1. **Built-in providers** (``BUILTIN_TTS_PROVIDERS`` in
   :mod:`tools.tts_tool`) — native Python implementations (edge, openai,
   elevenlabs, …). **Always win** — plugins cannot shadow them.
2. **Command-type providers** declared under ``tts.providers.<name>:
   type: command`` (PR #17843, commit ``2facea7f7``). Wire any local
   CLI into Hermes with shell-template placeholders. **Wins over a
   same-name plugin** — config is more local than plugin install.
3. **Plugin-registered providers** (this ABC). For backends that need a
   Python SDK, streaming bytes, OAuth refresh, or voice-listing APIs
   the shell-template grammar can't reasonably express.

Built-ins-always-win is enforced at registration time
(:func:`agent.tts_registry.register_provider` rejects names in
``BUILTIN_TTS_PROVIDERS`` with a warning) AND at dispatch time
(:func:`tools.tts_tool._dispatch_to_plugin_provider` re-checks
defensively). The dispatcher also rejects plugin dispatch when a same-
name command provider is configured.

Providers live in ``<repo>/plugins/tts/<name>/`` (built-in plugins, no
shipped today) or ``~/.hermes/plugins/tts/<name>/`` (user-installed).
None ship in-tree as of issue #30398 — the hook is additive
infrastructure waiting for a real consumer (Cartesia, Fish Audio, …).

Response contract
-----------------
:meth:`TTSProvider.synthesize` writes the audio bytes to ``output_path``
and returns the path as a string. Implementations should raise on
failure — the dispatcher converts exceptions into the standard
``{success: False, error: …}`` JSON envelope the rest of Hermes
expects.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_FORMAT = "mp3"
VALID_OUTPUT_FORMATS = frozenset({"mp3", "wav", "ogg", "opus", "flac"})


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class TTSProvider(abc.ABC):
    """Abstract base class for a text-to-speech backend.

    Subclasses must implement :attr:`name` and :meth:`synthesize`.
    Everything else has sane defaults — override only what your provider
    needs.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier used in ``tts.provider`` config.

        Lowercase, no spaces. Examples: ``cartesia``, ``fishaudio``,
        ``deepgram``. Names that collide with a built-in TTS provider
        (``edge``, ``openai``, ``elevenlabs``, ``minimax``, ``gemini``,
        ``mistral``, ``xai``, ``fish``, ``piper``, ``kittentts``, ``neutts``) are
        rejected at registration time.
        """

    @property
    def display_name(self) -> str:
        """Human-readable label shown in ``hermes tools``.

        Defaults to ``name.title()`` (e.g. ``Cartesia`` for ``cartesia``).
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

    def list_voices(self) -> List[Dict[str, Any]]:
        """Return voice catalog entries.

        Each entry::

            {
                "id": "voice-abc-123",                # required
                "display": "Aria — neutral female",    # optional; defaults to id
                "language": "en-US",                   # optional
                "gender": "female",                    # optional
                "preview_url": "https://...mp3",       # optional
            }

        Default: empty list (provider has no enumerable voices or
        doesn't surface them via API).
        """
        return []

    def list_models(self) -> List[Dict[str, Any]]:
        """Return model catalog entries.

        Each entry::

            {
                "id": "sonic-2",                       # required
                "display": "Sonic 2",                  # optional
                "languages": ["en", "es", "fr"],       # optional
                "max_text_length": 5000,               # optional
            }

        Default: empty list (provider has a single fixed model or
        doesn't expose model selection).
        """
        return []

    def get_setup_schema(self) -> Dict[str, Any]:
        """Return provider metadata for the ``hermes tools`` picker.

        Used by ``tools_config.py`` to inject this provider as a row in
        the Text-to-Speech provider list. Shape::

            {
                "name": "Cartesia",                    # picker label
                "badge": "paid",                       # optional short tag
                "tag": "Ultra-low-latency streaming",  # optional subtitle
                "env_vars": [                          # keys to prompt for
                    {"key": "CARTESIA_API_KEY",
                     "prompt": "Cartesia API key",
                     "url": "https://play.cartesia.ai/console"},
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

    def default_model(self) -> Optional[str]:
        """Return the default model id, or None if not applicable."""
        models = self.list_models()
        if models:
            return models[0].get("id")
        return None

    def default_voice(self) -> Optional[str]:
        """Return the default voice id, or None if not applicable."""
        voices = self.list_voices()
        if voices:
            return voices[0].get("id")
        return None

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = DEFAULT_OUTPUT_FORMAT,
        **extra: Any,
    ) -> str:
        """Synthesize ``text`` and write audio bytes to ``output_path``.

        Returns the absolute path to the written file as a string
        (typically just echoes ``output_path``). Raises on failure —
        the dispatcher converts exceptions to the standard
        ``{success: False, error: ...}`` JSON envelope.

        Args:
            text: The text to synthesize. Already truncated to the
                provider's max length by the dispatcher.
            output_path: Absolute path where the audio file should be
                written. Parent directory is guaranteed to exist.
            voice: Voice identifier from :meth:`list_voices`, or None
                to use :meth:`default_voice`.
            model: Model identifier from :meth:`list_models`, or None
                to use :meth:`default_model`.
            speed: Optional speech-rate multiplier (1.0 = normal).
                Providers that don't support speed control should
                ignore this argument.
            format: Output audio format. Implementations should match
                the requested format when possible; if unsupported,
                pick the closest equivalent and ensure ``output_path``
                ends with the correct extension.
            **extra: Forward-compat parameters future schema versions
                may expose. Implementations should ignore unknown keys.
        """

    def stream(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        format: str = "opus",
        **extra: Any,
    ) -> Iterator[bytes]:
        """Stream synthesized audio bytes.

        Optional. Providers that don't support streaming raise
        :class:`NotImplementedError` (the default) and the dispatcher
        falls back to :meth:`synthesize` + read-whole-file.

        Args mirror :meth:`synthesize`. Default ``format`` is ``opus``
        because the primary streaming use case is voice-bubble
        delivery (Telegram et al.) which requires Opus.
        """
        raise NotImplementedError(
            f"TTS provider {self.name!r} does not implement streaming "
            "synthesis. Use synthesize() instead, or implement stream() "
            "if your backend supports it."
        )

    @property
    def voice_compatible(self) -> bool:
        """Whether output is suitable for voice-bubble delivery.

        Mirrors the ``tts.providers.<name>.voice_compatible`` field
        from PR #17843. When True, the gateway's voice-message
        delivery pipeline runs ffmpeg conversion to Opus if needed.
        When False, output is delivered as a regular audio attachment.

        Default: False (safe — providers opt in explicitly).
        """
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_output_format(value: Optional[str]) -> str:
    """Clamp an output_format value to the valid set.

    Invalid values are coerced to :data:`DEFAULT_OUTPUT_FORMAT` rather
    than rejected so the tool surface is forgiving of agent mistakes.
    """
    if not isinstance(value, str):
        return DEFAULT_OUTPUT_FORMAT
    v = value.strip().lower()
    if v in VALID_OUTPUT_FORMATS:
        return v
    return DEFAULT_OUTPUT_FORMAT

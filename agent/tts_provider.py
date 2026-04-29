"""
Text-to-Speech Provider ABC
===========================

Abstract base class for a pluggable TTS backend. Plugin implementations
subclass :class:`TtsProvider` and register via
``PluginContext.register_tts_provider()``; the ``text_to_speech`` tool
dispatches to them when ``tts.provider`` in ``config.yaml`` selects a
non-legacy name.

This mirrors :mod:`agent.image_gen_provider` in shape so the two plugin
systems stay consistent for reviewers and maintainers.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class TtsProvider(abc.ABC):
    """Abstract base class for a text-to-speech backend.

    Subclasses must implement :meth:`synthesize` and the :attr:`name`
    property. Every other hook has a sane default — override only what
    your provider actually needs to expose.
    """

    # -- identity -------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier used in ``tts.provider`` config.

        Lowercase, no spaces. Examples: ``edge``, ``elevenlabs``,
        ``volcengine``.
        """

    @property
    def display_name(self) -> str:
        """Human-readable label shown in ``hermes tools``.

        Defaults to ``name.title()``; override for proper casing / branding.
        """
        return self.name.title()

    # -- runtime gates --------------------------------------------------

    def is_available(self) -> bool:
        """Return True when this provider can service synthesis calls.

        Typically checks for required environment variables using
        ``hermes_cli.env_loader.get_env_value`` (profile-aware).
        Default: True (providers with no external dependency are always
        available).
        """
        return True

    def max_text_length(self) -> int:
        """Return the per-call character cap for this backend.

        The ``text_to_speech`` tool splits longer inputs. Provider-specific
        values allow e.g. ElevenLabs' 40k to coexist with Edge's 4k.
        """
        return 4000

    # -- picker / setup metadata ---------------------------------------

    def list_voices(self) -> List[Dict[str, Any]]:
        """Return voice-catalog entries for the ``hermes tools`` picker.

        Each entry looks like::

            {
                "id": "alloy",              # required
                "display": "Alloy",         # optional; defaults to id
                "lang": "en",               # optional
                "gender": "neutral",        # optional
                "note": "...",              # optional
            }
        """
        return []

    def default_voice(self) -> Optional[str]:
        """Return the default voice id, or None if not applicable."""
        return None

    def get_setup_schema(self) -> Dict[str, Any]:
        """Return provider metadata shown in ``hermes tools``.

        Format::

            {
                "name": "<Display Name>",
                "badge": "paid" | "free" | "preview" | "",
                "tag": "<short tagline>",
                "env_vars": [
                    {"key": "FOO_API_KEY", "prompt": "...", "url": "..."},
                    ...
                ],
            }
        """
        return {
            "name": self.display_name,
            "badge": "",
            "tag": "",
            "env_vars": [],
        }

    # -- core synthesis -------------------------------------------------

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Synthesize ``text`` and write audio to disk.

        Parameters
        ----------
        text:
            The text to render. Already validated against
            :meth:`max_text_length` by the caller.
        output_path:
            Caller-suggested path. Providers may write to this exact path
            or choose a different extension (e.g. ``.mp3`` -> ``.ogg``
            when native Opus is produced) and return the actual path in
            the result dict.
        config:
            The ``tts.<provider-name>`` sub-dictionary from ``config.yaml``.
            Empty dict when no provider-specific config exists.

        Returns
        -------
        dict
            On success::

                {
                    "success": True,
                    "file_path": <actual path written>,
                    "format": "mp3" | "wav" | "ogg",
                    "native_opus": <bool — True when the file is already
                                   opus/ogg and can skip ffmpeg Opus
                                   conversion>,
                    "voice_compatible": <bool — True when the file is
                                         ready for Telegram voice-bubble
                                         delivery>,
                }

            On failure::

                {
                    "success": False,
                    "error": <user-facing message>,
                    "error_type": "config" | "dependency" | "runtime",
                }

        Providers should **return** dicts rather than raising. Exceptions
        are caught at the dispatcher level and translated into
        ``error_type="runtime"`` results, but returning dicts lets the
        provider pick a more descriptive ``error_type``.
        """

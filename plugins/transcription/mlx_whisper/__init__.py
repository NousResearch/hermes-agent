"""MLX Whisper STT provider for macOS Apple Silicon.

Registers as a plugin-provided :class:`TranscriptionProvider` so users can
set ``stt.provider: mlx_whisper`` in ``config.yaml``.  Uses Apple's MLX
framework with GPU / Neural Engine acceleration — 3× faster than
faster-whisper on Apple Silicon.

Model selection
---------------
Short aliases resolve to ``mlx-community`` HF repos:

    tiny           → mlx-community/whisper-tiny-mlx
    base           → mlx-community/whisper-base-mlx      (default)
    small          → mlx-community/whisper-small-mlx
    medium         → mlx-community/whisper-medium-mlx
    large / large-v3 → mlx-community/whisper-large-v3-mlx
    turbo / large-v3-turbo → mlx-community/whisper-large-v3-turbo

Any HF repo id can be passed directly as the model.

Usage
-----
.. code-block:: bash

    pip install mlx-whisper
    hermes config set stt.provider mlx_whisper
    # optional:
    hermes config set stt.model mlx-community/whisper-small-mlx

Setup wizard auto-detection is handled by the provider's
:meth:`get_setup_schema` — it reports ``mlx_whisper`` as an available
option on Darwin / arm64.

References
----------
- https://github.com/ml-explore/mlx-examples/tree/main/whisper
- https://huggingface.co/mlx-community
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model alias map — short names → HF repo ids
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "mlx-community/whisper-base-mlx"

_MODEL_ALIASES: Dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _resolve_model(model: Optional[str]) -> str:
    """Translate a short alias to a full HF repo id, or pass through."""
    if not model:
        return _DEFAULT_MODEL
    key = model.strip().lower()
    return _MODEL_ALIASES.get(key, model)


def _is_apple_silicon() -> bool:
    """Return True on macOS arm64 (Apple Silicon), False otherwise."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _mlx_whisper_importable() -> bool:
    """Check whether ``mlx_whisper`` can be imported."""
    try:
        __import__("mlx_whisper")
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class MLXWhisperProvider(TranscriptionProvider):
    """Speech-to-text backend using Apple MLX Whisper (local, free).

    Implements the :class:`agent.transcription_provider.TranscriptionProvider`
    protocol — attribute ``name``, :meth:`transcribe`, :meth:`is_available`,
    :meth:`list_models`, :meth:`default_model`, and
    :meth:`get_setup_schema`.

    Users enable it by setting ``stt.provider: mlx_whisper`` in
    ``config.yaml``.
    """

    @property
    def name(self) -> str:
        """Stable short identifier — ``stt.provider`` value."""
        return "mlx_whisper"

    @property
    def display_name(self) -> str:
        """Human-readable label shown in ``hermes tools``."""
        return "MLX Whisper"

    def is_available(self) -> bool:
        """Return True on Apple Silicon with mlx_whisper installed."""
        if not _is_apple_silicon():
            return False
        return _mlx_whisper_importable()

    def list_models(self) -> List[Dict[str, Any]]:
        """Expose the curated model aliases."""
        seen: set = set()
        models: List[Dict[str, Any]] = []
        for alias, repo in _MODEL_ALIASES.items():
            if repo in seen:
                # Skip duplicates (large, large-v3 → same repo)
                continue
            seen.add(repo)
            models.append({
                "id": repo,
                "display": f"{alias} ({repo})",
            })
        return models

    def default_model(self) -> str:
        return _DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        """Return provider metadata for the tools picker / setup wizard."""
        return {
            "name": self.display_name,
            "badge": "free",
            "tag": "Local MLX Whisper — Apple Silicon GPU",
            "env_vars": [],
        }

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Transcribe *file_path* with MLX Whisper.

        Returns the standard envelope on both success and failure.
        """
        if not self.is_available():
            return {
                "success": False,
                "transcript": "",
                "error": (
                    "MLX Whisper requires macOS Apple Silicon and "
                    "mlx-whisper installed (pip install mlx-whisper)"
                ),
                "provider": self.name,
            }

        resolved = _resolve_model(model)

        try:
            import mlx_whisper  # noqa: I001  — lazy, inside transcribe
        except ImportError:
            return {
                "success": False,
                "transcript": "",
                "error": "mlx-whisper not installed (pip install mlx-whisper)",
                "provider": self.name,
            }

        try:
            result = mlx_whisper.transcribe(
                file_path,
                path_or_hf_repo=resolved,
            )
            transcript = (result.get("text") or "").strip()
            logger.info(
                "Transcribed %s via mlx_whisper (%s, %d chars)",
                Path(file_path).name, resolved, len(transcript),
            )
            return {
                "success": True,
                "transcript": transcript,
                "provider": self.name,
            }
        except Exception as exc:
            logger.error(
                "MLX Whisper transcription failed: %s", exc, exc_info=True,
            )
            return {
                "success": False,
                "transcript": "",
                "error": f"MLX Whisper transcription failed: {exc}",
                "provider": self.name,
            }


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx):
    """Register the MLX Whisper :class:`TranscriptionProvider` plugin.

    Called by the Hermes plugin loader during discovery.  The provider
    is only registered on macOS (darwin) — on other platforms the
    registration is silently skipped.
    """
    if not _is_apple_silicon():
        logger.debug(
            "mlx_whisper plugin: skipping registration (not macOS arm64)"
        )
        return

    try:
        ctx.register_transcription_provider(MLXWhisperProvider())
        logger.info("mlx_whisper transcription provider registered")
    except Exception as exc:
        logger.warning(
            "mlx_whisper plugin registration failed: %s", exc,
        )

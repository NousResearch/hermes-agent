"""``@file:`` audio attachments -> transcript, when ``stt.transcribe_audio_files`` is on.

Desktop, TUI and CLI attach a file as an ``@file:`` reference; an audio file then reached the
model as a binary "use your tools" stub, so a profile without a transcription tool got no words.
The gateway's inbound media path honours the same flag for platform uploads. Inspired by
ChatGPT's "upload audio, get a transcript" (Oct 6 2026 release notes).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent.model_metadata import estimate_tokens_rough


def _audio_transcription_enabled() -> bool:
    from tools.transcription_tools import _load_stt_config, is_stt_enabled
    from utils import is_truthy_value
    cfg = _load_stt_config()
    return is_stt_enabled(cfg) and is_truthy_value(cfg.get("transcribe_audio_files", False), default=False)


def _is_audio_path(path: Path) -> bool:
    from tools.transcription_common import SUPPORTED_FORMATS
    return path.suffix.lower() in SUPPORTED_FORMATS


def _transcript_block(ref, path: Path) -> tuple[str | None, str | None] | None:
    """``(warning, block)`` for an audio ``@file:`` ref, or ``None`` to fall through to the
    ordinary path expansion (flag off, not audio, ranged ref, or transcription failed: the
    binary stub then still hands the model the path)."""
    if ref.line_start is not None or not _is_audio_path(path) or not path.is_file():
        return None
    if not _audio_transcription_enabled():
        return None
    from tools.transcription_tools import transcribe_audio
    result = transcribe_audio(str(path), source="context_reference")
    transcript = (result.get("transcript") or "").strip() if result.get("success") else ""
    if not transcript:
        return None
    from agent.context_references import _agent_visible_path
    body = f'Transcript of audio file `{_agent_visible_path(path)}`:\n"{transcript}"'
    return None, f"🎙️ {ref.raw} ({estimate_tokens_rough(body)} tokens)\n{body}"


async def expand_audio_reference(ref, cwd: Path, *, allowed_root: Path | None):
    """Transcript expansion for an audio ``@file:`` ref, else ``None``. Path resolution and the
    credential-path refusal are the same calls the text path makes, run before any upload."""
    if ref.kind != "file" or not _is_audio_path(Path(ref.target)):
        return None
    from agent.context_references import _ensure_reference_path_allowed, _resolve_path
    path = _resolve_path(cwd, ref.target, allowed_root=allowed_root)
    _ensure_reference_path_allowed(path)
    # STT can take seconds to minutes: keep it off the event loop (to_thread copies the
    # context, so a served profile's HERMES_HOME override picks that profile's stt config).
    return await asyncio.to_thread(_transcript_block, ref, path)

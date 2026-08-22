"""Voice-server partial-stream helpers.

These helpers live here (not inside ``gateway/run.py``) so the voice-only
turn-id reconciliation, visible-text normalization, and stream-cleanup
glue do not pollute the general turn-dispatch path. ``run.py`` imports
them as private ``_voice_*`` aliases at the callsites that already use
them; behavior is unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from utils import is_truthy_value

logger = logging.getLogger(__name__)


def partial_llm_streaming_enabled(config: dict, adapter: Any) -> bool:
    """Return true unless the voice server platform explicitly disables LLM deltas."""
    adapter_extra = getattr(getattr(adapter, "config", None), "extra", None)
    if isinstance(adapter_extra, dict) and "partial_llm_streaming" in adapter_extra:
        return is_truthy_value(adapter_extra.get("partial_llm_streaming"), default=True)

    for root in (
        config.get("gateway") if isinstance(config.get("gateway"), dict) else None,
        config,
    ):
        if not isinstance(root, dict):
            continue
        platforms = root.get("platforms")
        if not isinstance(platforms, dict):
            continue
        voice_cfg = platforms.get("voice_server")
        if not isinstance(voice_cfg, dict):
            continue
        extra = voice_cfg.get("extra")
        if isinstance(extra, dict) and "partial_llm_streaming" in extra:
            return is_truthy_value(extra.get("partial_llm_streaming"), default=True)
        if "partial_llm_streaming" in voice_cfg:
            return is_truthy_value(voice_cfg.get("partial_llm_streaming"), default=True)
    return True


def turn_ids_from_result(result: Any) -> list[str]:
    if not isinstance(result, dict):
        return []
    turn_ids: list[str] = []
    raw_turn_ids = result.get("voice_turn_ids")
    if isinstance(raw_turn_ids, (list, tuple)):
        for raw_turn_id in raw_turn_ids:
            turn_id = str(raw_turn_id or "").strip()
            if turn_id and turn_id not in turn_ids:
                turn_ids.append(turn_id)
    turn_id = str(result.get("voice_turn_id") or "").strip()
    if turn_id and turn_id not in turn_ids:
        turn_ids.append(turn_id)
    return turn_ids


def visible_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("[[audio_as_voice]]", "")
    text = text.replace("[[as_document]]", "")
    text = re.sub(r"MEDIA:\s*\S+", "", text)
    return text.strip()


def assistant_message_matches_voice_final(msg: Any, visible_final: str) -> bool:
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return False
    return bool(visible_final) and visible_text(msg.get("content")) == visible_final


def stamp_voice_turn_id_on_final_assistant(
    messages: Any,
    *,
    visible_final: str,
    voice_turn_id: str,
) -> bool:
    if not isinstance(messages, list) or not voice_turn_id:
        return False
    for msg in reversed(messages):
        if assistant_message_matches_voice_final(msg, visible_final):
            msg["voice_turn_id"] = voice_turn_id
            return True
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and not msg.get("tool_calls"):
            msg["voice_turn_id"] = voice_turn_id
            return True
    return False


def cleanup_assistant_stream(adapter: Any, metadata: Any) -> None:
    cleanup = getattr(adapter, "cleanup_assistant_stream", None)
    if callable(cleanup):
        try:
            cleanup(metadata=metadata if isinstance(metadata, dict) else None)
        except Exception as exc:
            logger.debug("voice stream cleanup failed: %s", exc)


def scrub_voice_turn_id_from_result(result: Any, turn_id: str) -> bool:
    if not isinstance(result, dict) or not turn_id:
        return False
    changed = False
    if str(result.get("voice_turn_id") or "") == turn_id:
        result.pop("voice_turn_id", None)
        changed = True
    raw_turn_ids = result.get("voice_turn_ids")
    if isinstance(raw_turn_ids, list):
        filtered_turn_ids = [
            existing for existing in raw_turn_ids if str(existing or "") != turn_id
        ]
        if len(filtered_turn_ids) != len(raw_turn_ids):
            if filtered_turn_ids:
                result["voice_turn_ids"] = filtered_turn_ids
            else:
                result.pop("voice_turn_ids", None)
            changed = True
    for msg in result.get("messages") or []:
        if isinstance(msg, dict) and str(msg.get("voice_turn_id") or "") == turn_id:
            msg.pop("voice_turn_id", None)
            changed = True
    return changed

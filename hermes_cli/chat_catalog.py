"""Admit only chat models into shared chat catalogs.

Generation models are identified by the capability type or name shape the
catalog already publishes. Missing, malformed, or unknown metadata fails open.
No single model id is special-cased.
"""

from __future__ import annotations

import re
from typing import Any

# Structured catalog types that mean "this row is not a chat completion".
_GENERATION_TYPES = frozenset({
    "image",
    "image_generation",
    "image-generation",
    "image-gen",
    "video",
    "video_generation",
    "video-generation",
    "video-gen",
})

_GENERATION_OUTPUTS = frozenset({"image", "video"})
_CHAT_OUTPUTS = frozenset({"text", "chat"})

# Separator-qualified generation tokens, plus the generation id tokens the
# DeepInfra chat catalog already excludes. ``vision`` / ``vl`` are not tokens
# here — those remain chat models.
_GENERATION_ID_RE = re.compile(
    r"(?i)(?:^|[/_.-])(?:"
    r"image(?:[/_.-]gen(?:eration)?)?|"
    r"video(?:[/_.-]gen(?:eration)?)?|"
    r"text[/_.-]to[/_.-](?:image|video)|"
    r"stable-diffusion|sdxl|flux"
    r")(?=$|[/_.-])"
)

# Ids a live catalog item already proved are generation-only, even when the
# id itself has no generation token. Process-local: the proof came from this
# process's catalog parse.
_seen_generation_ids: set[str] = set()


def note_generation_model(model_id: Any) -> None:
    key = str(model_id or "").strip().lower()
    if key:
        _seen_generation_ids.add(key)


def model_id_is_generation(model_id: Any) -> bool:
    """Whether *model_id* names an image/video generation surface."""
    return bool(_GENERATION_ID_RE.search(str(model_id or "").strip()))


def _tag_list_is_generation(tags: Any) -> bool:
    if not isinstance(tags, list):
        return False
    return any(str(tag or "").strip().lower() in _GENERATION_TYPES for tag in tags)


def _output_modalities_are_generation(outputs: Any) -> bool:
    if not isinstance(outputs, list) or not outputs:
        return False
    kinds = {str(item or "").strip().lower() for item in outputs if str(item or "").strip()}
    return bool(kinds & _GENERATION_OUTPUTS) and not bool(kinds & _CHAT_OUTPUTS)


def catalog_item_is_generation(item: Any) -> bool:
    """Fail open unless this catalog row's own type or name proves it is generation-only."""
    if not isinstance(item, dict):
        return False
    if model_id_is_generation(item.get("id")):
        return True
    capabilities = item.get("capabilities")
    if isinstance(capabilities, dict):
        model_type = str(capabilities.get("type") or "").strip().lower()
        if model_type in _GENERATION_TYPES:
            return True
        if _tag_list_is_generation(capabilities.get("tags")):
            return True
    top_type = item.get("type")
    if isinstance(top_type, str) and top_type.strip().lower() in _GENERATION_TYPES:
        return True
    if _tag_list_is_generation(item.get("tags")):
        return True
    architecture = item.get("architecture")
    if isinstance(architecture, dict) and _output_modalities_are_generation(architecture.get("output_modalities")):
        return True
    metadata = item.get("metadata")
    if isinstance(metadata, dict) and _tag_list_is_generation(metadata.get("tags")):
        return True
    return False


def note_catalog_item(item: Any) -> bool:
    """Remember a generation row and report whether it must leave the chat catalog."""
    if not catalog_item_is_generation(item):
        return False
    if isinstance(item, dict):
        note_generation_model(item.get("id"))
    return True


def is_known_non_chat_model(model_id: Any) -> bool:
    """True when the id's name shape, or a catalog item already parsed, says it is not chat."""
    mid = str(model_id or "").strip()
    return model_id_is_generation(mid) or mid.lower() in _seen_generation_ids


def without_generation_models(models):
    """Drop generation ids, preserving a list subclass (curated-fallback marker)."""
    if not models:
        return models
    kept = [model for model in models if not is_known_non_chat_model(model)]
    if len(kept) == len(list(models)):
        return models
    try:
        return type(models)(kept)
    except TypeError:
        return kept


def chat_catalog_ids(items: Any) -> list[str]:
    """Model ids from a ``/models`` payload, minus generation rows."""
    if not isinstance(items, list):
        return []
    ids: list[str] = []
    for item in items:
        if not isinstance(item, dict) or "id" not in item:
            continue
        if note_catalog_item(item):
            continue
        ids.append(item["id"])
    return ids
